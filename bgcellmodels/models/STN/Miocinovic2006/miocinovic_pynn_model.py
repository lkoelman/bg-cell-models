import math, logging

import numpy as np
import neuron
import bluepyopt.ephys as ephys
import pyNN.neuron

from bgcellmodels.common import electrotonic, nrnutil, treeutils, logutils
from bgcellmodels.morphology import morph_ni
from bgcellmodels.models.STN import GilliesWillshaw as gillies
from bgcellmodels.models.STN import Miocinovic2006 as miocinovic
from bgcellmodels.models.axon.mcintyre2002 import AxonMcintyre2002
from bgcellmodels.extensions.pynn import ephys_models as ephys_pynn


h = neuron.h
gillies.load_mechanisms()

logger = logging.getLogger('miocinovic_model')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format=logutils.DEFAULT_FORMAT)


class StnMorphModel(ephys_pynn.PynnCellModelBase):
    """
    Morphological STN model using any morphology in combination
    with Miocinovic (2006) cell template.

    DEVNOTES
    --------

    Called by ID._build_cell() defined in module pyNN/neuron/simulator.py
    
    - Population._create_cells()
        - Population.cell_type.model(**cell_parameters)
            - PynnCellModelBase.__init__()
                - instantiate()
    """

    # Combined with celltype.receptors in MorphCellType constructor
    # to make celltype.receptor_types in format 'region.receptor'
    regions = ['proximal', 'distal']

    def __init__(self, **kwargs):
        """
        Custom handling of parameters
        """
        cell_param_names = StnMorphType.default_parameters.keys() + \
                           StnMorphType.extra_parameters.keys()
        
        # Make parameters accessible as attributes
        for param_name, param_value in kwargs.iteritems():
            if param_name in cell_param_names:
                setattr(self, param_name, param_value)
            else:
                raise ValueError('Unknown argument "{}":'.format(
                        param_name), param_value)

        self.instantiate()
        self._init_synapses()

        # Attributes required by PyNN
        # TODO: adapt to make NetCon created by Projection come from axon end
        self.source_section = self.icell.soma[0]
        self.source = self.icell.soma[0](0.5)._ref_v
        
        self.rec = h.NetCon(self.source, None,
                            self.get_threshold(), 0.0, 0.0,
                            sec=self.source_section)
        self.spike_times = h.Vector(0) # see pyNN.neuron.recording.Recorder._record()
        self.traces = {}
        self.recording_time = False


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator.
        """
        # Get arguments from __init__
        template_name = self.template_name
        morphology_path = self.morphology_path
        streamlines_path = self.streamlines_path
        axon_class = self.axon_class # e.g. AxonMcintyre2002
        if sim is None:
            sim = ephys_pynn.ephys_sim_from_pynn()

        # Get the Hoc template
        miocinovic.load_template(template_name) # xopen -> loads once
        template_constructor = getattr(h, template_name)
        
        # Instantiate template
        self.icell = icell = template_constructor()
        icell.with_extracellular = 0

        # Load morphology into template
        morphology = ephys.morphologies.NrnFileMorphology(morphology_path, do_replace_axon=False)
        morphology.instantiate(sim=sim, icell=icell)

        # Setup biophysical properties
        icell.del_unused_sections()
        icell.insert_biophys()
        nseg_extra = electrotonic.set_min_nseg_hines(icell.all, f_lambda=100.0)
        icell.set_biophys_spatial()

        # Create and append axon
        if streamlines_path is not None:
            self._init_axon(axon_class, streamlines_path)


    def _init_axon(self, axon_class, streamlines_path):
        """
        Create and append axon.

        @post   self.axon contains references to axonal sections.
        """
        axon_builder = axon_class(logger=logger,
                            without_extracellular=self.without_extracellular)
        tracks_coords = morph_ni.load_streamlines(
                            streamlines_path, max_num=1, min_length=2.0)
        axon_coords = tracks_coords[0]

        # Build axon
        axonal_secs = list(self.icell.axonal)
        if len(axonal_secs) > 0:
            # Attach axon to axon stub/AIS if present
            axon_terminal_secs = treeutils.leaf_sections(axonal_secs[0], subtree=True)
            assert len(axon_terminal_secs) == 1
            axon_parent_sec = axon_terminal_secs[0]
        else:
            # Attach axon directly to soma
            axon_parent_sec = self.icell.soma[0]

        axon = axon_builder.build_along_streamline(axon_coords,
                    terminate='nodal_cutoff', interp_method='arclength',
                    parent_cell=self.icell, parent_sec=axon_parent_sec,
                    connection_method='translate_axon_start')
    
        self.axon = axon


    def _post_build(self, population, pop_index):
        """
        Hook called after Population._create_cells() -> ID._build_cell()
        is executed.

        @override   EphysModelWrapper._post_build()
        """
        self._init_memb_noise(population, pop_index)


    def _init_memb_noise(self, population, pop_index):
        # Insert membrane noise
        if self.membrane_noise_std > 0:
            # Configure RNG to generate independent stream of random numbers.
            num_picks = int(pyNN.neuron.state.duration / pyNN.neuron.state.dt)
            seed = (1e4 * population.pop_gid) + pop_index
            rng, init_rng = nrnutil.independent_random_stream(
                                    num_picks, pyNN.neuron.state.mcellran4_rng_indices,
                                    force_low_index=seed)
            rng.normal(0, 1)
            self.noise_rng = rng
            self.noise_rng_init = init_rng

            soma = self.icell.soma[0]
            self.noise_stim = stim = h.ingauss2(soma(0.5))
            std_scale =  1e-2 * sum((seg.area() for seg in soma)) # [mA/cm2] to [nA]
            stim.mean = 0.0
            stim.stdev = self.membrane_noise_std * std_scale
            stim.noiseFromRandom(rng)
        else:
            def fdummy():
                pass
            self.noise_rng = None
            self.noise_rng_init = fdummy


    def _init_synapses(self):
        """
        Initialize synapses on this neuron.

        @override   EphysModelWrapper._init_synapses()
        """
        # Create data structure for synapses
        super(StnMorphModel, self)._init_synapses()

        # Indicate to Connector that we don't allow multiple NetCon per synapse
        self.allow_synapse_reuse = False

        # Sample cell regions
        soma = self.icell.soma[0]
        h.distance(0, 0.5, sec=soma) # reference for distance measurement
        is_proximal = lambda seg: h.distance(1, seg.x, sec=seg.sec) < 120.0
        is_distal = lambda seg: h.distance(1, seg.x, sec=seg.sec) >= 100.0

        # Sample each region uniformly and place synapses there
        synapse_spacing = 0.25 # [um]
        target_secs = list(self.icell.somatic) + list(self.icell.basal) + \
                      list(self.icell.apical)
        self._cached_region_segments = {}
        self._cached_region_segments['proximal'] = []
        self._cached_region_segments['distal'] = []
        for sec in target_secs:
            nsyn = math.ceil(sec.L / synapse_spacing)
            for i in xrange(int(nsyn)):
                seg = sec((i+1.0)/nsyn)
                if is_proximal(seg):
                    self._cached_region_segments['proximal'].append(seg)
                if is_distal(seg):
                    self._cached_region_segments['distal'].append(seg)


    def get_synapses(self, region, receptors, num_contacts, **kwargs):
        """
        Get synapse in subcellular region for given receptors.
        Called by Connector object to get synapse for new connection.

        @override   PynnCellModelBase.get_synapse()
        """
        syns = self.make_synapses_cached_region(region, receptors, 
                                                num_contacts, **kwargs)
        synmap_key = tuple(sorted(receptors))
        self._synapses[region].setdefault(synmap_key, []).extend(syns)
        return syns


    def memb_init(self):
        """
        Initialization function required by PyNN.

        @override     EphysModelWrapper.memb_init()
        """
        super(StnMorphModel, self).memb_init()

        self.noise_rng_init()

    def get_threshold(self):
        """
        Get spike threshold for soma membrane potential (used for NetCon)
        """
        return -10.0


class GilliesSwcModel(StnMorphModel):
    """
    Morphological SWC model that uses the channel density distributions
    and morphology from the original Gillies (2005) model.
    """

    def __init__(self, **kwargs):
        super(GilliesSwcModel, self).__init__(**kwargs)


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator.
        """
        # Get arguments from __init__
        template_name = self.template_name
        morphology_path = self.morphology_path
        streamlines_path = self.streamlines_path
        axon_class = self.axon_class # e.g. AxonMcintyre2002
        if sim is None:
            sim = ephys_pynn.ephys_sim_from_pynn()

        # Get the Hoc template
        miocinovic.load_template(template_name) # xopen -> loads once
        template_constructor = getattr(h, template_name)
        
        # Instantiate template
        self.icell = icell = template_constructor()
        icell.with_extracellular = not self.without_extracellular

        # Load morphology into template
        morphology = ephys.morphologies.NrnFileMorphology(morphology_path, do_replace_axon=False)
        morphology.instantiate(sim=sim, icell=icell)

        # Setup biophysical properties
        ais_diam = 1.4 # nodal diam from McIntyre axon
        ais_relative_length = 0.2
        icell.create_AIS(ais_diam, ais_relative_length, 0, 0.0)
        icell.del_unused_sections()
        icell.insert_biophys()

        # Instead of cel.set_biophys_spatial(), load channel density
        # distributions from file.
        self._init_gbar()

        # Create and append axon
        if streamlines_path is not None:
            self._init_axon(axon_class, streamlines_path)


    def _init_gbar(self):
        """
        Load channel conductances from Gillies & Wilshaw data files.
        """
        sec_arrays_lists = {'soma': 'somatic', 'dend': 'basal'}
        gbar_names = ['gk_KDR', 'gcaT_CaT', 'gcaL_HVA', 'gcaN_HVA', 
                        'gk_sKCa', 'gk_Kv31', 'gk_Ih']
        
        for gbar_name in gbar_names:
            gbar_mat = gillies.load_gbar_dist(gbar_name)
            for secarray_name, seclist_name in sec_arrays_lists.items():
                for i, sec in enumerate(getattr(self.icell, seclist_name)):
                    tree_index, array_index = miocinovic.swc_to_gillies_index(
                                                i, secarray_name=secarray_name)
                    # Get samples for section
                    sample_mask = (gbar_mat[:,0] == tree_index) & (gbar_mat[:,1] == array_index)
                    gbar_samples = gbar_mat[sample_mask]
                    if gbar_samples.ndim == 1:
                        gbar_samples = gbar_samples[np.newaxis, :]
                    xvals_gvals = gbar_samples[:, [2, 3]]
                    # Choose closest sample to assign gbar (nseg discretization mismatch)
                    for seg in sec:
                        i_close = np.abs(xvals_gvals[:,0] - seg.x).argmin()
                        setattr(seg, gbar_name, xvals_gvals[i_close, 1])
                    # for seg_x, gbar_val in xvals_gvals:
                    #     setattr(sec(seg_x), gbar_name, gbar_val)


class StnMorphType(ephys_pynn.MorphCellType):
    """
    Cell type associated with a PyNN population.
    """

    model = StnMorphModel

    # NOTE: default_parameters is used to make 'schema' for checking & converting datatypes
    default_parameters = {
        # 'calculate_lfp': False,
        # 'lfp_sigma_extracellular': 0.3,
        # 'lfp_electrode_x': 100.0,
        # 'lfp_electrode_y': 100.0,
        # 'lfp_electrode_z': 100.0,
        # 'tau_m_scale': 1.0,
        # 'membrane_noise_std': 0.0,
        'max_num_gpe_syn': 19,
        'max_num_ctx_syn': 30,
        'max_num_stn_syn': 10,
    }

    # NOTE: extra_parameters are passed to model.__init__(), can contain strings,
    #       but cannot be passed as argument to cell type unless hack below is used
    extra_parameters = {
        'template_name': 'STN_morph_arcdist',
        'morphology_path': 'placeholder/path',
        'streamlines_path': 'placeholder/path',
        'axon_class': AxonMcintyre2002,
        'without_extracellular': False,
        'default_GABA_mechanism': 'GABAsyn2',
        'default_GLU_mechanism': 'GLUsyn',
    }

    default_initial_values = {'v': -65.0}
    # recordable = ['spikes', 'v', 'lfp']

    # Combined with self.model.regions by MorphCellType constructor
    receptor_types = ['AMPA', 'NMDA', 'AMPA+NMDA',
                      'GABAA', 'GABAB', 'GABAA+GABAB']

    def __init__(self, **parameters):
        """
        Trick for allowing extra parameters as kwargs.
        """
        self.extra_parameters = {
            k: parameters.pop(k, v) for k,v in StnMorphType.extra_parameters.items()
        }
        ephys_pynn.MorphCellType.__init__(self, **parameters)


    def can_record(self, variable):
        """
        Override or it uses pynn.neuron.record.recordable_pattern.match(variable)
        """
        return super(StnMorphType, self).can_record(variable)


################################################################################
# Testing
################################################################################


def test_simulate_population(export_locals=False):
    """
    Test PyNN model creation, running, and recording.

    @see    Based on test in:
            https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
    """
    from pyNN.random import RandomDistribution
    from pyNN.utility import init_logging
    import pyNN.neuron as nrn

    init_logging(logfile=None, debug=True)
    nrn.setup()

    # GPe cell population
    parameters = {
        'template_name': 'STN_morph_arcdist',
        'morphology_path': '/home/luye/workspace/bgcellmodels/bgcellmodels/models/STN/Miocinovic2006/morphologies/type1RD_axonless-with-AIS.swc',
        'streamlines_path': '/home/luye/Documents/mri_data/Waxholm_rat_brain_atlas/WHS_DTI_v1_ALS/S56280_track_filter-ROI-STN.tck',
    }
    cell_type = StnMorphType(**parameters)
    p1 = nrn.Population(5, cell_type)

    # Modify population
    # p1.rset('Ra', RandomDistribution('uniform', low=100., high=300.))
    p1.initialize(v=-63.0)

    # Stimulation electrode
    current_source = nrn.StepCurrentSource(times=[50.0, 110.0, 150.0, 210.0],
                                           amplitudes=[0.4, 0.6, -0.2, 0.2])
    p1.inject(current_source)

    # Stimulation spike source
    p2 = nrn.Population(1, nrn.SpikeSourcePoisson(rate=100.0))
    connector = nrn.AllToAllConnector()
    syn = nrn.StaticSynapse(weight=0.1, delay=2.0)
    prj_alpha = nrn.Projection(p2, p1, connector, syn, 
                               receptor_type='distal.AMPA')

    # Recording
    # p1.record(['apical(1.0).v', 'soma(0.5).ina'])
    nrn.run(250.0)

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


def test_instantiate_cell(export_locals=False):
    """
    Test cell instantiation without PyNN.
    """
    cell = StnMorphModel()
    sim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
    cell.instantiate(sim=sim)

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


if __name__ == '__main__':
    test_simulate_population(export_locals=True)