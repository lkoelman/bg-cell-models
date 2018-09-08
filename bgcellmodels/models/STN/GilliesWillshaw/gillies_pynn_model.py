"""
PyNN compatible cell models for Gillies STN cell model.

@author     Lucas Koelman

@date       15/03/2018

"""

import neuron
h = neuron.h
import numpy as np

# Load NEURON libraries, mechanisms
import os, os.path
script_dir = os.path.dirname(__file__)
neuron.load_mechanisms(os.path.join(script_dir, 'mechanisms'))

# Load STN cell Hoc libraries
prev_cwd = os.getcwd()
os.chdir(script_dir)
# os.environ['HOC_LIBRARY_PATH'] = os.environ.get('HOC_LIBRARY_PATH', '') + ':' + script_dir
h.xopen("gillies_createcell.hoc") # instantiates all functions & data structures on Hoc object
os.chdir(prev_cwd)

import bgcellmodels.extensions.pynn.ephys_models as ephys_pynn
from bgcellmodels.extensions.pynn.ephys_locations import SomaDistanceRangeLocation
from bgcellmodels.common.stdutil import dotdict
from bgcellmodels.common import treeutils, nrnutil #, logutils
import pyNN.neuron as nrnsim
import lfpsim # loads Hoc functions

# Debug messages
# logutils.setLogLevel('quiet', [
#     'bluepyopt.ephys.parameters', 
#     'bluepyopt.ephys.mechanisms', 
#     'bluepyopt.ephys.morphologies'])


def define_locations():
    """
    Define locations / regions on the cell that will function as the target
    of synaptic connections.

    @return     list(SomaDistanceRangeLocation)
                List of location / region definitions.
    """

    proximal_dend = SomaDistanceRangeLocation(
        name='proximal_dend',
        seclist_name='basal',
        min_distance=5.0,
        max_distance=100.0)

    distal_dend = SomaDistanceRangeLocation(
        name='distal_dend',
        seclist_name='basal',
        min_distance=100.0,
        max_distance=1000.0)

    return [proximal_dend, distal_dend]


class StnCellModel(ephys_pynn.EphysModelWrapper):
    """
    Model class for Gillies STN cell.

    NOTES
    ------

    - instantiated using Population.cell_type.model(**parameters) 
      and assigned to ID._cell

        - parameters are those passed to the CellType on creation

    - !!! Don't forget to set initial ion concentrations globally.


    EXAMPLE
    -------

    >>> cell = StnCellModel()
    >>> nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
    >>> icell = cell.instantiate(sim=nrnsim)
    
    """

    # _ephys_locations = define_locations()

    # Combined with celltype.receptors in EphysCellType constructor
    # to make celltype.receptor_types in format 'region.receptor'
    regions = ['proximal', 'distal']

    # Must define 'default_parameters' in associated cell type
    parameter_names = [
        # See workaround for non-numerical parameters
        # 'default_GABA_mechanism',
        # 'default_GLU_mechanism',
        'calculate_lfp',
        'lfp_sigma_extracellular',
        'lfp_electrode_x',
        'lfp_electrode_y',
        'lfp_electrode_z',
        'tau_m_scale',
        'membrane_noise_std',
        'max_num_gpe_syn',
        'max_num_ctx_syn',
        'max_num_stn_syn',
    ]

    # FIXME: workaround: set directly as property on the class because
    # PyNN only allows numerical parameters
    default_GABA_mechanism = 'GABAsyn2'
    default_GLU_mechanism = 'GLUsyn'

    # Related to PyNN properties
    _mechs_params_dict = {
        'STh':  ['gpas'],
        'Na':   ['gna'],
        'NaL':  ['gna'],
        'KDR':  ['gk'],
        'Kv31': ['gk'],
        'sKCa': ['gk'],
        'Ih':   ['gk'],
        'CaT':  ['gcaT'],
        'HVA':  ['gcaL', 'gcaN'],
        'Cacum':[],
    }
    rangevar_names = [p+'_'+m for m,params in _mechs_params_dict.iteritems() for p in params]
    gleak_name = 'gpas_STh'
    tau_m_scaled_regions = ['somatic', 'basal', 'apical', 'axonal']
    rangevar_scaled_regions = {}
    for rangevar in rangevar_names:
        parameter_names.append(rangevar + '_scale')


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator

        @override       ephys.models.CellModel.instantiate()
                        
                        Since the wrapped model is a pure Hoc model completely
                        defined by its Hoc template, i.e. without ephys
                        morphology, parameters, or mechanisms definitions,
                        we have to override instantiate().
        """
        cell_idx = h.make_stn_cell_global()
        cell_idx = int(cell_idx)
        self.icell = h.SThcells[cell_idx]


    def _post_instantiate(self):
        """
        Post-instantiation setup code. Inserts noise sources if requested.
        """
        # Insert membrane noise
        if self.membrane_noise_std > 0:
            # Configure RNG to generate independent stream of random numbers.
            num_picks = int(nrnsim.state.duration / nrnsim.state.dt)
            rng, init_rng = nrnutil.independent_random_stream(
                                    num_picks, nrnsim.state.mcellran4_rng_indices,
                                    start_low_index=nrnsim.state.rank_rng_seed)
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


    def memb_init(self):
        """
        Initialization function required by PyNN.

        @override     EphysModelWrapper.memb_init()
        """
        for sec in self.icell.all:
            h.ion_style("na_ion",1,2,1,0,1, sec=sec)
            h.ion_style("k_ion",1,2,1,0,1, sec=sec)
            h.ion_style("ca_ion",3,2,1,1,1, sec=sec)

        self.noise_rng_init()


    def get_threshold(self):
        """
        Get spike threshold for soma membrane potential (used for NetCon)
        """
        return -10.0


    def _init_synapses(self):
        """
        Initialize synapses on this neuron.

        @override   EphysModelWrapper._init_synapses()
        """
        # Indicate to Connector that we don't allow multiple NetCon per synapse
        self.allow_synapse_reuse = False

        # Sample each region uniformly and place synapses there
        soma = self.icell.soma[0]
        synapse_spacing = 0.25
        segment_gen = treeutils.ascend_with_fixed_spacing(
                            soma(0.5), synapse_spacing)
        uniform_segments = [seg for seg in segment_gen]

        # Sample cell regions
        h.distance(0, 0.5, sec=soma) # reference for distance measurement

        is_proximal = lambda seg: h.distance(1, seg.x, sec=seg.sec) < 120.0
        proximal_segments = [seg for seg in uniform_segments if is_proximal(seg)]

        is_distal = lambda seg: h.distance(1, seg.x, sec=seg.sec) >= 100.0
        distal_segments = [seg for seg in uniform_segments if is_distal(seg)]

        # Synapses counts are fixed
        num_gpe_syn = self.max_num_gpe_syn
        num_ctx_syn = self.max_num_ctx_syn
        num_stn_syn = self.max_num_stn_syn
        rng = np.random # TODO: make from base seed + self ID
        
        proximal_indices = rng.choice(len(proximal_segments), 
                                num_gpe_syn, replace=False)

        distal_indices = rng.choice(len(distal_segments), 
                                num_ctx_syn+num_stn_syn, replace=False)

        # Fill synapse lists
        self._synapses = {}
        self._synapses['proximal'] = prox_syns = {}
        self._synapses['distal'] = dist_syns = {}

        # Get constuctors for NEURON synapse mechanisms
        make_gaba_syn = getattr(h, self.default_GABA_mechanism)
        make_glu_syn = getattr(h, self.default_GLU_mechanism)

        prox_syns[('GABAA', 'GABAB')] = prox_gaba_syns = []
        for seg_index in proximal_indices:
            syn = make_gaba_syn(proximal_segments[seg_index])
            prox_gaba_syns.append(dotdict(synapse=syn, used=0,
                mechanism=self.default_GABA_mechanism))
        
        dist_syns[('AMPA', 'NMDA')] = dist_glu_syns = []
        for seg_index in distal_indices:
            syn = make_glu_syn(distal_segments[seg_index])
            dist_glu_syns.append(dotdict(synapse=syn, used=0,
                mechanism=self.default_GLU_mechanism))


    def _update_position(self, xyz):
        """
        Called when the cell's position is changed, e.g. when changing 
        the space/structure of the parent Population.

        @effect     Adds xyz to all coordinates of the root sections and then
                    calls h.define_shape() so that whole tree is translated.
        """
        if self.calculate_lfp:
            # translate the root section and re-define shape to translate entire cell
            # source_ref = h.SectionRef(sec=self.source_section)
            # root_sec = source_ref.root
            root_sec = self.icell.soma[0]
            # root_sec.push() # 3D functions operate on CAS

            def get_coordinate(i, sec):
                return [h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)]

            # initial define shape to make sure 3D info is present
            h.define_shape(sec=root_sec)
            # root_origin = get_coordinate(0, root_sec)

            # FIXME: uncomment cell position update after test
            # # Translate each point of root_sec so that first point is at xyz
            # for i in range(int(h.n3d(sec=root_sec))):
            #     old_xyz = get_coordinate(i, root_sec)
            #     old_diam = h.diam3d(i, sec=root_sec)
            #     h.pt3dchange(i,
            #         xyz[0]-root_origin[0]+old_xyz[0],
            #         xyz[1]-root_origin[1]+old_xyz[0],
            #         xyz[2]-root_origin[2]+old_xyz[0],
            #         old_diam)

            # # redefine shape to translate tree based on updated root position
            # h.define_shape(sec=root_sec)
            # # h.pop_section()


    def _init_lfp(self):
        """
        Initialize LFP sources for this cell.

        @pre        Named parameter 'lfp_electrode_coords' must be passed
                    to cell

        @return     lfp_tracker : nrn.POINT_PROCESS
                    Object with recordable variable 'summed' that represents 
                    the cell's summed LFP contributions
        """
        if not self.calculate_lfp:
            self.lfp = None
            return
        
        # define_shape() called in _update_position()
        # h.define_shape(sec=self.icell.soma[0])
        coords = h.Vector([self.lfp_electrode_x, 
                           self.lfp_electrode_y,
                           self.lfp_electrode_z])
        sigma = self.lfp_sigma_extracellular

        # Method A: using LFP summator object
        self.lfp_tracker = h.LfpTracker(
                                self.icell.soma[0], True, "PSA", sigma, 
                                coords, self.icell.somatic, self.icell.basal)


class StnCellType(ephys_pynn.EphysCellType):
    """
    Encapsulates an STN model described as a BluePyOpt Ephys model 
    for interoperability with PyNN.
    """

    # The encapsualted model available as class attribute 'model'
    model = StnCellModel

    # NOTE: default_parameters is used to make 'schema' for checking & converting datatypes
    default_parameters = {
        # 'default_GABA_mechanism': 'GABAsyn',
        # 'default_GLU_mechanism': 'GLUsyn',
        'calculate_lfp': False,
        'lfp_sigma_extracellular': 0.3,
        'lfp_electrode_x': 100.0,
        'lfp_electrode_y': 100.0,
        'lfp_electrode_z': 100.0,
        'tau_m_scale': 1.0,
        'membrane_noise_std': 0.0,
        'max_num_gpe_syn': 19,
        'max_num_ctx_syn': 30,
        'max_num_stn_syn': 10,
    }
    # extra_parameters = {}
    # default_initial_values = {'v': -65.0}
    # recordable = ['spikes', 'v', 'lfp']

    # Combined with self.model.regions by EphysCellType constructor
    receptor_types = ['AMPA', 'NMDA', 'AMPA+NMDA',
                      'GABAA', 'GABAB', 'GABAA+GABAB']

    def get_schema(self):
        """
        Get mapping of parameter names to allowed parameter types.
        """
        # Avoids specifying default values for each scale parameter and
        # thereby calling the property setter for each of them
        schema = super(StnCellType, self).get_schema()
        schema.update({v+'_scale': float for v in self.model.rangevar_names})
        return schema


    def can_record(self, variable):
        """
        Override or it uses pynn.neuron.record.recordable_pattern.match(variable)
        """
        return super(StnCellType, self).can_record(variable)


def test_stn_cells_multiple(export_locals=True):
    """
    Test creation of multiple instances of the Gillies STN model.
    """
    stn_cells = []
    for i in range(5):
        cell_idx = h.make_stn_cell_global()
        cell_idx = int(cell_idx)
        stn_cells.append(h.SThcells[cell_idx])

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


def test_stn_pynn_population(export_locals=True):
    """
    Test creation of PyNN population of STN cells.
    """
    from pyNN.utility import init_logging
    import pyNN.neuron as nrn

    init_logging(logfile=None, debug=True)
    nrn.setup()

    # STN cell population
    cell_type = StnCellType()
    p1 = nrn.Population(5, cell_type)

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
        receptor_type='distal_dend.Exp2Syn')

    # Recording
    # p1.record(['apical(1.0).v', 'soma(0.5).ina'])
    nrn.run(250.0)

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


if __name__ == '__main__':
    # test_stn_cells_multiple()
    # test_stn_pynn_population()

    # Make single cell
    import bluepyopt.ephys as ephys
    cell = StnCellModel()
    sim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
    icell = cell.instantiate(sim=sim)
