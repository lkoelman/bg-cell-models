"""
PyNN compatible cell models for GPe cell model.

@author     Lucas Koelman

@date       7/03/2018

"""

from neuron import h
import numpy as np

# PyNN imports
from pyNN.parameters import ArrayParameter

from bgcellmodels.extensions.pynn import cell_base, ephys_models as ephys_pynn
from bgcellmodels.extensions.pynn.ephys_locations import SomaDistanceRangeLocation
from bgcellmodels.morphology import morph_3d
from bgcellmodels.models.axon.foust2011 import AxonFoust2011

import gunay_model

# logutils.setLogLevel('quiet', [
#     'bluepyopt.ephys.parameters',
#     'bluepyopt.ephys.mechanisms'])


def define_synapse_locations():
    """
    Define locations / regions on the cell that will function as the target
    of synaptic connections.

    @return     list(SomaDistanceRangeLocation)
                List of location / region definitions.
    """

    # NOTE: distances are not realistic since Gunay model adjusts
    #       compartment dimensions to 1 micron after loading morphology
    proximal_dend = SomaDistanceRangeLocation(
        name='proximal_dend',
        seclist_name='basal',
        min_distance=0.01,
        max_distance=1.5)

    distal_dend = SomaDistanceRangeLocation(
        name='distal_dend',
        seclist_name='basal',
        min_distance=1.5,
        max_distance=1000.0)

    return [proximal_dend, distal_dend]


class GPeCellModel(ephys_pynn.EphysModelWrapper):
    """
    Gunay (2008) multi-compartmental GPe cell model.
    """
    _ephys_morphology = gunay_model.define_morphology(
                        'morphology/bg0121b_axonless_GENESIS_import.swc',
                        replace_axon=False)
    
    _ephys_mechanisms = gunay_model.define_mechanisms(
                        'config/mechanisms.json')

    _param_locations = gunay_model.define_locations(
                        'config/locations.json')

    # Ephys parameters will automaticall by converted to PyNN parameters
    _ephys_parameters = gunay_model.define_parameters(
                            'config/params_gunay2008_GENESIS.json',
                            'config/map_params_gunay2008_v2.json',
                            _param_locations)

    # _ephys_locations = define_synapse_locations()
    regions = ['proximal', 'distal']

    # Related to PyNN properties
    _mechs_params_dict = {
        # Nonspecific channels
        'HCN':      ['gmax'],
        'HCN2':     ['gmax'],
        'pas':      ['g'],
        # Na channels
        'NaF':      ['gmax'],
        'NaP':      ['gmax'],
        # K-channels
        'Kv2':      ['gmax'],
        'Kv3':      ['gmax'],
        'Kv4f':     ['gmax'],
        'Kv4s':     ['gmax'],
        'KCNQ':     ['gmax'],
        'SK':       ['gmax'],
        # Calcium channels / buffering
        'CaHVA':    ['gmax'],
        # 'Calcium':  [''],
    }
    rangevar_names = [p+'_'+m for m,params in _mechs_params_dict.iteritems() for p in params]
    gleak_name = 'g_pas'

    # map rangevar to cell region where it should be scaled, default is 'all'
    tau_m_scaled_regions = ['somatic', 'basal', 'apical', 'axonal']
    rangevar_scaled_seclists = {} 


    # Properties defined for synapse position
    synapse_spacing = 0.2
    region_boundaries = {
        'proximal': (0.0, 2.0),   # (um)
        'distal':   (1.0, 1e12),  # (um)
    }

    # Regions for extracellular stim (DBS) & rec (LFP)
    seclists_with_extracellular = ['all']

    # Spike threshold (mV)
    spike_threshold = {
        'soma': -10.0,
        'AIS': -10.0,
        'axon_terminal': -10.0,
    }

    def __init__(self, *args, **kwargs):
        # Define parameter names before calling superclass constructor
        self.parameter_names = GPeCellType.default_parameters.keys() + \
                               GPeCellType.extra_parameters.keys()
        for rangevar in self.rangevar_names:
            self.parameter_names.append(rangevar + '_scale')
        
        super(GPeCellModel, self).__init__(*args, **kwargs)
    

    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator

        @note   The call order is:

                - GpeCellModel.__init__()
                `- EphysModelWrapper.__init__()
                    `- ephys.models.CellModel.__init__()
                    `- cell_base.MorphModelBase.__init__()
                     `- GpeCellModel.instantiate()
                      `- EphysModelWrapper.instantiate()

        @override       ephys.models.CellModel.instantiate()
        """
        # Call instantiate method from Ephys model class
        ephys_pynn.EphysModelWrapper.instantiate(self, sim)

        # Transform morphology
        if len(self.transform) > 0 and not np.allclose(self.transform, np.eye(4)):
            morph_3d.transform_sections(self.icell.all, self.transform)

        # Create and append axon
        if len(self.streamline_coordinates_mm) > 0:
            self._init_axon(self.axon_class)

        # Fix conductances if axon is present (compensate loading)
        # self._init_gbar()

        # Init extracellular stimulation & recording
        if self.with_extracellular:
            self._init_emfield()

        # Adjust compartment dimensions like in GENESIS code
        self._fix_compartment_dimensions()


    def _init_axon(self, axon_class):
        """
        Initialize axon and update source sections for spike connections.
        """
        axon_builder = axon_class(
            without_extracellular=not self.with_extracellular)

        use_gap_junction = getattr(self, 'axon_using_gap_junction', False)

        # Choose parent section for axon
        axonal_secs = list(self.icell.axonal)
        if len(axonal_secs) > 0:
            axon_parent_sec = axonal_secs[-1]
            assert len(axon_parent_sec.children()) == 0
        else:
            # Attach axon directly to soma
            axon_parent_sec = self.icell.soma[0]

        # We already have an AIS so don't build it
        axon_builder.initial_comp_sequence.pop(0) # = ['aismyelin']

        # Build entire axon
        axon = axon_builder.build_along_streamline(
                    self.streamline_coordinates_mm,
                    termination_method='nodal_cutoff',
                    interp_method='arclength',
                    parent_cell=self.icell,
                    parent_sec=axon_parent_sec,
                    connection_method='translate_axon_closest',
                    connect_gap_junction=use_gap_junction,
                    gap_conductances=(getattr(self, 'gap_pre_conductance', None),
                                      getattr(self, 'gap_post_conductance', None)),
                    tolerance_mm=1e-4)
    
        self.axon = axon

        # Change source for NetCons (see pyNN.neuron.simulator code)
        terminal_sec = list(self.icell.axonal)[-1]
        terminal_source = terminal_sec(0.5)._ref_v # source for connections
        self.source_section = terminal_sec
        self.source = terminal_source

        # Support for multicompartment connections
        terminal_gid = self.owning_gid + int(1e6)
        self.region_to_gid['axon_terminal'] = terminal_gid
        self.gid_to_source[terminal_gid] = terminal_source
        self.gid_to_section[terminal_gid] = terminal_sec

        # Uncomment for gap junction solution
        if use_gap_junction:
            # super(GPeCellModel, self)._init_axon(axon_class)

            # Add AIS as a source section for connections
            source_sec = self.axon['aisnode'][0]
            source_gid = self.owning_gid + int(2e6)

            self.region_to_gid['AIS'] = source_gid
            self.gid_to_source[source_gid] = source_sec(0.5)._ref_v
            self.gid_to_section[source_gid] = source_sec


    def _fix_compartment_dimensions(self):
        """
        Normalizes all compartment dimensions to 1 um as in original
        Gunay (2008) model code.

        Note that this is a huge headache since dimensions are not
        biphysically realistic anymore.
        """
        def fix_dims(secs):
            for sec in secs:
                sec.L = 1.0
                for seg in sec:
                    seg.diam = 1.0

        # Only fix section defined in SWC morphology
        fix_dims(self.icell.somatic)
        fix_dims(self.icell.basal)

        axonal_sections = list(self.icell.axonal)
        axon_stub = axonal_sections[0]
        fix_dims([axon_stub])

        # NOTE: in unscaled morphology axon_diam / soma_diam = 1.5 / 13.4 = 0.112
        
        # Taper AIS section only
        # - AIS is 20 micron long over 5 segments. 
        # - Taper this so input resistance is low but there is still enough current to jump myelinated section.
        # sec = axonal_sections[1]
        # assert 'aisnode' in sec.name()
        # old_area = sum((seg.area() for seg in sec))

        # Because there are only two samples per section, the trapezoidal
        # interpolation rule will automatically result in tapering diameter in
        # the 5 consecutive segments of the AIS
        # start_diam = self.axon_taper_diam
        # h.pt3dchange(0, start_diam, sec=sec)

        # Rescale AIS conductances
        # new_area = sum((seg.area() for seg in sec))
        # new_diams = [seg.diam for seg in sec]
        # fac_area = new_area/old_area
        # print("New diams {}, area reduction {}".format(new_diams, fac_area))
        # for seg in sec:
        #     seg.g_NaF_Foust = seg.g_NaF_Foust * old_area / new_area

        # Taper axon diameter
        # start_diam = 0.05    # ~= 0.112 * soma_diam
        # stop_diam = 1.2     # diam of Foust (2011) nodal sections
        # delta_diam = stop_diam - start_diam
        # stop_length = 250.0 # taper diam over this distance
        # for i, sec in enumerate(self.axon['ordered']):
        #     for j in range(int(h.n3d(sec=sec))):
        #         loc = h.arc3d(j, sec=sec) / sec.L # x of 3d sample
        #         seg = sec(loc)
        #         if i == j == 0:
        #             h.distance(0, loc, sec=sec)
        #         stub_dist = h.distance(seg.x, sec=sec)
        #         if stub_dist > stop_length:
        #             return
        #         taper_diam = start_diam + stub_dist / stop_length * delta_diam
        #         h.pt3dchange(j, taper_diam, sec=sec)
        #         print("Stub dist is %f (x=%f), diam is %f (assigned %f)" % (stub_dist, loc, seg.diam, taper_diam))
        

    def _update_position(self, xyz):
        pass



class GPeCellType(cell_base.MorphCellType):
    """
    Encapsulates a GPe model described as a BluePyOpt Ephys model 
    for interoperability with PyNN.

    @see    Based on definition of SimpleNeuronType and standardized cell types in:
                - https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
                - https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/standardmodels/cells.py

            And on documentation at:
                - http://neuralensemble.org/docs/PyNN/backends/NEURON.html#using-native-cell-models

    IMPLEMENTATION NOTES
    --------------------

    @note   The class must be callable with parameters passed as keyword arguments.
            These are passed up the inheritance chain to BaseModelType, which
            fills the parameter space.

    @note   The encapsulated model (class attribute 'model') can be any object
            as long as the __call__ method instantiates it in NEURON, and accepts
            keyword arguments containing parameters and values
    """

    # The encapsualted model available as class attribute 'model'
    model = GPeCellModel

    # Defaults for our custom PyNN parameters
    default_parameters = {
        'default_GABA_mechanism': np.array('GABAsyn'),
        'default_GLU_mechanism': np.array('GLUsyn'),
        'membrane_noise_std': 0.0,
        # Biophysical properties
        'tau_m_scale': 1.0,
        # Extracellular stim & rec
        'with_extracellular': False,
        'electrode_coordinates_um' : ArrayParameter([]),
        'rho_extracellular_ohm_cm' : 0.03, 
        'transfer_impedance_matrix': ArrayParameter([]),
        # 3D specification
        'transform': ArrayParameter([]),
        # Axon
        'streamline_coordinates_mm': ArrayParameter([]), # Sequence([])
        'axon_using_gap_junction': True,
        'gap_pre_conductance': 1e-5,
        'gap_post_conductance': 1e-3,
        'axon_taper_diam': 0.05,
    }

    # Defaults for Ephys parameters
    default_parameters.update({
        # ephys_model.params.values are ephys.Parameter objects
        # ephys_param.name is same as key in ephys_model.params
        p.name.replace(".", "_"): p.value for p in model._ephys_parameters
    })

    # NOTE: extra_parameters supports non-numpy types. 
    extra_parameters = {
        'axon_class': AxonFoust2011,
    }


    # extra_parameters = {}
    default_initial_values = {'v': -68.0}
    
    # Combined with self.model.regions by MorphCellType constructor
    receptor_types = ['AMPA', 'NMDA', 'AMPA+NMDA',
                      'GABAA', 'GABAB', 'GABAA+GABAB']


    def can_record(self, variable):
        """
        Override or it uses pynn.neuron.record.recordable_pattern.match(variable)
        """
        return super(GPeCellType, self).can_record(variable)

GpeProtoCellType = GPeCellType


class GpeArkyCellType(GPeCellType):
    """
    GPe ArkyPallidal cell. It uses the same Gunay (2008) GPe cell model with
    modified parameters to reduce sponaneous firing rate and rebound firing.

    Sources
    -------

    Abdi, Mallet et al (2015) - Prototypical and Arkypallidal Neurons ...
        - See abstract: weaker persistent Na current and rebound firing
        - Fig. 7

    Bogacz, Moraud, et al (2016) - Properties of Neurons in External ...
        - Fig. 3
    """

    # Defaults for our custom PyNN parameters
    default_parameters = dict(GPeCellType.default_parameters)

    default_parameters.update({
        'gmax_NaP_scale': 0.45,
    })
    

def test_record_gpe_model(export_locals=False):
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
        'Ra_basal': 200.0
    }
    cell_type = GPeCellType(**parameters)
    p1 = nrn.Population(5, cell_type)
    
    print(p1.get('Ra_basal'))
    p1.rset('Ra_basal', RandomDistribution('uniform', low=100., high=300.))
    print(p1.get('Ra_basal'))
    
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
    test_record_gpe_model(export_locals=True)
