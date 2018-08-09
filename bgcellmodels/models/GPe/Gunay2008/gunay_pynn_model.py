"""
PyNN compatible cell models for GPe cell model.

@author     Lucas Koelman

@date       7/03/2018

"""

from neuron import h
import numpy as np

# from pyNN.standardmodels import StandardCellType
# from pyNN.neuron.cells import NativeCellType

import bgcellmodels.extensions.pynn.ephys_models as ephys_pynn
from bgcellmodels.extensions.pynn.ephys_locations import SomaDistanceRangeLocation

import gunay_model

# Debug messages
from bgcellmodels.common.stdutil import dotdict
from bgcellmodels.common import treeutils #, logutils

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

    # Our custom PyNN parameters
    # Must define 'default_parameters' in associated cell type
    parameter_names = [
        # See workaround for non-numerical parameters
        # 'GABA_synapse_mechanism',
        # 'GLU_synapse_mechanism',
        'tau_m_scale',
    ]

    # FIXME: workaround, so far PyNN only allows numerical parameters
    GABA_synapse_mechanism = 'GABAsyn'
    GLU_synapse_mechanism = 'GLUsyn'

    # Related to PyNN properties
    _mechs_params_dict = {
        'HCN':      ['gmax'],
        'leak':     ['gmax'],
        'NaF':      ['gmax'],
        'NaP':      ['gmax'],
        'Kv2':      ['gmax'],
        'Kv3':      ['gmax'],
        'Kv4f':     ['gmax'],
        'Kv4s':     ['gmax'],
        'KCNQ':     ['gmax'],
        'SK':       ['gmax'],
        'CaH':      ['gmax'],
    }
    rangevar_names = [p+'_'+m for m,params in _mechs_params_dict.iteritems() for p in params]
    gleak_name = 'gmax_leak'
    tau_m_scaled_regions = ['somatic', 'basal', 'apical', 'axonal']
    rangevar_scaled_regions = {}
    for rangevar in rangevar_names:
        parameter_names.append(rangevar + '_scale')
    

    def memb_init(self):
        """
        Initialization function required by PyNN.

        @override     EphysModelWrapper.memb_init()
        """
        gunay_model.fix_comp_dimensions(self)


    def _init_synapses(self):
        """
        Initialize synapses on this neuron.

        @override   EphysModelWrapper._init_synapses()
        """
        # Sample each region uniformly and place synapses there
        soma = self.icell.soma[0]
        synapse_spacing = 0.25
        segment_gen = treeutils.ascend_with_fixed_spacing(
                            soma(0.5), synapse_spacing)
        uniform_segments = [seg for seg in segment_gen]

        # Sample cell regions
        h.distance(0, 0.5, sec=soma) # reference for distance measurement

        is_proximal = lambda seg: h.distance(1, seg.x, sec=seg.sec) < 5.0
        proximal_segments = [seg for seg in uniform_segments if is_proximal(seg)]

        is_distal = lambda seg: h.distance(1, seg.x, sec=seg.sec) > 3
        distal_segments = [seg for seg in uniform_segments if is_distal(seg)]

        # Synapses counts are fixed
        num_gpe_syn = 8
        num_str_syn = 22
        num_stn_syn = 10
        rng = np.random # TODO: make from base seed + self ID
        
        proximal_indices = rng.choice(len(proximal_segments), 
                                num_gpe_syn+num_str_syn, replace=False)

        distal_indices = rng.choice(len(distal_segments), num_stn_syn,
                                replace=False)

        # Fill synapse lists
        self._synapses = {}
        self._synapses['proximal'] = prox_syns = {}
        self._synapses['distal'] = dist_syns = {}

        # Get constuctors for NEURON synapse mechanisms
        make_gaba_syn = getattr(h, self.GABA_synapse_mechanism)
        make_glu_syn = getattr(h, self.GLU_synapse_mechanism)

        prox_syns[('GABAA', 'GABAB')] = prox_gaba_syns = []
        for seg_index in proximal_indices:
            syn = make_gaba_syn(proximal_segments[seg_index])
            prox_gaba_syns.append(dotdict(synapse=syn, used=0,
                mechanism=self.GABA_synapse_mechanism))
        
        dist_syns[('AMPA', 'NMDA')] = dist_glu_syns = []
        for seg_index in distal_indices:
            syn = make_glu_syn(distal_segments[seg_index])
            dist_glu_syns.append(dotdict(synapse=syn, used=0,
                mechanism=self.GLU_synapse_mechanism))


    def _update_position(self, xyz):
        pass


    def get_threshold(self):
        """
        Get spike threshold for soma membrane potential (used for NetCon)

        @override   EphysModelWrapper.get_threshold()
        """
        return -10.0


class GPeCellType(ephys_pynn.EphysCellType):
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
        # 'GABA_synapse_mechanism': 'GABAsyn',
        # 'GLU_synapse_mechanism': 'GLUsyn',
        'tau_m_scale': 1.0,
    }

    # Defaults for Ephys parameters
    default_parameters.update({
        # ephys_model.params.values are ephys.Parameter objects
        # ephys_param.name is same as key in ephys_model.params
        p.name.replace(".", "_"): p.value for p in model._ephys_parameters
    })


    # extra_parameters = {}
    # default_initial_values = {'v': -65.0}
    
    # Combined with self.model.regions by EphysCellType constructor
    receptor_types = ['AMPA', 'NMDA', 'AMPA+NMDA',
                      'GABAA', 'GABAB', 'GABAA+GABAB']

    def get_schema(self):
        """
        Get mapping of parameter names to allowed parameter types.
        """
        # Avoids specifying default values for each scale parameter and
        # thereby calling the property setter for each of them
        schema = super(GPeCellType, self).get_schema()
        schema.update({v+'_scale': float for v in self.model.rangevar_names})
        return schema


    def can_record(self, variable):
        """
        Override or it uses pynn.neuron.record.recordable_pattern.match(variable)
        """
        return super(GPeCellType, self).can_record(variable)


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
