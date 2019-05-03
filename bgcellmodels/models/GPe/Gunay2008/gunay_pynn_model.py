"""
PyNN compatible cell models for GPe cell model.

@author     Lucas Koelman

@date       7/03/2018

"""

from neuron import h
import numpy as np

# from pyNN.standardmodels import StandardCellType
# from pyNN.neuron.cells import NativeCellType

from bgcellmodels.extensions.pynn import cell_base, ephys_models as ephys_pynn
from bgcellmodels.extensions.pynn.ephys_locations import SomaDistanceRangeLocation

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
    TODO: make EPhysModelWrapper inherit from MorphModelBase
    - put MorphModelBase under MorphModelBase (all cells are morphological)
    - remove stuff from this class and superclass as necessary
    - add axon-specific stuff in new class? Or do it all optionally in this class
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

    # map rangevar to cell region where it should be scaled, default is 'all'
    tau_m_scaled_regions = ['somatic', 'basal', 'apical', 'axonal']
    rangevar_scaled_seclists = {} 


    # Properties defined for synapse position
    synapse_spacing = 0.2
    region_boundaries = {
        'proximal': (0.0, 2.0),   # (um)
        'distal':   (1.0, 1e12),  # (um)
    }

    # Spike threshold (mV)
    spike_threshold_source_sec = -10.0

    def __init__(self, *args, **kwargs):
        # Define parameter names before calling superclass constructor
        self.parameter_names = GPeCellType.default_parameters.keys()
        for rangevar in self.rangevar_names:
            self.parameter_names.append(rangevar + '_scale')
        
        super(GPeCellModel, self).__init__(*args, **kwargs)
    

    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator

        @override       ephys.models.CellModel.instantiate()
                        
                        Since the wrapped model is a pure Hoc model completely
                        defined by its Hoc template, i.e. without ephys
                        morphology, parameters, or mechanisms definitions,
                        we have to override instantiate().
        """

        super(GPeCellModel, self).instantiate(sim)

        # Adjust compartment dimensions like in GENESIS code
        gunay_model.fix_comp_dimensions(self)


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
        'tau_m_scale': 1.0,
        'membrane_noise_std': 0.0,
    }

    # Defaults for Ephys parameters
    default_parameters.update({
        # ephys_model.params.values are ephys.Parameter objects
        # ephys_param.name is same as key in ephys_model.params
        p.name.replace(".", "_"): p.value for p in model._ephys_parameters
    })


    # extra_parameters = {}
    default_initial_values = {'v': -68.0}
    
    # Combined with self.model.regions by MorphCellType constructor
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

GpeProtoCellType = GPeCellType


class GpeArkyCellType(cell_base.MorphCellType):
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

    # The encapsualted model available as class attribute 'model'
    model = GPeCellModel

    # Defaults for our custom PyNN parameters
    default_parameters = {
        'default_GABA_mechanism': np.array('GABAsyn'),
        'default_GLU_mechanism': np.array('GLUsyn'),
        'tau_m_scale': 1.0,
        'gmax_NaP_scale': 0.45,
        'membrane_noise_std': 0.0,
    }

    # Defaults for Ephys parameters
    default_parameters.update({
        p.name.replace(".", "_"): p.value for p in model._ephys_parameters
    })


    # extra_parameters = {}
    default_initial_values = {'v': -65.0}
    
    # Combined with self.model.regions by MorphCellType constructor
    receptor_types = ['AMPA', 'NMDA', 'AMPA+NMDA',
                      'GABAA', 'GABAB', 'GABAA+GABAB']

    def get_schema(self):
        """
        Get mapping of parameter names to allowed parameter types.
        """
        # Avoids specifying default values for each scale parameter and
        # thereby calling the property setter for each of them
        schema = super(GpeArkyCellType, self).get_schema()
        schema.update({v+'_scale': float for v in self.model.rangevar_names})
        return schema


    def can_record(self, variable):
        """
        Override or it uses pynn.neuron.record.recordable_pattern.match(variable)
        """
        return super(GpeArkyCellType, self).can_record(variable)


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
