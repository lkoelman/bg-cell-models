"""
PyNN compatible cell models for GPe cell model.

@author     Lucas Koelman

@date       7/03/2018

"""

# from pyNN.standardmodels import StandardCellType
from pyNN.neuron.cells import NativeCellType

import extensions.pynn.ephys_models as ephys_pynn
from extensions.pynn.ephys_locations import SomaDistanceRangeLocation

import gunay_model

# Debug messages
from common import logutils
logutils.setLogLevel('quiet', ['bluepyopt.ephys.parameters', 'bluepyopt.ephys.mechanisms'])


def define_gpe_locations():
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
        max_distance=100.0,
        syn_mech_names=['Exp2Syn'])

    distal_dend = SomaDistanceRangeLocation(
        name='distal_dend',
        seclist_name='basal',
        min_distance=100.0,
        max_distance=1000.0,
        syn_mech_names=['Exp2Syn'])

    return [proximal_dend, distal_dend]


class GPeCellModel(ephys_pynn.EphysModelWrapper):
    """
    """
    _ephys_morphology = gunay_model.define_morphology(
                        'morphology/bg0121b_axonless_GENESIS_import.swc',
                        replace_axon=False)
    
    _ephys_mechanisms = gunay_model.define_mechanisms(
                        'config/mechanisms.min.json')
    
    _ephys_parameters = gunay_model.define_parameters(
                        'config/params_hendrickson2011_GENESIS.min.json',
                        'config/map_params_hendrickson2011.min.json')

    _ephys_locations = define_gpe_locations()
    
    # _morph_func = functools.partial(
    #                     gunay_model.define_morphology,
    #                     'morphology/bg0121b_axonless_GENESIS_import.swc',
    #                     replace_axon=False)
    # _mechs_func = functools.partial(
    #                     gunay_model.define_mechanisms,
    #                     'config/mechanisms.min.json')
    # _params_func = functools.partial(
    #                     gunay_model.define_parameters,
    #                     'config/params_hendrickson2011_GENESIS.min.json',
    #                     'config/map_params_hendrickson2011.min.json',)


class GPeCellType(NativeCellType):
    """
    Encapsulates a GPe model described as a BluePyOpt Ephys model 
    for interoperability with PyNN.

    @see    Based on definition of SimpleNeuronType and standardized cell types in:
            https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
            https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/standardmodels/cells.py

            And on documentation at:
            http://neuralensemble.org/docs/PyNN/backends/NEURON.html#using-native-cell-models

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

    # TODO: to build interface properties: put generic things extracted from
    #       the Ephys CellModel in the encapsulated class, and put rest here.
    default_parameters = {
        # ephys_model.params.values are ephys.Parameter objects
        # ephys_param.name is same as key in ephys_model.params
        p.name.replace(".", "_"): p.value for p in model._ephys_parameters
    }

    # extra_parameters = {}
    # default_initial_values = {'v': -65.0}

    # Synapse receptor types per region
    receptor_types = [ # prefixes are ephys model secarray names
        'proximal_dend.Exp2Syn', 'distal_dend.Exp2Syn'
    ]


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
    p1 = nrn.Population(5, GPeCellType(**parameters))
    
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
