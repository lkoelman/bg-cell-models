"""
PyNN compatible cell models for GPe cell model.

@author     Lucas Koelman

@date       7/03/2018

"""

from pyNN.standardmodels import StandardCellType
from pyNN.neuron.cells import NativeCellType

import gunay_model


# TODO: do we need StandardCellType or is NativeCellType sufficient?
class GPeCellType(StandardCellType):
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
    model = ephys_model = gunay_model.define_cell() # non-instantiated ephys.CellModel

    # TODO: to build interface properties: put generic things extracted from
    #       the Ephys CellModel in the encapsulated class, and put rest here.
    @property
    def default_parameters(self):
        """
        Base class property 'default_parameters'.
        """
        return {
            # ephys_model.params.values are ephys.Parameter objects
            # ephys_param.name is same as key in ephys_model.params
            ephys_param.name: ephys_param.value for ephys_param in ephys_model.params.values()
        }


    @property
    def default_initial_values(self):
        """
        Base class property 'default_initial_values'.
        """
        return {} # TODO: is this necessary?
    

    

    def __init__(self, *args, **kwargs):
        """
        
        IMPLEMENTATION NOTES
        --------------------
        """
        pass