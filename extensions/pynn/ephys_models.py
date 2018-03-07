"""
Module for working with BluePyOpt cell models in PyNN.

@author     Lucas Koelman

@date       14/02/2018


USEFUL EXAMPLES
---------------

https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/cells.py
https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/standardmodels/cells.py

"""

import ephys


class EphysModelPyNNClient(ephys.models.CellModel):
    """
    Subclass of Ephys CellModel that conforms to the interface required
    by the 'model' attribute of a PyNN CellType class.

    @see    Based on definition of SimpleNeuronType and standardized cell types in:
            https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
            https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/standardmodels/cells.py

            And on documentation at:
            http://neuralensemble.org/docs/PyNN/backends/NEURON.html#using-native-cell-models


    USAGE
    -----

    Instantiate the model description like you would a regular Ephys CellModel:

        >>> cell_descr = EphysCellModel(
        >>>                 'GPe',
        >>>                 morph=define_morphology(...),
        >>>                 mechs=define_mechanisms(...),
        >>>                 params=define_parameters(...))
        >>> 
        >>> class MyPyNNCellType(pyNN.standardmodels.StandardCellType):
        >>>     model = cell_descr


    IMPLEMENTATION NOTES
    --------------------

    @note   The encapsulated model (CellType attribute 'model') can be any object
            as long as the __call__ method instantiates it in NEURON, accepts
            keyword arguments containing parameters and values, and returns
            the instantiated model object.
    """

    def __call__(self, **cell_parameters):
        """
        Instantiate the cell model.

        @see        Called by _build_cell(...) in module PyNN.neuron.simulator
                    https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/simulator.py

        @effect     calls self.instantiate()
        """
        # TODO: modify self.params based on cell_parameters if any
        return self.instantiate(sim=TODO)

