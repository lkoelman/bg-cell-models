"""
Module for working with BluePyOpt cell models in PyNN.

@author     Lucas Koelman

@date       14/02/2018


USEFUL EXAMPLES
---------------

https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/cells.py
https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/standardmodels/cells.py
https://github.com/apdavison/BluePyOpt/blob/pynn-models/bluepyopt/ephys_pyNN/models.py

"""

from copy import deepcopy
import bluepyopt.ephys as ephys


class CellModelMeta(type):
    """
    Create new type that is subclass of ephys.models.CellModel and
    automatically registers Ephys parameters as python properties.
    """

    def __new__(this_meta_class, new_class_name, new_class_bases, new_class_namespace):
        """
        Create new type for a class definition that has this metaclass
        as its metaclass.

        @note       This method is called once every time a class is defined with 
                    MechanismType as its metaclass.
        """
        # Process mechanism variables declared in class definition

        print("These are the bases: {}".format(new_class_bases))
        print(type(new_class_bases))

        # modified_bases = tuple(list(new_class_bases) + [ephys.models.CellModel])
        parameter_names = [] # for pyNN

        for e_param in new_class_namespace.get("_ephys_parameters", []):
            param_name = e_param.name
            
            def __get_ephys_param(self):
                return self.params[param_name].value
            
            def __set_ephys_param(self, value):
                self.params[param_name].value = value
                self.params[param_name].instantiate(self.sim, self.icell)

            # Change name of functions for clarity
            param_name_nodots = param_name.replace(".", "_")
            __get_ephys_param.__name__ = "__get_" + param_name_nodots
            __set_ephys_param.__name__ = "__set_" + param_name_nodots
            parameter_names.append(param_name_nodots)

            # Insert into namespace as properties
            new_class_namespace[param_name_nodots] = property(
                                                    fget=__get_ephys_param,
                                                    fset=__set_ephys_param)

        # Parameter names for pyNN
        new_class_namespace['parameter_names'] = parameter_names

        return type.__new__(this_meta_class, new_class_name, 
                            new_class_bases, new_class_namespace)


#     def __init__(new_class, new_class_name, new_class_bases, new_class_namespace):
#         """
#         Transform ephys parameters defined in subclass to properties for pyNN
# 
#         @param      new_class : type
#                     the class object returned by __new__
# 
#         @note       This method is called once every time a class is defined with 
#                     this class as its metaclass.
#         """
#         setattr(new_class, 'myprop', property(fget=__get_func, fset=__set_func))



class EphysModelWrapper(ephys.models.CellModel):
    """
    Subclass of Ephys CellModel that conforms to the interface required
    by the 'model' attribute of a PyNN CellType class.

    This is a modified version of an Ephys CellModel that allows multiple
    instances of a cell to be created. As opposed to the original CellModel class,
    this class instantiates the cell in its __init__ method.

    @see    Based on definition of SimpleNeuronType and standardized cell types in:
                https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
                https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/standardmodels/cells.py

            And on documentation at:
                http://neuralensemble.org/docs/PyNN/backends/NEURON.html#using-native-cell-models

    """

    __metaclass__ = CellModelMeta

    @staticmethod
    def create_empty_cell(
            template_name,
            sim,
            seclist_names=None,
            secarray_names=None):
        """
        Create an empty cell in Neuron

        @override   CellModel.create_empty_cell
                    The original function tries to recreate the template every
                    time so isn't suitable for instantiating multiple copies.
        """
        # TODO: find out how to create additional copies of a cell
        if not hasattr(sim.neuron.h, template_name):
            template_string = ephys.models.CellModel.create_empty_template(
                                                        template_name,
                                                        seclist_names,
                                                        secarray_names)
            sim.neuron.h(template_string)

        template_function = getattr(sim.neuron.h, template_name)

        return template_function()


    def __init__(self, *args, **kwargs):
        """
        Factory method to create a cell that will not be stored on the
        cell description object.

        @param      **kwargs : dict(str, object)
                    Parameter name, value pairs

        @return     cell : HocObject
                    Return value of Hoc template.

        @see        Called by _build_cell(...) in module PyNN.neuron.simulator
                    https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/simulator.py
        """
        # TODO: check if keyword arguments contain parameters and set them

        # First make params for ephys.models.CellModel
        # Get parameters from sublass
        model_name = self.__class__.__name__
        super(EphysModelWrapper, self).__init__(
            model_name,
            morph=self._ephys_morphology,
            mechs=self._ephys_mechanisms,
            params=deepcopy(self._ephys_parameters)) # copy the class parameters

        # TODO: modify self.params based on cell_parameters if any
        # TODO: get the simulator from pyNN and don't override parameters
        sim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
        self.sim = sim
        

        icell = self.create_empty_cell(
                    self.name,
                    sim=sim,
                    seclist_names=self.seclist_names,
                    secarray_names=self.secarray_names)

        self.icell = icell
        icell.gid = self.gid # TODO: see how PyNN manages GID

        self.morphology.instantiate(sim=sim, icell=icell)

        for mechanism in self.mechanisms:
            mechanism.instantiate(sim=sim, icell=icell)
        
        # NOTE: default params will be passed by pyNN Population
        kwarg_parameters = []
        for param_name, param_value in kwargs.iteritems():
            if param_name in self.parameter_names:
                setattr(self, param_name, param_value) # calls property setter -> param.instantiate()
                kwarg_parameters.append(param_name)

        # Apply default parameters from class definition
        for param in self.params.values():
            if param.name.replace(".", "_") not in kwarg_parameters:
                param.instantiate(sim=sim, icell=icell)

        # Attributes needed for PyNN
        self.source_section = icell.soma[0]
        self.source = icell.soma[0](0.5)._ref_v
        # self.parameter_names = ... set in metaclass
        self.traces = {}
        self.recording_time = False


    # TODO: map attributes to (instantiated?) model's attributes as in pyNN/neuron/cells.py