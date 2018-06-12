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
import re
import itertools
from copy import copy, deepcopy

import bluepyopt.ephys as ephys
from pyNN.neuron import state as nrn_state, h
from pyNN.neuron.cells import NativeCellType

ephys_nrn_sim = None

rng_structural_variability = h.Random(
    nrn_state.mpi_rank + nrn_state.native_rng_baseseed)


def ephys_sim_from_pynn():
    """
    Get Ephys NrnSimulator without changing parameters of wrapped
    NEURON simulator from those set by pyNN.
    """
    global ephys_nrn_sim

    if ephys_nrn_sim is None:
        cvode_active = nrn_state.cvode.active()
        cvode_minstep = nrn_state.cvode.minstep()
        ephys_nrn_sim = ephys.simulators.NrnSimulator(
                            dt=nrn_state.dt,
                            cvode_active=cvode_active,
                            cvode_minstep=cvode_minstep)
    return ephys_nrn_sim


def make_valid_attr_name(name):
    """
    Make name into a valid attribute name.

    @param      name : str

    @return     modified_name : str
                String that is a valid attribute name.
    """
    return name.replace(".", "_")


class UnitFetcherPlaceHolder(dict):
    def __getitem__(self, key):
        return 'V'


class EphysCellType(NativeCellType):
    """
    PyNN native cell type that has Ephys model as 'model' attribute.
    """
    # TODO: units property with __getitem__ implemented
    units = UnitFetcherPlaceHolder()


    def __init__(self, **kwargs):
        """
        The instantated cell type is passed to Population.

        @param      extra_receptors : iterable(str)

                    Synaptic mechanism names of synapses that should be allowed
                    on this cell type.

        @post       The list of receptors in self.receptor_types will be 
                    updated with each receptor in 'with_receptors' for every
                    cell location/region.
        """
        extra_receptors = kwargs.pop('extra_receptors', None)
        super(EphysCellType, self).__init__(**kwargs)

        # Combine receptors defined on the cell type with regions
        # defined on the model class
        celltype_receptors = type(self).receptor_types
        if extra_receptors is None:
            all_receptors = celltype_receptors
        else:
            all_receptors = celltype_receptors + list(extra_receptors)
        
        region_receptors = []
        for region in self.model.regions:
            for receptor in all_receptors:
                region_receptors.append(region + "." + receptor)

        self.receptor_types = region_receptors



class CellModelMeta(type):
    """
    Create new type that is subclass of ephys.models.CellModel and
    automatically registers Ephys parameters as python properties.
    """

    def __new__(this_meta_class, new_class_name, new_class_bases, new_class_namespace):
        """
        Create new type for a class definition that has this metaclass
        as its metaclass.

        @effect     Converts each Ephys.Parameter object stored in subclass
                    attribute '_ephys_parameters' into a class property,
                    and stores its name in the 'parameter_names' attribute.

        @note       This method is called once every time a class is defined with 
                    MechanismType as its metaclass.
        """
        # Process mechanism variables declared in class definition
        # modified_bases = tuple(list(new_class_bases) + [ephys.models.CellModel])
        
        parameter_names = new_class_namespace.get("parameter_names", [])
        
        for e_param in new_class_namespace.get("_ephys_parameters", []):
            param_name = e_param.name
            
            # NOTE: self.params is set in __init__()
            def __get_ephys_param(self):
                return self.params[param_name].value
            
            def __set_ephys_param(self, value):
                self.params[param_name].value = value
                self.params[param_name].instantiate(self.sim, self.icell)

            # Change name of functions for clarity
            param_name_nodots = make_valid_attr_name(param_name)
            __get_ephys_param.__name__ = "__get_" + param_name_nodots
            __set_ephys_param.__name__ = "__set_" + param_name_nodots
            
            parameter_names.append(param_name_nodots)

            # Insert into namespace as properties
            new_class_namespace[param_name_nodots] = property(
                                                    fget=__get_ephys_param,
                                                    fset=__set_ephys_param)

        # Make Ephys locations into class properties
        # location_names = []
        # for ephys_loc in new_class_namespace.get("_ephys_locations", []):

        #     loc_name = ephys_loc.name
        #     loc_name_nodots = make_valid_attr_name(loc_name)
            
        #     # NOTE: self.params is set in __init__()
        #     def __get_location(self):
        #         return self.locations[loc_name].instantiate(self.sim, self.icell)

        #     # Change name of getter function
        #     __get_location.__name__ = "__get_" + loc_name_nodots
        #     location_names.append(loc_name_nodots)

        #     # Insert into namespace as properties
        #     new_class_namespace[loc_name_nodots] = property(fget=__get_location)

        # Parameter names for pyNN
        new_class_namespace['parameter_names'] = parameter_names
        # new_class_namespace['location_names'] = location_names

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


    @note   Called by ID._build_cell() defined in
                https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/simulator.py
            
            ID._cell = Population.cell_type.model(**cell_parameters)


    @see    Based on definition of SimpleNeuronType and standardized cell types in:
                https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
                https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/standardmodels/cells.py

            And on documentation at:
                http://neuralensemble.org/docs/PyNN/backends/NEURON.html#using-native-cell-models
    
    ATTRIBUTES
    ----------

    @attr   <location_name> : nrn.Section or nrn.Segment

            Each location defined on the cell has a corresponding attribute.
            This is a dynamic attribute: the Segment or Section is sampled
            from the region.


    @attr   <param_name> : float

            Each parameter defined for thsi cell has a corresponding attribute
    

    @attr   locations : dict<str, Ephys.location>

            Dict containing all locations defined on the cell.
            Keys are the same as the location attributes of this cell.


    @attr   synapses : dict<str, list(nrn.POINT_PROCESS)>

            Synapse object that synapse onto this cell.
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

                    The original function tries to recreate the Hoc template every
                    time so isn't suitable for instantiating multiple copies.
        """
        if not hasattr(sim.neuron.h, template_name):
            template_string = ephys.models.CellModel.create_empty_template(
                                                        template_name,
                                                        seclist_names,
                                                        secarray_names)
            sim.neuron.h(template_string)

        template_function = getattr(sim.neuron.h, template_name)

        return template_function()


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator

        The default behaviour implemented here works for cells that have ephys
        morphology, mechanism, and parameter definitions. If you want to use
        a Hoc cell without these definitions, you should subclass to override
        this behaviour.

        @override       ephys.models.CellModel.instantiate()
        """

        # Instantiate NEURON cell
        icell = self.create_empty_cell(
                    self.name,
                    sim=sim,
                    seclist_names=self.seclist_names,
                    secarray_names=self.secarray_names)

        self.icell = icell
        # icell.gid = self.gid # gid = int(ID) where ID._cell == self

        self.morphology.instantiate(sim=sim, icell=icell)

        for mechanism in self.mechanisms:
            mechanism.instantiate(sim=sim, icell=icell)

        for param in self.params.values():
            param.instantiate(sim=sim, icell=icell)


    def __init__(self, *args, **kwargs):
        """
        As opposed to the original CellModel class,
        this class instantiates the cell in its __init__ method.

        @param      **kwargs : dict(str, object)
                    Parameter name, value pairs

        @post       self.icell contains the instantiated Hoc cell model

        """
        # Get parameter definitions from class attributes of subclass.
        model_name = self.__class__.__name__

        # Ensure self.params has valid names, and does not refer to class params
        params = deepcopy(getattr(self, '_ephys_parameters', None))
        if params is not None:
            for e_param in params:
                e_param.name = make_valid_attr_name(e_param.name)

        super(EphysModelWrapper, self).__init__(
            model_name,
            morph=getattr(self, '_ephys_morphology', None),
            mechs=getattr(self, '_ephys_mechanisms', None),
            params=params)

        self.sim = ephys_sim_from_pynn()
        self.instantiate(sim=self.sim)

        # Add locations / regions
        # self.locations = {}
        # for static_loc in self._ephys_locations:
        #     loc = copy(static_loc)
        #     # loc.sim = self.sim
        #     # loc.icell = self.icell
        #     self.locations[make_valid_attr_name(loc.name)] = loc

        # NOTE: default params will be passed by pyNN Population
        for param_name, param_value in kwargs.iteritems():
            # - self.params are the Ephys parameters
            # - these are already set in instantiate() so don't set again
            if (param_name in self.params) and (self.params[param_name].value == param_value):
                continue
            elif param_name in self.parameter_names:
                setattr(self, param_name, param_value)

        # Synapses will map the mechanism name to the synapse object
        # and is set by our custom Connection class
        # self.synapses = {} # mech_name: str -> list(nrn.POINT_PROCESS)
        # Make synapse objects as targets for connections
        self._init_synapses()

        # Attributes required by PyNN
        self.source_section = self.icell.soma[0]
        self.source = self.icell.soma[0](0.5)._ref_v
        
        self.rec = h.NetCon(self.source, None,
                            self.get_threshold(), 0.0, 0.0,
                            sec=self.source_section)
        self.spike_times = h.Vector(0) # see pyNN.neuron.recording.Recorder._record()
        self.traces = {}
        self.recording_time = False
        # self.parameter_names = (set in subclass body & metaclass)

        # NOTE: _init_lfp() is called by our custom Population class after
        #       updating each cell's position


    def memb_init(self):
        """
        Set initial values for all variables in this cell.

        @override   Implements the memb_init() function that is part of the
                    pyNN interface for cell models.
        """
        raise NotImplementedError("Please implement an initializer for your "
                "custom cell model.")


    def get_threshold(self):
        """
        Get spike threshold for self.source variable (usually points to membrane
        potential). This threshold is used when creating NetCon connections.

        @override   Implements get_threshold() which belongs to the pyNN
                    interface for cell models.

        @return     threshold : float
        """
        raise NotImplementedError("Please set the spike threshold for your "
                "custom cell model.")


    def _init_synapses(self):
        """
        Initialize synapses on this neuron.

        @pre        self._synapses is empty or unset

        @post       self._synapses is a nested dict with depth=2 and following
                    structure:
                    { region: {receptors: list({ 'synapse':syn, 'used':int }) } }
        """
        raise NotImplementedError("Subclass should place synapses in dendritic tree.")


    def _init_lfp(self):
        """
        Initialize LFP sources for this cell.
        Override in subclass to implement this.

        @return     lfp_tracker : nrn.POINT_PROCESS
                    Object with recordable variable that represents the cell's
                    summed LFP contributions
        """
        return None


    def _update_position(self, xyz):
        """
        Called when the cell's position is changed, e.g. when changing 
        the space/structure of the parent Population.

        @effect     Adds xyz to all coordinates of the root sections and then
                    calls h.define_shape() so that whole tree is translated.
        """

        # # translate the root section and re-define shape to translate entire cell
        # source_ref = h.SectionRef(sec=self.source_section)
        # root_sec = source_ref.root # pushes section
        # h.pop_section()

        # # initial define shape to make sure 3D info is present
        # h.define_shape(sec=root_sec)

        # for i in range(int(h.n3d(sec=root_sec))):
        #     h.pt3dchange(i, xyz[0], xyz[1], xyz[2], h.diam3d(i, sec=root_sec))

        # # redefine shape to translate tree based on updated root position
        # h.define_shape(sec=root_sec)

        raise NotImplementedError("Subclass should update cell position.")


    def get_synapse_list(self, region, receptors):
        """
        Return list of synapses in region that have given
        neutotransmitter receptors.

        @param      region : str
                    Region descriptor.

        @param      receptors : enumerable(str)
                    Receptor descriptors.

        @return     synapse_list : list(dict())
                    List of synapse containers
        """
        return next((syns for recs, syns in self._synapses[region].items() if all(
                        (ntr in recs for ntr in receptors))), [])


    def get_synapses_by_mechanism(self, mechanism):
        """
        Get all synapses that are an instance of the given NEURON mechanism
        name.

        @param      mechanism : str
                    NEURON mechanism name

        @return     synapse_list : list(dict())
                    List of synapse containers
        """
        return sum((synlist for region in self._synapses.values() for recs, synlist in region.items()), [])


    def get_synapse(self, region, receptors, mark_used, **kwargs):
        """
        Get a synapse in cell region for given neurotransmitter
        receptors.

        @param      region : str
                    Region descriptor.

        @param      receptors : enumerable(str)
                    Receptor descriptors.

        @param      mark_used : bool
                    Whether the synapse should be marked as used
                    (targeted by a NetCon).

        @param      **kwargs
                    (Unused) extra keyword arguments for use when overriding
                    this method in a subclass.

        @return     synapse, num_used : tuple(nrn.POINT_PROCESS, int)
                    NEURON point process object that can serve as the
                    target of a NetCon object, and the amount of times
                    the synapse has been used before as the target
                    of a NetCon.
        """
        syn_list = self.get_synapse_list(region, receptors)
        if len(syn_list) == 0:
            raise Exception("Could not find any synapses for "
                "region '{}' and receptors'{}'".format(region, receptors))

        min_used = min((meta.used for meta in syn_list))
        metadata = next((meta for meta in syn_list if meta.used==min_used))
        if mark_used:
            metadata.used += 1 # increment used counter
        return metadata.synapse, min_used


    def resolve_synapses(self, spec):
        """
        Resolve string definition of a synapse or point process
        on this cell (used by Recorder).

        @param      spec : str
                    
                    Synapse specifier in format "mechanism_name[slice]" where
                    'slice' as a slice expression like '::2' or integer.

        @return     list(nrn.POINT_PROCESS)
                    List of NEURON synapse objects.
        """
        matches = re.search(r'^(?P<mechname>\w+)(\[(?P<slice>[\d:]+)\])', spec)
        mechname = matches.group('mechname')
        slice_expr = matches.group('slice') # "i:j:k"
        slice_parts = slice_expr.split(':') # ["i", "j", "k"]
        slice_parts_valid = [int(i) if i!='' else None for i in slice_parts]
        syn_metadata = self.get_synapses_by_mechanism(mechname)
        synlist = [meta.synapse for meta in syn_metadata]
        
        if len(synlist) == 0:
            return []
        elif len(slice_parts) == 1: # zero colons
            return [synlist[int(slice_parts_valid[0])]]
        else: # at least one colon
            slice_object = slice(*slice_parts_valid)
            return synlist[slice_object]

        # elif len(slice_parts) == 2: # one colon
        #     start = slice_parts[0]
        #     stop = slice_parts[1]
        #     if start == stop == '':
        #         return self.synlist[:]
        #     elif start == '':
        #         return self.synlist[:int(stop)]
        #     elif stop == '':
        #         return self.synlist[int(start):]
        #     else:
        #         return self.synlist[int(start):int(stop)]
        # elif len(slice_parts) == 3: # two colons


    def resolve_section(self, spec):
        """
        Resolve a section specification.

        @return     nrn.Section
                    The section intended by the given section specifier.


        @param      spec : str
                    
                    Section specifier in the format 'section_container[index]',
                    where 'section_container' is the name of a secarray, Section,
                    or SectionList that is a public attribute of the icell,
                    and the index part is optional.


        @note       The default secarray for an Ephys CellModel are:
                    'soma', 'dend', 'apic', 'axon', 'myelin'.
                    
                    The default SectionLists are:
                    'somatic', 'basal', 'apical', 'axonal', 'myelinated', 'all'

                    If 'section_container' matches any of these names, it will
                    be treated as such. If not, it will be treated as an indexable
                    attribute of the icell instance.
        """
        matches = re.search(r'^(?P<secname>\w+)(\[(?P<index>\d+)\])?', spec)
        sec_name = matches.group('secname')
        sec_index = matches.group('index')
        icell = self.icell
        
        if sec_index is None:
            return getattr(icell, sec_name)
        else:
            sec_index = int(sec_index)
        
        if sec_name in ['soma', 'dend', 'apic', 'axon', 'myelin']:
            # target is a secarray
            return getattr(icell, sec_name)[sec_index]
        
        elif sec_name in ['somatic', 'basal', 'apical', 'axonal', 'myelinated', 'all']:
            # target is a SectionList (does not support indexing)
            seclist = getattr(icell, sec_name)
            return next(itertools.islice(seclist, sec_index, sec_index + 1))
        
        else:
            # assume target is a secarray
            return getattr(icell, sec_name)[sec_index]
