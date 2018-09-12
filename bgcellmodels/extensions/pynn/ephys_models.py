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
import math
from copy import deepcopy
# from abc import ABCMeta, abstractmethod

import numpy as np
import bluepyopt.ephys as ephys
from pyNN.neuron import state as nrn_state, h
from pyNN.neuron.cells import NativeCellType
import quantities as pq

import logging
logger = logging.getLogger('ephys_models')

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


# Define additional units not recognized by quantities module
pq.UnitQuantity('nanovolt', pq.V * 1e-9, symbol='nV')

class UnitFetcherPlaceHolder(dict):
    """
    At the moment there isn't really a robust way to get the correct
    units for all variables. 

    This can be set explicity for each possible variable that is recordable
    from the cell type, but with our custom TraceSpecRecorder the variable name 
    can be anything.
    """
    def __getitem__(self, key):
        """
        Return units according to trace naming convention.

        @return     units : str
                    Unit string accepted by 'Quantities' python package
        """
        if key.lower().startswith('v'):
            return 'mV'
        elif key.lower().startswith('isyn'):
            return 'nA'
        elif key.lower().startswith('gsyn'):
            return 'uS' # can also be nS but this is for our GABAsyn/GLUsyn
        elif key.lower().startswith('i'):
            return 'mA/cm^2' # membrane currents of density mechanisms
        elif key.lower().startswith('g'):
            return 'S/cm^2' # membrane conductances of density mechanisms
        elif key.lower().startswith('lfp'):
            return 'nV' # LFP calculator mechanism 'LfpSumStep'
        elif key == 'spikes':
            return 'ms'
        else:
            return 'dimensionless'


class EphysCellType(NativeCellType):
    """
    PyNN native cell type that has Ephys model as 'model' attribute.

    Attributes
    ----------

    @attr   units : UnitFetcherPlaceHolder

            Required by PyNN, celltype must have method `units(varname)` that
            returns the units of recorded variables

    
    @attr   receptor_types : list(str)

            Required by PyNN: receptor types accepted by Projection constructor.
            This attribute is created dynamically by combining
            celltype.model.region with the celltype.receptor_types declared
            in the subclass
    """

    # Population.find_units() queries this for units
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
        
        # Parameter names defined in class namespace (not Ephys parameters)
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


class PynnCellModelBase(object):
    """
    Base functionality for our custom PyNN cell models.

    This class implements all methods required to work with our custom
    Connector and Recorder classes or marks them as abstract methods
    for implementation in the subclass.
    """
    # FIXME: Problems with conflicting metaclass in subclass
    # __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        """
        As opposed to the original CellModel class,
        this class instantiates the cell in its __init__ method.

        @param      **kwargs : dict(str, object)
                    Parameter name, value pairs

        @post       self.icell contains the instantiated Hoc cell model

        """

        # Create cell in NEURON (Sections, parameters)
        self.instantiate()

        # NOTE: default params will be passed by pyNN Population
        for param_name, param_value in kwargs.iteritems():
            if param_name in self.parameter_names:
                # self.parameter_names is a list defined in the subclass body.
                # User is responsible for handling parameters in subclass methods.
                setattr(self, param_name, param_value)
            else:
                logger.warning("Unrecognized parameter {}. Ignoring.".format(param_name))

        # Make synapse data structure in format 
        # {'region': {'receptors': list({ 'synapse':syn, 'used':int }) } }
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


    # @abstractmethod
    def memb_init(self):
        """
        Set initial values for all variables in this cell.

        @override   memb_init() required by PyNN interface for cell models.
        """
        for sec in self.icell.all:
            for seg in sec:
                seg.v = self.v_init # set using pop.init(v=v_init) or default_initial_values


    # @abstractmethod
    def get_threshold(self):
        """
        Get spike threshold for self.source variable (usually points to membrane
        potential). This threshold is used when creating NetCon connections.

        @override   get_threshold() required by pyNN interface for cell models.

        @return     threshold : float
        """
        raise NotImplementedError("Please set the spike threshold for your "
                                  "custom cell model.")


    def _init_synapses(self):
        """
        Initialize synapse map.

        By default it just creates an empty synapse map, but the subclass
        can override this to create synapses upon cell creation.

        @post       self._synapses is a nested dict with depth=2 and following
                    structure:
                    { region: {receptors: list({ 'synapse':syn, 'used':int }) } }
        """
        self._synapses = {region: {} for region in self.regions}


    # @abstractmethod
    def get_synapses(self, region, receptors, num_contacts, **kwargs):
        """
        Get synapse in subcellular region for given receptors.
        Called by Connector object to get synapse for new connection.

        Parameters
        ---------

        @param      region : str
                    Region descriptor.

        @param      receptors : enumerable(str)
                    Receptor descriptors.

        @param      mark_used : bool
                    Whether synapse is 'consumed' by caller and should be 
                    marked as used.

        @param      **kwargs
                    (Unused) extra keyword arguments for use when overriding
                    this method in a subclass.
        
        Returns
        -------

        @return     synapse, num_used : tuple(nrn.POINT_PROCESS, int)
                    
                    NEURON point process object that can serve as the
                    target of a NetCon object, and the amount of times
                    the synapse has been used before as the target
                    of a NetCon.
        """
        raise NotImplementedError("Implement get_synapse() in subclass by "
                                  "making use of any of the available methods. "
                                  "E.g. get_existing_synapse(), ... ")


    def make_synapses_cached_region(self, region, receptors, num_synapses, **kwargs):
        """
        Make a new synapse by sampling a cached region.

        @pre    model must have attribute '_cached_region_segments' containing a
                dict[str, list(nrn.Segment)] that maps the region name
                to eligible segments in that region.
        """
        rng = kwargs.get('rng', np.random)
        region_segs = self._cached_region_segments[region]
        seg_ids = rng.choice(len(region_segs), 
                             num_synapses, replace=False)
        if 'mechanism' in kwargs:
            mech_name = kwargs['mechanism']
        elif all((rec in ('AMPA', 'NMDA') for rec in receptors)):
            mech_name = self.default_GLU_mechanism
        elif all((rec in ('GABAA', 'GABAB') for rec in receptors)):
            mech_name = self.default_GABA_mechanism

        return [getattr(h, mech_name)(region_segs[i]) for i in seg_ids]


    def make_new_synapse(self, receptors, segment, mechanism=None):
        """
        Make a new synapse that implements given receptors in the
        given segment.

        @see    get_synapse() for documentation
        """
        if mechanism is not None:
            syn = getattr(h, mechanism)(segment)
        elif all((rec in ('AMPA', 'NMDA') for rec in receptors)):
            syn = getattr(h, self.default_GLU_mechanism)(segment)
        elif all((rec in ('GABAA', 'GABAB') for rec in receptors)):
            syn = getattr(h, self.default_GABA_mechanism)(segment)
        else:
            raise ValueError("No synapse mechanism found that implements all "
                             "receptors {}".format(receptors))
        return syn


    def make_synmap_tree():
        """
        Initialize data structure that tracks location of synapses in
        dendritic tree.
        """
        # Algorithm idea:
        # - each section is a node of a tree datastructure
        # - each node keeps its own length =L=, number of synapses =nsyn=, and region =region=
        # - each node can query same properties for entire subtree by recursive descent
        # - to find a section:
        #     - query subtree =L=, =nsyn=, filter nodes using =region=
        #         - no nodes with matching region: =L=0=
        #     - descend to child branch where =nsyn/L= is smaller than subtree average
        #     - if node's =nsyn/L= is smaller than stored average: stop
        raise NotImplementedError()



    # def make_synapse_balanced(self, region, receptors, mechanism=None):
    #     """
    #     Make a new synapse in location so that synapses are distributed 
    #     in balanced way over dendritic tree.
    #     """
    #     # TODO: decide approach
    #     # - if completely random -> get unbalanced tree
    #     # - if completely balanced -> unrealistic layout
    #     # - decide at which point the choice will be random rather than smallest density
        
    #     # TODO: move execute-once code to _init_synapses
    #     sibling_synapses = self.get_synapse_list(region, receptors)
    #     total_siblings = len(sibling_synapses)
    #     dendritic = list(self.icell.basal) + list(self.icell.apical)
    #     dendritic_segments = [seg for sec in dendritic for seg in sec \
    #                             if self.segment_in_region(seg, region)]
    #     region_length = sum((seg.sec.L/seg.sec.nseg for seg in dendritic_segments))
    #     synapse_density = total_siblings / region_length

    #     # Get roots of dendrite
    #     dendrite_roots = set()
    #     for sec in self.icell.somatic:
    #         for child in sec.children():
    #             if child in dendritic:
    #                 dendrite_roots.add(child)

    #     # Ascend breadth-first, check subtree, pick random if criterion OK
    #     queue = list(dendrite_roots)
    #     while queue:
    #         node = queue.pop(0) # FIFO queue
    #         # Get entire subtree of current node
    #         subtree_secs = h.SectionList()
    #         subtree_secs.subtree(sec=node)
    #         subtree_list = list(subtree_secs)
    #         region_segments = [seg for sec in subtree_secs for seg in sec \
    #                             if self.segment_in_region(seg, region)]
    #         region_length = sum((seg.sec.L/seg.sec.nseg for seg in region_segments))
    #         num_siblings = sum((1.0 for syn in sibling_synapses if (
    #                             syn.get_segment().sec in subtree_list)))
    #         region_density = num_siblings / region_length
            
    #         if region_length > 0 and region_density < synapse_density:
    #             # pick random segment in subtree (filter by region) with nsyn/L < target
    #             # - set new target density to subtree density
    #             # - pick random segment with density < subtree density
    #             target_segments = []
    #             return pick_random_segment(region_segments)

    #         for child in node.children():
    #             queue.append(child)
        
    #     # If this point reached: tree is perfectly balanced -> imbalance it
    #     # pick random section in whole tree (filter by region)
    #     return pick_random_segment(dendritic_segments)


    # def get_existing_synapse(self, region, receptors, mark_used):
    #     """
    #     Get existing synapse in region with given receptors.

    #     This method is useful for cells that create all synapses upon
    #     initialization.

    #     @see    get_synapse() for documentation
    #     """
    #     syn_list = self.get_synapse_list(region, receptors)
    #     if len(syn_list) == 0:
    #         raise Exception("Could not find any synapses for "
    #             "region '{}' and receptors'{}'".format(region, receptors))

    #     min_used = min((meta.used for meta in syn_list))
    #     metadata = next((meta for meta in syn_list if meta.used==min_used))
    #     if mark_used:
    #         metadata.used += 1 # increment used counter
    #     return metadata.synapse, min_used


    def get_synapse_list(self, region, receptors):
        """
        Get synapses in region that implements all given receptors.

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
        Get synapses of given NEURON mechanism.

        @param      mechanism : str
                    NEURON mechanism name

        @return     synapse_list : list(synapse)
                    List of synapses in region
        """
        return sum((synlist for region in self._synapses.values() for recs, synlist in region.items()), [])


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
        # Deconstruct synapse specifier
        matches = re.search(r'^(?P<mechname>\w+)(\[(?P<slice>[\d:]+)\])', spec)
        mechname = matches.group('mechname')
        slice_expr = matches.group('slice') # "i:j:k"
        slice_parts = slice_expr.split(':') # ["i", "j", "k"]
        slice_parts_valid = [int(i) if i!='' else None for i in slice_parts]
        
        synlist = self.get_synapses_by_mechanism(mechname)
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


        @note       The default Section arrays for Ephys.CellModel are:
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


class EphysModelWrapper(ephys.models.CellModel, PynnCellModelBase):
    """
    Subclass of Ephys CellModel that conforms to the interface required
    by the 'model' attribute of a PyNN CellType class.

    This is a modified version of an Ephys CellModel that allows multiple
    instances of a cell to be created. As opposed to the original CellModel class,
    this class instantiates the cell in its __init__ method.


    @note   Called by ID._build_cell() defined in
            https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/simulator.py
            as follows:
            
                > ID._cell = Population.cell_type.model(**cell_parameters)


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
    

    @attr   locations : dict[str, Ephys.location]

            Dict containing all locations defined on the cell.
            Keys are the same as the location attributes of this cell.


    @attr   synapses : dict[str, list(nrn.POINT_PROCESS)]

            Synapse object that synapse onto this cell.
    """

    __metaclass__ = CellModelMeta

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

        # Cell must be instantiated _before_ applying parameters
        self.sim = ephys_sim_from_pynn()
        self.instantiate(sim=self.sim)

        # NOTE: default params will be passed by pyNN Population
        for param_name, param_value in kwargs.iteritems():
            if (param_name in self.params) and (self.params[param_name].value == param_value):
                # self.params are the Ephys parameters : these are already set
                # in `self.instantiate()` so don't set them again
                continue
            elif (param_name in self.parameter_names) or param_name.endswith('_scale'):
                # self.parameter_names is a list defined in the subclass body.
                # Ephys parameters are also added to to this list (see metaclass).
                # User is responsible for handling parameters in subclass methods.
                setattr(self, param_name, param_value)
            else:
                logger.warning("Unrecognized parameter {}. Ignoring.".format(param_name))

        # Add locations / regions
        # self.locations = {}
        # for static_loc in self._ephys_locations:
        #     loc = copy(static_loc)
        #     # loc.sim = self.sim
        #     # loc.icell = self.icell
        #     self.locations[make_valid_attr_name(loc.name)] = loc

        # Synapses will map the mechanism name to the synapse object
        # and is set by our custom Connection class
        # self.synapses = {} # mech_name: str -> list(nrn.POINT_PROCESS)
        # Make synapse objects as targets for connections
        self._init_synapses()
        self._post_instantiate()

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
        # self._init_lfp()


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


    def _post_instantiate(self):
        """
        Hook for subclass to put post-instantiation code.
        This is code to be executed after morphology and all sections
        have been created.

        @pre    cell has been instantiated in NEURON: all Sections have been
                created and mechanisms inserted.

        @pre    all parameters passed to the cell have been applied
        """
        pass


#    def get_synapse(self, region, receptors, mark_used, **kwargs):
#        """
#        Get synapse in subcellular region for given receptors.
#        Called by Connector object to get synapse for new connection.
#
#        @override   PynnCellModelBase.get_synapse()
#        """
#        return self.get_existing_synapse(region, receptors, mark_used)


    def _init_lfp(self):
        """
        Initialize LFP sources for this cell.
        Override in subclass to implement this.

        @return     lfp_tracker : nrn.POINT_PROCESS
                    Object with recordable variable that represents the cell's
                    summed LFP contributions
        """
        pass


    def _update_position(self, xyz):
        """
        Called when the cell's position is changed, e.g. when changing 
        the space/structure of the parent Population.

        @effect     Adds xyz to all coordinates of the root sections and then
                    calls h.define_shape() so that whole tree is translated.
        """

        raise NotImplementedError("Subclass should update cell position.")


    def _set_tau_m_scale(self, value):
        """
        Setter for parameter 'tau_m'. Sets membrane time constant.

        @pre    Subclass must define attributes 'gleak_name' and
                'tau_m_scaled_regions'.
        """
        if not hasattr(self, '_tau_m_scale'):
            self._tau_m_scale = 1.0
        if value == self._tau_m_scale:
            return
        
        # If not yet scaled, save the base values for g_leak and cm
        # if self._tau_m_scale == 1.0:
        #     for region_name in scaled_regions:
        #         if region_name == 'all':
        #             continue
        #         sl = list(getattr(self.icell, region_name))
        #         self._base_gl[region_name] = sum((getattr(sec, self.gleak_name) for sec in sl)) / len(sl)
        #         self._base_cm[region_name] = sum((sec.cm for sec in sl)) / len(sl)

        # Extract the mechanism name
        matches = re.search(r'.+_([a-zA-Z0-9]+)$', self.gleak_name)
        gleak_mech = matches.groups()[0]

        # Scale tau_m in all compartments
        for region_name in self.tau_m_scaled_regions:
            for sec in getattr(self.icell, region_name):
                # If section has passive conductance: distribute scale factor 
                # over Rm (gleak) and Cm
                distribute_scale = h.ismembrane(gleak_mech, sec=sec)
                if distribute_scale:
                    cm_factor = math.sqrt(value)
                    gl_factor = 1.0 / math.sqrt(value)
                else:
                    cm_factor = value
                for seg in sec:
                    if distribute_scale:
                        setattr(seg, self.gleak_name, 
                                gl_factor * getattr(seg, self.gleak_name))
                    seg.cm = cm_factor * seg.cm
        
        self._tau_m_scale = value


    def _get_tau_m_scale(self):
        """
        Getter for parameter 'tau_m'. Get average membrane time constant.
        """
        if not hasattr(self, '_tau_m_scale'):
            self._tau_m_scale = 1.0
        return self._tau_m_scale


    tau_m_scale = property(fget=_get_tau_m_scale, fset=_set_tau_m_scale)

    # Ion channel & RANGE variable scaling.
    # - ion channel scaling can be implemented in general way by overriding
    # __getattr__ and __setattr__, and defining a dict that gives the scaled
    # SectionLists per ion channel.
    # - Alternatively, you can make/modify an Ephys parameter.

    def __getattr__(self, name):
        """
        Override getattr to support dynamic properties that are not
        explicitly declared.

        For now, scale factors for NEURON range variables can be set.

        @pre    Subclass must define attribute 'rangevar_names'

        @note   __getattr__ is called as last resort and the default behavior
                is to raise AttributeError
        """
        matches_scale_factor = re.search(r'^(\w+)_scale$', name)
        if (matches_scale_factor is not None) and (any(
            [v.startswith(matches_scale_factor.groups()[0]) for v in self.rangevar_names])):
        
            # search for NEURON RANGE variable to be scaled
            varname = matches_scale_factor.groups()[0]
            private_attr_name = '_' + varname + '_scale'

            # Only if it has been scaled before does the scale attribute exist
            if not hasattr(self, private_attr_name):
                return 1.0 # not scaled
            else:
                # Call will bypass this method if attribute exists
                return getattr(self, private_attr_name)
        else:
            # return super(EphysModelWrapper, self).__getattr__(name)
            # raise AttributeError
            return self.__getattribute__(name)


    def __setattr__(self, name, value):
        """
        Allow scaling of any NEURON RANGE variables by intercepting
        assignments to attributes of the format '<rangevar>_scale', i.e.
        any NEURON range variable names suffixed by '_scale'.

        @pre    Subclass must define attribute 'rangevar_names'
        """
        # Attribute must a declared rangevar name + '_scale'
        matches = re.search(r'^(\w+)_scale$', name)
        if (matches is None) or (not any(
                [v.startswith(matches.groups()[0]) for v in self.rangevar_names])):
            return super(EphysModelWrapper, self).__setattr__(name, value)
        varname = matches.groups()[0]
        private_attr_name = '_' + varname + '_scale'

        if not hasattr(self, private_attr_name):
            old_scale = 1.0
        else:
            old_scale = getattr(self, private_attr_name)
        if value == old_scale:
            return # scale is already correct
        relative_scale = value / old_scale

        # Extract the mechanism name
        matches = re.search(r'.+_([a-zA-Z0-9]+)$', varname)
        if matches is None:
            raise ValueError('Could not extract NEURON mechanism name '
                             'from RANGE variable name {}'.format(varname))
        mechname = matches.groups()[0]

        # Finally, scale range variable
        for region_name in self.rangevar_scaled_regions.get(varname, ['all']):
            # If no regions defined, use SectionList 'all' containing all sections
            for sec in getattr(self.icell, region_name):
                if not h.ismembrane(mechname, sec=sec):
                    continue
                for seg in sec:
                    setattr(seg, varname, relative_scale * getattr(seg, varname))
        
        # Indicate the current scale factor applied to attribute
        setattr(self, private_attr_name, value)
            

