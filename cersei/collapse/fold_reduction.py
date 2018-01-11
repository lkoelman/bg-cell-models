"""
Object-oriented interface for various compartmental cell reduction methods.

@author Lucas Koelman
@date   24-08-2017
"""

from abc import abstractmethod, ABCMeta

from common.nrnutil import ExtSecRef, getsecref
from neuron import h

from fold_algorithm import ReductionMethod
import marasco_folding as marasco
import mapsyn, redutils, interpolation as interp

# logging of DEBUG/INFO/WARNING messages
import logging
logging.basicConfig(format='%(levelname)s:%(message)s @%(filename)s:%(lineno)s', level=logging.DEBUG)
logname = 'folding'
logger = logging.getLogger(logname) # create logger for this module


################################################################################
# Reduction classes
################################################################################


class FoldReduction(object):
    """
    Class grouping methods and data used for reducing
    a compartmental cable model of a NEURON cell.
    """

    __metaclass__ = ABCMeta # Python2 way

    _FOLDER_CLASSES = {
        ReductionMethod.Marasco: marasco.MarascoFolder
    }


    def __init__(
            self,
            soma_secs=None,
            dend_secs=None,
            fold_root_secs=None,
            method=None,
            mechs_gbars_dict=None,
            mechs_params_nogbar=None,
            gleak_name=None):
        """
        Initialize reduction of NEURON cell with given root Section.

        @param  soma_secs       list of root Sections for the cell (up to first branch points).
                                This list must not contain any Section in dend_secs

        @param  dend_secs       list(Section): flat list of non-somatic sections

        @param  fold_root_secs  list(Section): where each section is a 'root' branchpoint for
                                folding: i.e. this section and all sections lower in the tree
                                should never be folded/collapsed.

        @param  method          ReductionMethod instance
        """

        # Parameters for reduction method (set by user)
        self._REDUCTION_PARAMS = {method: {} for method in list(ReductionMethod)}

        # Reduction algorithm
        self.active_method = method
        FolderClass = self._FOLDER_CLASSES[method]
        self.folder = FolderClass()
        self.folder.reduction = self # bidirectional association

        # Model mechanisms
        self.gleak_name = gleak_name
        if mechs_gbars_dict is None:
            raise ValueError("Must provide mechanisms to conductances map!")
        self.set_mechs_params(mechs_gbars_dict, mechs_params_nogbar)

        # Find true root section
        first_root_ref = ExtSecRef(sec=soma_secs[0])
        root_sec = first_root_ref.root; h.pop_section() # pushes CAS

        # Save unique sections
        self._soma_refs = [ExtSecRef(sec=sec) for sec in soma_secs]
        self._dend_refs = [ExtSecRef(sec=sec) for sec in dend_secs]

        # Save root sections
        self._root_ref = getsecref(root_sec, self._soma_refs)
        allsecrefs = self.all_sec_refs
        self._fold_root_refs = [getsecref(sec, allsecrefs) for sec in fold_root_secs]

        # Set NetCon to be mapped
        self._syns_tomap = []
        self._map_syn_info = []


    @property
    def all_sec_refs(self):
        """
        Get list of SectionRef to all sections.
        """
        return list(self._soma_refs) + list(self._dend_refs)


    @property
    def soma_refs(self):
        """
        Get list of SectionRef to somatic sections.
        """
        return self._soma_refs


    @property
    def dend_refs(self):
        """
        Get list of SectionRef to dendritic sections.
        """
        return self._dend_refs

    def set_syns_tomap(self, syns):
        """
        Set synapses to map.

        @param syns     list(SynInfo)
        """
        self._syns_tomap = syns


    @property
    def map_syn_info(self):
        """
        Synapse properties before and after mapping (electrotonic properties etc.)
        """
        return self._map_syn_info


    def set_mechs_params(self, gbar_dict, param_dict):
        """
        Set mechanism names and their conductances
        """
        self.mechs_gbars_dict = gbar_dict
        self.mechs_params_nogbar = param_dict

        # Merge two dicts
        self.mechs_params_all = dict(param_dict)
        for mech, gbars in gbar_dict.iteritems():
            if mech in self.mechs_params_all:
                old_params = self.mechs_params_all[mech]
                new_params = list(set(old_params + gbars))
            else:
                new_params = list(gbars)
            self.mechs_params_all[mech] = new_params

        self.gbar_names = [gname+'_'+mech for mech,chans in gbar_dict.iteritems()
                                            for gname in chans]
        self.active_gbar_names = list(self.gbar_names)
        self.active_gbar_names.remove(self.gleak_name)



    def update_refs(self, soma_refs=None, dend_refs=None):
        """
        Update Section references after sections have been created/destroyed/substituted.

        @param soma_refs    list of SectionRef to at least all new soma sections
                            (may also contain existing sections)

        @param dend_refs    list of SectionRef to at least all new dendritic sections
                            (may also contain existing sections)
        """
        # Destroy references to deleted sections
        self._soma_refs = [ref for ref in self._soma_refs if ref.exists()]
        self._dend_refs = [ref for ref in self._dend_refs if ref.exists()]

        # Add newly created sections
        if soma_refs is not None:
            self._soma_refs = list(set(self._soma_refs + soma_refs)) # get unique references

        if dend_refs is not None:
            self._dend_refs = list(set(self._dend_refs + dend_refs)) # get unique references


    def set_reduction_params(self, params, method=None):
        """
        Set entire parameters dict for reduction
        """
        if method is None:
            method = self.active_method
        self._REDUCTION_PARAMS[method] = params


    def set_reduction_param(self, method, pname, pval):
        """
        Set a reduction parameter.
        """
        self._REDUCTION_PARAMS[method][pname] = pval


    def get_reduction_param(self, param, method=None):
        """
        Get reduction parameter for given method.
        """
        if method is None:
            method = self.active_method
        
        return self._REDUCTION_PARAMS[method][param]


    def destroy(self):
        """
        Release references to all stored data
        """
        # Parameters for reduction method (set by user)
        self._REDUCTION_PARAMS = None
        self._mechs_gbars_dict = None

        self._soma_refs = None
        self._dend_refs = None
        self._root_ref = None
        self._fold_root_refs = None

        self._syns_tomap = None
        self._map_syn_info = None


    def preprocess_cell(self, method):
        """
        Pre-process cell: calculate properties & prepare data structures
        for reduction procedure

        @param  method      ReductionMethod instance: the reduction method that we
                                should preprocess for.

        @pre        The somatic and dendritic sections have been set

        @post       Computed properties will be available as attributes
                    on Section references in _soma_refs and _dend_refs,
                    in addition to other side effects specified by the
                    specific preprocessing function called.
        """
        # Assign initial identifiers (gid)
        self.assign_initial_sec_gids()

        for secref in self.all_sec_refs:
            # Calculate path-accumulated properties for entire tree:
            redutils.sec_path_props(secref, 
                                    self.get_reduction_param('f_lambda'), 
                                    self.gleak_name)
            secref.is_original = True
            self.assign_region_label(secref)

        # Save Section properties for whole tree
        ## Which properties to save
        range_props = dict(self.mechs_params_all) # RANGE properties to save
        range_props.update({'': ['diam', 'cm']})
        sec_custom_props = ['gid', 'pathL0', 'pathL1', 'pathri0', 'pathri1', 
                            'pathLelec0', 'pathLelec1']
        seg_custom_props = ['pathL_seg', 'pathri_seg', 'pathL_elec']

        ## Build tree data structure
        self.orig_tree_props = redutils.save_tree_properties_ref(
                                    self._root_ref, self.all_sec_refs, range_props,
                                    sec_assigned_props=sec_custom_props,
                                    seg_assigned_props=seg_custom_props)

        # Custom preprocessing function
        self.folder.preprocess_reduction()

        # Compute synapse mapping info
        if any(self._syns_tomap):

            # Existing synapse attributes to save (SectionRef attributes)
            save_ref_attrs = ['table_index', 'tree_index', 'gid']

            # Mapping parameters
            Z_freq = self.get_reduction_param('Z_freq', method)
            init_func = self.get_reduction_param('Z_init_func', method)
            linearize_gating = self.get_reduction_param('Z_linearize_gating', method)

            # Compute mapping data
            syn_info = mapsyn.get_syn_info(
                                self.soma_refs[0].sec,
                                self.all_sec_refs,
                                syn_tomap=self._syns_tomap,
                                Z_freq=Z_freq,
                                gleak_name=self.gleak_name,
                                init_cell=init_func,
                                linearize_gating=linearize_gating,
                                save_ref_attrs=save_ref_attrs)

            self._map_syn_info = syn_info


    def postprocess_fold(self, new_sec_refs):
        """
        Operations performed after equivalent sections have been created for
        current folding pass.
        """
        for ref in new_sec_refs:
            self.assign_region_label(ref)
        
        self.fix_section_properties(new_sec_refs)

        # prepare for next iteration
        self.update_refs(dend_refs=new_sec_refs)


    def map_synapses(self, method=None):
        """
        Map any synapses if present

        @see    set_syns_tomap() for setting synapses.

        @pre    Any synapses provided through syns_tomap must be preprocessed
                for mapping (electronic properties computed), with results
                stored in map_syn_info attribute.
        """
        if method is None:
            method = self.active_method

        # Map synapses to reduced cell
        if any(self._map_syn_info):
            logger.debug("Mapping synapses...")

            # Mapping parameters
            Z_freq          = self.get_reduction_param('Z_freq', method)
            init_func       = self.get_reduction_param('Z_init_func', method)
            linearize       = self.get_reduction_param('Z_linearize_gating', method)
            map_method      = self.get_reduction_param('syn_map_method', method)

            # Map synapses (this modifies syn_info objects)
            mapsyn.map_synapses(self.soma_refs[0], self.all_sec_refs, self._map_syn_info,
                                init_func, Z_freq, linearize_gating=linearize,
                                method=map_method)
        else:
            logger.debug("No synapse data available for mapping.")



    def reduce_model(self, num_passes, method=None, map_synapses=True):
        """
        Do a fold-based reduction of the compartmental cell model.

        @param  num_passes      number of 'folding' passes to be done. One pass corresponds to
                                folding at a particular node level (usually the highest).
        """
        if method is None:
            method = self.active_method
        self.active_method = method # indicate what method we are using

        self.preprocess_cell(method)

        for i_pass in xrange(num_passes):
            # Do a folding pass
            new_refs = self.folder.fold_one_pass(i_pass)

            self.postprocess_fold(new_refs)

            logger.debug('Finished folding pass {}'.format(i_pass))

        # Finalize reduction process
        self.folder.postprocess_reduction()
        self.update_refs()

        # Map synapses onto new sections
        if map_synapses:
            self.map_synapses(method=method)

        logger.debug('Finished cell reduction with method {}'.format(method))

    ############################################################################
    # Cell model-specific / virtual methods
    ############################################################################

    @abstractmethod
    def assign_initial_sec_gids(self):
        """
        Assign identifiers to Sections.

        @post   all SectionRef.gid attributes are set
        """
        raise NotImplementedError(
                "This function is model-specific and must be implemented for "
                "each cell model individually.")


    @abstractmethod
    def assign_new_sec_gids(self, node_ref, all_refs, par_ref=None):
        """
        Assign identifiers to newly created Sections

        @post   all SectionRef.gid attributes are set for newly created Sections
        """
        raise NotImplementedError(
                "This function is model-specific and must be implemented for "
                "each cell model individually.")


    @abstractmethod
    def assign_region_label(self, secref):
        """
        Assigns a region label to the section.

        @post   SectionRef.region_label: <str> is set
        """
        raise NotImplementedError(
                "This function is model-specific and must be implemented for "
                "each cell model individually.")


    def get_interpolation_path_secs(self, secref):
        """
        Return Sections forming a path from soma to dendritic terminal/endpoint.
        The path is used for interpolating spatially non-uniform properties.

        @param  secref      <SectionRef> section for which a path is needed

        @return             <list(Section)>
        """
        return interp.get_interpolation_path_sections(secref, self.orig_tree_props)


    @abstractmethod
    def fix_section_properties(self, new_sec_refs):
        """
        Fix properties of newly created sections.
        """
        raise NotImplementedError(
                "This function is model-specific and must be implemented for "
                "each cell model individually.")


    # @abstractmethod
    def fix_topology_below_roots(self):
        """
        Virtual method: assign topology numbers for sections located below subtrees of folding
        roots. Only neccessary for reduction based on automatic clustering.

        @note   can be called by Folder class
        """
        pass
