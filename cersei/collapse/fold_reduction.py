"""
Object-oriented interface for various compartmental cell reduction methods.

@author Lucas Koelman
@date   24-08-2017
"""

from common.treeutils import ExtSecRef, getsecref
from neuron import h

from fold_algorithm import ReductionMethod
import marasco_folding as marasco
import stratford_folding as stratford
import mapsyn, redutils

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

    TODO: make interface/virtual class CollapsingAlgorithm and let 
    Marasco and Stratford algorithms implement methods
    """

    # For each step in reduction process: a dict mapping ReductionMethod -> (func, arg_names)
    
    _PREPROC_FUNCS = {
        ReductionMethod.Marasco:    (marasco.preprocess_impl, []),
        ReductionMethod.Stratford:  (stratford.preprocess_impl, []),
    }

    _PREP_FOLD_FUNCS = {
        ReductionMethod.Marasco:    (marasco.prepare_folds_impl, []),
        ReductionMethod.Stratford:  (stratford.prepare_folds_impl, []),
    }

    _CALC_FOLD_FUNCS = {
        ReductionMethod.Marasco:    (marasco.calc_folds_impl, []),
        ReductionMethod.Stratford:  (stratford.calc_folds_impl, []),
    }

    _MAKE_FOLD_EQ_FUNCS = {
        ReductionMethod.Marasco:    (marasco.make_folds_impl, []),
        ReductionMethod.Stratford:  (stratford.make_folds_impl, []),
    }

    _POSTPROC_FUNCS = {
        ReductionMethod.Marasco:    (marasco.postprocess_impl, []),
        ReductionMethod.Stratford:  (stratford.post_process_impl, []),
    }

    # Make accessible by step
    _REDUCTION_STEPS_FUNCS = {
        'preprocess':       _PREPROC_FUNCS,
        'prepare_fold':     _PREP_FOLD_FUNCS,
        'calculate_fold':   _CALC_FOLD_FUNCS,
        'make_fold':        _MAKE_FOLD_EQ_FUNCS,
        'postprocess':      _POSTPROC_FUNCS,
    }


    def register_reduction_step(self, method, step, func, kwarg_list=None):
        """
        Register function to perform reduction step 'step' for reduction method 'method'/

        @param  method          ReductionMethod member

        @param  step            ReductionStep member
        """
        if kwarg_list is None:
            kwarg_list = []
        self._REDUCTION_STEPS_FUNCS[step][method] = (func, kwarg_list)


    def __init__(
            self, 
            soma_secs, 
            dend_secs, 
            fold_root_secs, 
            method,
            mechs_gbars_dict=None,
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
        self._REDUCTION_PARAMS = dict((method, {}) for method in list(ReductionMethod))

        # Reduction method
        self.active_method = method
        self.gleak_name = gleak_name
        if mechs_gbars_dict is None:
            raise ValueError("Must provide mechanisms to conductances map!")
        self.set_mechs_gbars_dict(mechs_gbars_dict)

        # Find true root section
        first_root_ref = ExtSecRef(sec=soma_secs[0])
        root_sec = first_root_ref.root # pushes CAS
        
        # Save unique sections
        self._soma_refs = [ExtSecRef(sec=sec) for sec in soma_secs]
        self._dend_refs = [ExtSecRef(sec=sec) for sec in dend_secs]

        # Save ion styles
        ions = ['na', 'k', 'ca']
        self._ion_styles = dict(((ion, h.ion_style(ion+'_ion')) for ion in ions))
        h.pop_section() # pops CAS

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
    

    def get_mechs_gbars_dict(self):
        """
        Get dictionary of mechanism names and their conductances.
        """
        return self._mechs_gbars_dict
    

    def set_mechs_gbars_dict(self, val):
        """
        Set mechanism names and their conductances
        """
        self._mechs_gbars_dict = val
        if val is not None:
            self.gbar_names = [gname+'_'+mech 
                                    for mech,chans in val.iteritems() 
                                        for gname in chans]
            self.active_gbar_names = list(self.gbar_names)
            self.active_gbar_names.remove(self.gleak_name)

    # make property
    mechs_gbars_dict = property(get_mechs_gbars_dict, set_mechs_gbars_dict)


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


    def set_reduction_params(self, method, params):
        """
        Set parameters for given reduction method.
        """
        self._REDUCTION_PARAMS[method] = params


    def get_reduction_param(self, method, param):
        """
        Get reduction parameter for given method.
        """
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


    def _exec_reduction_step(self, step, method, step_args=None):
        """
        Execute reduction step 'step' using method 'method'
        """
        try:
            func, arg_names = self._REDUCTION_STEPS_FUNCS[step][method]
        
        except KeyError:
            raise NotImplementedError("{} function not implemented for "
                                      "reduction method {}".format(step, method))
        
        else:
            user_params = self._REDUCTION_PARAMS[method]
            user_kwargs = dict((kv for kv in user_params.iteritems() if kv[0] in arg_names)) # get required args
            
            if step_args is None:
                step_args = []

            logger.anal("Executing reduction step {}".format(step))
            
            func(*step_args, **user_kwargs)


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
        self.assign_initial_identifiers()

        # Save Section properties for whole tree
        ## Calculate path-accumulated properties for entire tree
        for secref in self.all_sec_refs:
            # Assign path length, path resistance, electrotonic path length to each segment
            redtools.sec_path_props(secref, f_lambda, gleak_name)
        
        ## Which properties to save
        range_props = dict(self.mechs_gbars_dict) # RANGE properties to save
        range_props.update({'': ['diam', 'cm']})
        sec_custom_props = ['gid', 'pathL0', 'pathL1', 'pathri0', 'pathri1', 'pathLelec0', 'pathLelec1']
        seg_custom_props = ['pathL_seg', 'pathri_seg', 'pathL_elec']
        
        ## Build tree data structure
        self.orig_tree_props = redutils.save_tree_properties_ref(
                                    self._root_ref, range_props,
                                    sec_assigned_props=sec_custom_props,
                                    seg_assigned_props=seg_custom_props)

        # Custom preprocessing function
        self._exec_reduction_step('preprocess', method, step_args=[self])

        # Compute synapse mapping info
        if any(self._syns_tomap):

            # Existing synapse attributes to save (SectionRef attributes)
            save_ref_attrs = ['table_index', 'tree_index', 'gid']

            # Mapping parameters
            Z_freq          = self.get_reduction_param(method, 'Z_freq')
            init_func       = self.get_reduction_param(method, 'Z_init_func')
            linearize_gating= self.get_reduction_param(method, 'Z_linearize_gating')
            
            # Compute mapping data
            syn_info = mapsyn.get_syn_info(self.soma_refs[0].sec, self.all_sec_refs, 
                                syn_tomap=self._syns_tomap, Z_freq=Z_freq, 
                                init_cell=init_func, linearize_gating=linearize_gating,
                                save_ref_attrs=save_ref_attrs)

            self._map_syn_info = syn_info


    def prepare_folds(self, method):
        """
        Prepare next fold operation.
        """
        self._exec_reduction_step('prepare_fold', method, step_args=[self])


    def calc_folds(self, method, i_pass):
        """
        Fold branches at branch points identified by given criterion.
        """
        self._exec_reduction_step('calculate_fold', method, step_args=[self, i_pass])


    def make_fold_equivalents(self, method):
        """
        Make equivalent Sections for branches that have been folded.
        """
        self._exec_reduction_step('make_fold', method, step_args=[self])


    def postprocess_cell(self, method=None):
        """
        Post-process cell: perform any additional modifications for reduction
        algorithm.
        """
        if method is None:
            method = self.active_method

        # Execute custom postprocess function
        self._exec_reduction_step('postprocess', method, step_args=[self])
        

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
            Z_freq          = self.get_reduction_param(method, 'Z_freq')
            init_func       = self.get_reduction_param(method, 'Z_init_func')
            linearize       = self.get_reduction_param(method, 'Z_linearize_gating')
            map_method      = self.get_reduction_param(method, 'syn_map_method')
            
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

        # Start reduction process
        self.preprocess_cell(method)

        # Fold one pass at a time
        for i_pass in xrange(num_passes):
            self.prepare_folds(method)
            self.calc_folds(method, i_pass)
            self.make_fold_equivalents(method)
            logger.debug('Finished folding pass {}'.format(i_pass))

        # Finalize reduction process
        self.postprocess_cell(method)

        # Map synapses
        if map_synapses:
            self.map_synapses(method=method)

        logger.debug('Finished cell reduction with method {}'.format(method))

    ############################################################################
    # Cell model-specific / virtual methods
    ############################################################################

    def assign_initial_identifiers(self):
        """
        Assign identifiers to Sections.

        @post   all SectionRef.gid attributes are set
        """
        raise NotImplementedError(
                "This function is model-specific and must be implemented for "
                "each cell model individually.")


    def assign_new_identifiers(self, node_ref, all_refs, par_ref=None):
        """
        Assign identifiers to newly created Sections

        @post   all SectionRef.gid attributes are set for newly created Sections
        """
        raise NotImplementedError(
                "This function is model-specific and must be implemented for "
                "each cell model individually.")


    def get_interpolation_path_sections(self, secref):
        """
        Return Sections forming a path from soma to dendritic terminal/endpoint. 
        The path is used for interpolating spatially non-uniform properties.
        
        @param  secref      <SectionRef> section for which a path is needed

        @return             <list(Section)>
        """
        raise NotImplementedError(
                "This function is model-specific and must be implemented for "
                "each cell model individually.")


    def set_ion_styles(self, secref):
        """
        Set correct ion styles for each Section.

        @param  secref  <SectionRef> section to set ion styles for

        @effect         By default, ion styles of merged Sections are looked up
                        in the saved original tree properties and copied to the
                        sections (if they are all the same). This method may be
                        overridden.
        """
        # Look up ion styles info of sections that have been merged into this one
        filter_fun = lambda node: node.gid in secref.zipped_sec_gids
        merged_props = redutils.find_secprops(reduction.orig_tree_props, filter_fun)
        ionstyles_dicts = [p.ion_styles for p in merged_props]

        # They must all be the same or we have a problem with mechanisms
        final_styles = ionstyles_dicts[0]
        for styles_dict in ionstyles_dicts[1:]
            for ion, style_flags in styles_dict.iteritems()
                if final_styles[ion] != style_flags:
                    raise ValueError(
                        "Cannot merge Sections with distinct ion styles! "
                        "Distinct styles found for sections with gids {} "
                        "for ion {}".format(secref.zipped_sec_gids, ion))

        # Copy styles to target Section
        redutils.set_ion_styles(secref.sec, **final_styles)


    def fix_topology_below_roots(self):
        """
        Assign topology numbers for sections located below the folding roots.

        @note   assigned to key 'fix_topology_func'
        """
        raise NotImplementedError(
                "This function is model-specific and must be implemented for "
                "each cell model individually.")