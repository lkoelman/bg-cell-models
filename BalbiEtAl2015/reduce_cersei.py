"""
Reduction of Balbi et al. motoneuron model using CERSEI folding tools
"""

from common.treeutils import ExtSecRef, getsecref
from cersei.collapse.fold_reduction import ReductionMethod, FoldReduction
from cersei.collapse.marasco_folding import assign_identifiers_dfs

from neuron import h

# Make Gillies model files findable
cell_model_dir = "../BalbiEtAl2015"
import sys
sys.path.append(cell_model_dir)

################################################################################
# Cell model-specific implementations of reduction functions
################################################################################

class BalbiFoldReduction(FoldReduction):
    """
    Model-specific functions for folding reduction of Balbi et al. (2015)
    motoneuron model.
    """

    def assign_initial_identifiers(reduction):
        """
        Assign identifiers to Sections.
        """

        assign_identifiers_dfs(reduction._root_ref, reduction.all_sec_refs)


    def assign_new_identifiers(reduction, node_ref, all_refs, par_ref=None):
        """
        Assign identifiers to newly created Sections
        """

        # assign a unique cell GID
        if not hasattr(node_ref, 'gid'):
            # Assume that only collapsed sections have no gid
            node_ref.gid = node_ref.zip_id

        childsecs = node_ref.sec.children()
        childrefs = [getsecref(sec, all_refs) for sec in childsecs]
        for childref in childrefs:
            reduction.assign_new_identifiers(childref, all_refs, parref=node_ref)


    def get_interpolation_path_sections(reduction, secref):
        """
        Return sections along path from soma to distal end of dendrites used
        for interpolating dendritic properties.

        TODO: determine based on secref's gid
        """
        # Choose stereotypical path for interpolation
        # Get path from root node to this sections
        start_sec = reduction._root_ref.sec
        end_sec = h.dend[300] # TODO: choose a representative terminal section?
        # dend[9] 9, 76, 305, 378, 300 # terminal sections with mechanism 'L_Ca'
        # dend[300] # terminal section with both mechanisms 'L_Ca' and 'mAHP'
        
        calc_path = h.RangeVarPlot('v')
        
        # Start in the utmost root
        start_sec.push()
        calc_path.begin(0.5)
        
        # End in a terminal section
        end_sec.push()
        calc_path.end(0.5)
        
        # let NEURON traverse the path
        root_path = h.SectionList()
        calc_path.list(root_path) # store path in list
        
        h.pop_section()
        h.pop_section()

        return list(root_path)


    def set_ion_styles(reduction):
        """
        Set correct ion styles for each Section.
        """
        # Set ion styles
        # TODO: look up in orig_tree_props


    def fix_topology_below_roots(reduction):
        """
        Assign topology numbers for sections located below the folding roots.

        @note   assigned to key 'fix_topology_func'
        """
        # TODO: set topology numbers, see how they are used


################################################################################
# Gillies Model Reduction Experiments
################################################################################

def balbi_marasco_reduction(tweak=True):
    """
    Make FoldReduction object with Marasco method.
    """

    import balbi_model
    BALBI_CELL_ID = 1
    balbi_model.make_cell_balbi(model_no=BALBI_CELL_ID)
    named_secs = balbi_model.get_named_sec_lists()

    # Group subtrees by trunk
    somatic = list(named_secs['soma'])
    # Dendritic sections have other subtrees (trunks = first level branchpoints)
    dendritic = list(named_secs['dend'])
    # Axonic sections are first subtree (trunk = AH)
    axonic = sum((list(named_secs[name]) for 
                    name in ('AH', 'IS', 'node', 'MYSA', 'FLUT', 'STIN')), [])
    
    nonsomatic = axonic + dendritic

    # Get the folding branchpoints for the dendritic subtrees
    dend_refs = [ExtSecRef(sec=sec) for sec in dendritic]
    root_ref = getsecref(named_secs['dend'][0], dend_refs) # root section of dendritic tree

    import treeutils as tree, cluster as clu
    clu.assign_topology_attrs(root_ref, dend_refs)
    trunk_refs = tree.find_roots_at_level(2, root_ref, dend_refs) # level 2 roots
    
    fold_roots = [named_secs['AH'][0]]
    fold_roots.extend([ref.sec for ref in trunk_refs])

    # Parameters for reduction
    # TODO: read ion concentrations after SaveState.restore() and write them in new function so it can be executed after modification of the model
    def motocell_setstate():
        """ Initialize cell for analyzing electrical properties """
        balbi_model.motocell_steadystate(BALBI_CELL_ID) # load correct states file

    # Reduce model
    red_method = ReductionMethod.Marasco
    reduction = FoldReduction(somatic, nonsomatic, fold_roots, red_method)

    # Reduction parameters
    reduction.gleak_name = balbi_model.gleak_name
    reduction.mechs_gbars_dict = balbi_model.balbi_gdict
    reduction.set_reduction_params(red_method, {
        'Z_freq' :              25.,
        'Z_init_func' :         motocell_setstate,
        'Z_linearize_gating' :  False,
        'gbar_scaling' :        'area',
        'syn_map_method' :      'Ztransfer',
        'post_tweak_funcs' :    [],
    })

    return reduction


def fold_balbi_marasco(export_locals=True):
    """
    Fold Gillies STN model using given reduction method
    
    @param  export_locals       if True, local variables will be exported to the global
                                namespace for easy inspection
    """
    # Make reduction object
    reduction = balbi_marasco_reduction()
    
    # Do reduction
    # TODO: in FoldReduction _*_FUNCS: go over functions and remove gillies-specific code
    reduction.reduce_model(num_passes=7)
    reduction.update_refs()

    if export_locals:
        globals().update(locals())

    return reduction._soma_refs, reduction._dend_refs

if __name__ == '__main__':
    fold_balbi_marasco()