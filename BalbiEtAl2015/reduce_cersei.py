"""
Reduction of Balbi et al. motoneuron model using CERSEI folding tools
"""

from common.treeutils import ExtSecRef, getsecref
from cersei.collapse.fold_reduction import ReductionMethod, FoldReduction
from cersei.collapse.marasco_folding import assign_identifiers_dfs

# Make model files findable
import balbi_model
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

    def __init__(self, **kwargs):
        """
        Initialize new Balbi cell model reduction.

        @param  balbi_motocell_id   (int) morphology file identifier
        """
        self.balbi_motocell_id = kwargs.pop('balbi_motocell_id')
        super(BalbiFoldReduction, self).__init__(**kwargs)


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


    def fix_topology_below_roots(reduction):
        """
        Assign topology numbers for sections located below the folding roots.
        """
        # TODO: set topology numbers, see how they are used
        pass


    # @FoldReduction.reduction_param(ReductionMethod.Marasco, 'Z_init_func')
    def init_cell_steadystate(self):
        """
        Initialize cell for analyzing electrical properties.
        
        TODO: read ion concentrations after SaveState.restore() once and
        hardcode them somewhere to restore them here. This is necessary since
        Balbi's function h.load_steadystate() cannot be called after modification of the model
        """
        balbi_model.motocell_steadystate(self.balbi_motocell_id) # loads correct states file


################################################################################
# Gillies Model Reduction Experiments
################################################################################

def make_fold_reduction(tweak=True):
    """
    Make FoldReduction object with Marasco method.

    TODO: move all cell-specific actions into subclass methods
    """
    # Instantiate NEURON model
    BALBI_CELL_ID = 1 # morphology file to load
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

    # Calculate topology attributes
    import treeutils as tree, cluster as clu
    clu.assign_topology_attrs(root_ref, dend_refs)
    trunk_refs = tree.find_roots_at_level(2, root_ref, dend_refs) # level 2 roots
    
    # At which nodes should tree be folded?
    fold_roots = [named_secs['AH'][0]]
    fold_roots.extend([ref.sec for ref in trunk_refs])

    # Reduce model
    reduction = BalbiFoldReduction(
                    method=ReductionMethod.Marasco,
                    balbi_motocell_id=BALBI_CELL_ID,
                    soma_secs=somatic, dend_secs=nonsomatic,
                    fold_root_secs=fold_roots, 
                    gleak_name=balbi_model.gleak_name,
                    mechs_gbars_dict=balbi_model.balbi_gdict,
                    mechs_params_dict=balbi_model.mechs_params_dict)

    # Extra Reduction parameters
    reduction.set_reduction_params(ReductionMethod.Marasco, 
        {
            'Z_freq' :              25.,
            'Z_init_func' :         reduction.init_cell_steadystate,
            'Z_linearize_gating' :  False,
            'f_lambda':             100.0,
            'gbar_scaling' :        'area',
            'syn_map_method' :      'Ztransfer',
            'post_tweak_funcs' :    [],
        })

    return reduction



def reduce_model_folding(export_locals=True):
    """
    Reduce cell model using given folding algorithm
    
    @param  export_locals       if True, local variables will be exported to the global
                                namespace for easy inspection
    """
    # Make reduction object
    reduction = make_fold_reduction()
    
    # Do reduction
    # TODO: in FoldReduction _*_FUNCS: go over functions and remove gillies-specific code
    reduction.reduce_model(num_passes=7)
    reduction.update_refs()

    if export_locals:
        globals().update(locals())

    return reduction._soma_refs, reduction._dend_refs


if __name__ == '__main__':
    reduce_model_folding()
