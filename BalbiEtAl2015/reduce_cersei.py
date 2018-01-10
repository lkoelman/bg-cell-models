"""
Reduction of Balbi et al. motoneuron model using CERSEI folding tools
"""

import re
import sys

pkg_root = ".." # root dir for our packages
sys.path.append(pkg_root)

import cersei.collapse.redutils as redutils
import cersei.collapse.cluster as cluster
import common.logutils as logutils
from common.nrnutil import ExtSecRef, getsecref
from cersei.collapse.fold_reduction import ReductionMethod, FoldReduction

import balbi_model
from neuron import h

logutils.setLogLevel('verbose', ['marasco', 'folding'])

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
        named_secs = balbi_model.get_named_sec_lists()

        # Group subtrees by trunk
        somatic = list(named_secs['soma'])
        # Dendritic sections have other subtrees (trunks = first level branchpoints)
        dendritic = list(named_secs['dend'])
        # Axonic sections are first subtree (trunk = AH)
        axonic = sum((list(named_secs[name]) for 
                        name in ('AH', 'IS', 'node', 'MYSA', 'FLUT', 'STIN')), [])
        # Whole cell
        wholecell = sum(named_secs.values(), [])
        
        nonsomatic = axonic + dendritic
        kwargs['soma_secs'] = somatic
        kwargs['dend_secs'] = nonsomatic

        # Get the folding branchpoints for the dendritic subtrees
        all_refs = [ExtSecRef(sec=sec) for sec in wholecell]
        dend_refs = [getsecref(sec, all_refs) for sec in dendritic]
        root_sec = dend_refs[0].root; h.pop_section() # true root, pushes CAS
        root_ref = ExtSecRef(sec=root_sec)

        # Choose root sections for folding operations
        cluster.assign_topology_attrs(root_ref, all_refs)
        trunk_refs = redutils.find_roots_at_level(2, root_ref, dend_refs) # level 2 roots
        
        # At which nodes should tree be folded?
        fold_roots = [named_secs['AH'][0]]
        fold_roots.extend([ref.sec for ref in trunk_refs])
        kwargs['fold_root_secs'] = fold_roots

        # Set all parameters
        kwargs['gleak_name'] = balbi_model.gleak_name
        kwargs['mechs_gbars_dict'] = balbi_model.gbar_dict
        kwargs['mechs_params_nogbar'] = balbi_model.mechs_params_nogbar
        super(BalbiFoldReduction, self).__init__(**kwargs)


    def assign_initial_sec_gids(reduction):
        """
        Assign identifiers to Sections.

        @effect     numbers all sections with a 'gid' attribute
        """
        start_id = 0
        redutils.subtree_assign_gids_dfs(
                    reduction._root_ref,
                    reduction.all_sec_refs,
                    parent_id=start_id)


    def assign_new_sec_gids(reduction, node_ref, all_refs, par_ref=None):
        """
        Assign identifiers to newly created Sections.

        @effect     assigns folded sections' 'zip_id' to their gid
        """

        # assign a unique cell GID
        if not hasattr(node_ref, 'gid'):
            # Assume that only collapsed sections have no gid
            node_ref.gid = node_ref.zip_id

        childsecs = node_ref.sec.children()
        childrefs = [getsecref(sec, all_refs) for sec in childsecs]
        for childref in childrefs:
            reduction.assign_new_sec_gids(childref, all_refs, par_ref=node_ref)


    def assign_region_label(reduction, secref):
        """
        Assign region labels to sections.
        """
        arrayname = re.sub(r"\[\d+\]", "", secref.sec.name())
        if secref.is_original:
            if arrayname == 'soma':
                secref.region_label = 'somatic'
            elif arrayname == 'dend':
                secref.region_label = 'dendritic'
            elif arrayname in ['AH', 'IS', 'node', 'MYSA', 'FLUT', 'STIN']:
                secref.region_label = 'axonic'
            else:
                raise Exception("Unrecognized original section {}".format(secref.sec))
        else:
            secref.region_label = '-'.join(sorted(secref.zipped_region_labels))


    def fix_section_properties(self, new_sec_refs):
        """
        Fix properties of newly created sections.

        @override   FoldReduction.fix_section_properties
        """
        # super call to fix ion styles
        super(BalbiFoldReduction, self).fix_section_properties(new_sec_refs)

        # Set global mechanism parameters again
        h.fix_global_params() # defined in 3_ins_ch.hoc

        # Set region-specific properties
        for ref in new_sec_refs:
            if not 'axonic' in ref.region_label:
                ref.sec.insert('extracellular')
                ref.sec.xraxial = 1e9
                ref.sec.xg = 1e10
                ref.sec.xc = 0
            else:
                raise Exception('Axonic sections should not be collapsed. '
                                '(section {}'.format(ref.sec))


    @staticmethod
    def init_cell_steadystate(self):
        """
        Initialize cell for analyzing electrical properties.
        
        @note   Ideally, we should restore the following, like SaveState.restore():
                - all mechanism STATE variables
                - voltage for all segments (seg.v)
                - ion concentrations (nao,nai,ko,ki, ...)
                - reversal potentials (ena,ek, ...)
        """
        h.celsius = 37

        # NOTE: cannot use h.load_steadystate(). This uses SaveState.restore() which means 
        #       between a save and a restore, you cannot create or delete sections, 
        #       NetCon objects, or point processes, nor change the number of segments, 
        #       insert or delete mechanisms, or change the location of point processes.
        
        # Uniform ion concentrations, verified in all sections after h.load_steadystate()
        h.nai0_na_ion = 10.0
        h.nao0_na_ion = 140.0
        h.ki0_k_ion = 54.0
        h.ko0_k_ion = 2.5
        h.cai0_ca_ion = 5e-5
        h.cao0_ca_ion = 2.0

        h.init()


################################################################################
# Gillies Model Reduction Experiments
################################################################################

def make_fold_reduction():
    """
    Make FoldReduction object with Marasco method.
    """
    # Instantiate NEURON model
    BALBI_CELL_ID = 1 # morphology file to load
    balbi_model.make_cell_balbi(model_no=BALBI_CELL_ID)
    

    # Reduce model
    reduction = BalbiFoldReduction(method=ReductionMethod.Marasco,
                                   balbi_motocell_id=BALBI_CELL_ID)

    # Extra Reduction parameters
    reduction.set_reduction_params({
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
    # TODO: in Folder interface methods: remove gillies-specific code
    reduction.reduce_model(num_passes=7)
    reduction.update_refs()

    if export_locals:
        globals().update(locals())

    return reduction._soma_refs, reduction._dend_refs


if __name__ == '__main__':
    reduce_model_folding()
