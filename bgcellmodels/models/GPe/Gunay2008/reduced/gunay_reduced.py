"""
Morphology reduction of Gunay (2008) GPe cell model
"""

import re
import logging

# import sys
# pkg_root = ".." # root dir for our packages
# sys.path.append(pkg_root)

from bgcellmodels.common import logutils
from bgcellmodels.cersei.collapse.fold_algorithm import ReductionMethod
from bgcellmodels.cersei.collapse.fold_reduction import FoldReduction
from bgcellmodels.models.GPe.Gunay2008 import gunay_model
from neuron import h

logger = logging.getLogger('gunay')
logutils.setLogLevel('verbose', ['gunay', 'marasco', 'folding'])

################################################################################
# Cell model-specific implementations of reduction functions
################################################################################

class GpeCellReduction(FoldReduction):
    """
    Model-specific functions for folding reduction of Gillies & Willshaw (2005)
    STN cell model.
    """

    def __init__(self, **kwargs):
        """
        Make new Gillies model reduction.

        @param  balbi_motocell_id   (int) morphology file identifier
        """

        ephys_cell, nrnsim = gunay_model.create_cell()
        icell = ephys_cell.icell


        # Set reduction parameters
        kwargs['soma_secs'] = list(icell.somatic)
        kwargs['dend_secs'] = list(icell.basal)
        kwargs['fold_root_secs'] = TODO

        # Set all parameters
        kwargs['gleak_name']            = gunay_model.gleak_name
        kwargs['mechs_gbars_dict']      = gunay_model.gbar_dict
        kwargs['mechs_params_nogbar']   = gunay_model.mechs_params_nogbar
        
        super(GpeCellReduction, self).__init__(**kwargs)


    def assign_region_label(reduction, secref):
        """
        Assign region labels to sections.
        """
        TODO: see documentation of this function in FoldReduction
        arrayname = re.sub(r"\[?\d+\]?", "", secref.sec.name())

        # Original sections
        if secref.is_original:
            if arrayname.endswith('soma'):
                secref.region_label = 'somatic'
            elif arrayname.endswith('dend'):
                secref.region_label = 'dendritic'
            else:
                raise Exception("Unrecognized original section {}".format(secref.sec))
        
        # Substituted / equivalent sections
        elif hasattr(secref, 'merged_region_labels'):
            secref.region_label = '-'.join(sorted(secref.merged_region_labels))

        elif hasattr(secref, 'orig_props'):
            secref.region_label = '-'.join(sorted(secref.orig_props.merged_region_labels))


    def fix_section_properties(self, new_sec_refs):
        """
        Fix properties of newly created sections.

        @override   abstract method FoldReduction.fix_section_properties
        """
        for ref in new_sec_refs:
            GpeCellReduction.set_ion_styles(ref)


    @staticmethod
    def init_cell_steadystate():
        """
        Initialize cell for analyzing electrical properties.
        
        @note   Ideally, we should restore the following, like SaveState.restore():
                - all mechanism STATE variables
                - voltage for all segments (seg.v)
                - ion concentrations (nao,nai,ko,ki, ...)
                - reversal potentials (ena,ek, ...)
        """
        # TODO
        pass


    @staticmethod
    def set_ion_styles(secref):
        """
        Set correct ion styles for each Section.

        @note   assigned to key 'set_ion_styles_func'
        """
        # NOTE: h.ion_style(ion, c_style, e_style, einit, eadvance, cinit)
        # TODO
        h.ion_style("na_ion",1,2,1,0,1, sec=secref.sec)
        h.ion_style("k_ion",1,2,1,0,1, sec=secref.sec)
        h.ion_style("ca_ion",3,2,1,1,1, sec=secref.sec)



def make_reduction(method, reduction_params=None, tweak=False):
    """
    Make FoldReduction object using given collasping method

    @param  method : ReductionMethod
            Accepted values are BushSejnowski and Marasco

    @param  reduction_params : dict[str, object]
            Parameters that define the reduction (optional).
            These parameters are determined by the reduction method
            but can be overridden.
    """
    if not isinstance(method, ReductionMethod):
        method = ReductionMethod.from_str(str(method))
    reduction = GpeCellReduction(method=method)

    # Common reduction parameters
    reduction.set_reduction_params({
            'Z_freq' :              25.,
            'Z_init_func' :         reduction.init_cell_steadystate,
            'Z_linearize_gating' :  False,
            'f_lambda':             100.0,
            'syn_scale_method' :    'Ai_syn_to_soma',
            'syn_position_method':  'root_distance_micron',
            })

    # Method-specific parameters
    if method == ReductionMethod.BushSejnowski:
        reduction.set_reduction_params({
            'gbar_init_method':     'area_weighted_average',
            'gbar_scale_method':    'surface_area_ratio',
            'passive_scale_method': 'surface_area_ratio',
            # 'gbar_scale_method':    'match_input_impedance_subtrees',
            # 'passive_scale_method': 'match_input_impedance_subtrees',
            ### Splitting cylinders based on L/lambda ##########################
            # 'split_criterion':      'eq_electrotonic_distance',
            # 'split_dX':             3.0,
            # 'lookahead_units':      'lambda',
            # 'lookahead_dX':         3.0,
            ### Splitting cylinders based on dL in micron ######################
            'split_criterion':      'micron_distance',
            'split_dX':             50.0,
        })
    
    elif method == ReductionMethod.Marasco:
        reduction.set_reduction_params({
            'gbar_scaling' :        'area',
            'set_ion_styles_func':  reduction.set_ion_styles,
            'post_tweak_funcs' :    [adjust_gbar_spontaneous] if tweak else [],
        })
    
    else:
        raise ValueError("Reduction method {} not supported".format(method))

    # Apply addition parameters (override)
    if reduction_params is not None:
        reduction.set_reduction_params(reduction_params)

    return reduction