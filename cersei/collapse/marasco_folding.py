"""
Morphology reduction using method described in Marasco & Migliore (2012)

@author Lucas Koelman
@date   5-12-2016
"""

# Python modules
import re
import math
PI = math.pi

# NEURON modules
import neuron
h = neuron.h
h.load_file("stdlib.hoc") # Load the standard library

# Own modules
import redutils, tree_edit as treeops
import common.electrotonic as electro
from common.nrnutil import ExtSecRef, seg_index
import cluster as clutools
from cluster import Cluster
import interpolation as interp
from fold_algorithm import FoldingAlgorithm, ReductionMethod
from marasco_merging import merge_seg_subtree

# logging of DEBUG/INFO/WARNING messages
import logging
logging.basicConfig(format='%(levelname)s:%(message)s @%(filename)s:%(lineno)s', level=logging.DEBUG)
logname = "marasco" # __name__
logger = logging.getLogger(logname) # create logger for this module

# Log to file
# fmtr = logging.Formatter('%(levelname)s:%(message)s @%(filename)s:%(lineno)s')
# fh = logging.FileHandler('reduce_marasco.log')
# fh.setFormatter(fmtr)
# logger.addHandler(fh)
# Log to stream
# ch = logging.StreamHandler(sys.stdout)
# ch.setFormatter(fmtr)
# logger.addHandler(ch)


################################################################################
# Interface implementations
################################################################################


class MarascoFolder(FoldingAlgorithm):
    """
    Marasco folding/collapsing algorithm.

    Original publication: Marasco, A., Limongiello, A. & Migliore, M. -
    Fast and accurate low-dimensional reduction of biophysically detailed 
    neuron models. Scientific Reports 2, (2012).
    """

    impl_algorithm = ReductionMethod.Marasco

    def __init__(self):
        """
        @note   FoldReduction class is responsible for maintaining
                bi-directional association
        """
        self.reduction = None


    def preprocess_reduction(self):
        """
        Preprocess cell for Marasco reduction. Execute once before all
        folding passes.
        """
        pass


    def fold(self, i_pass):
        """
        Do a folding pass.
        """

        self.prepare_folds_impl()
        fold_data = self.calc_folds_impl(i_pass)
        new_refs = self.make_folds_impl(fold_data)
        return new_refs


    def prepare_folds_impl(self):
        """
        Prepare next folding pass: assign topology information
        to each Section.

        (Implementation of interface declared in FoldingAlgorithm)
        """
        reduction = self.reduction

        root_ref = reduction._root_ref
        allsecrefs = reduction.all_sec_refs

        # Set properties used in calculation
        reduction.assign_new_sec_gids(root_ref, allsecrefs)
        redutils.subtree_assign_attributes(root_ref, allsecrefs, {'max_passes': 100})

        logger.info("\n###############################################################"
                    "\nAssigning topology & path properties ...\n")

        # Assign topology info (order, level, strahler number)
        clutools.assign_topology_attrs(root_ref, allsecrefs)
        reduction.fix_topology_below_roots()

        # Assign path properties
        f_lambda = self.reduction.get_reduction_param('f_lambda')
        for secref in allsecrefs:
            # Calculate path length, path resistance, electrotonic path length to each segment
            redutils.sec_path_props(secref, f_lambda, reduction.gleak_name)


    def calc_folds_impl(self, i_pass, Y_criterion='highest_level'):
        """
        Collapse branches at branch points identified by given criterion.
        """
        fold_pass = FoldingPass(i_pass)
        allsecrefs = self.reduction.all_sec_refs

        # Find collapsable branch points
        target_Y_secs = treeops.find_collapsable(allsecrefs, i_pass, Y_criterion)

        # Do collapse operation at each branch points
        fold_pass.clusters = calc_fold_equivalents(self.reduction, target_Y_secs, i_pass, allsecrefs)

        return fold_pass


    def make_folds_impl(self, fold_data):
        """
        Make equivalent Sections for branches that have been folded.

        @return     list(SectionRef) refs to new sections
        """

        # Mark Sections
        for secref in self.reduction.all_sec_refs:
            secref.is_substituted = False
            secref.is_deleted = False

        # Make new Sections
        eq_refs, newsecrefs = substitute_fold_equivalents(
                                self.reduction,
                                fold_data,
                                interp_prop='path_L',
                                interp_method='linear_neighbors',
                                gbar_scaling='area')

        return eq_refs


    def postprocess_reduction(self):
        """
        Post-process cell after Marasco reduction. Execute once after all
        folding passes.
        """
        reduction = self.reduction
        all_sec_refs = reduction.all_sec_refs

        # Tweaking
        tweak_funcs = reduction.get_reduction_param('post_tweak_funcs')
        for func in tweak_funcs:
            func(reduction)

        # Assign identifiers (for synapse placement etc.)
        reduction.assign_new_sec_gids(reduction._root_ref, all_sec_refs)

        # Assign topology info (order, level, strahler number)
        for fold_root in reduction._fold_root_refs:
            clutools.assign_topology_attrs(fold_root, all_sec_refs)


################################################################################
# Private functions
################################################################################


class FoldingPass(object):
    """
    Encapsulate data for one folding pass (all folds in one reduction step)

    Member data
    -----------

    i_pass : int
        number of the folding pass

    clusters : list(cluster.Cluster)
        data describing a collapsed fork

    """

    def __init__(self, i_pass): # TODO: add argument target_Y_secs
        self.i_pass = i_pass

    def do_folds(self):
        # loop that does all folds
        pass

    def fold_subtree(self, Ysec):
        # calculate one fold and return cluster (1st half of calc_fold_equivalents)
        pass

    def calc_fold_statistics(self, cluster):
        # calculate cluster statistics (2nd half of calc_fold_equivalents)
        pass

    def make_equivalent_secs(clusters):
        # create eqsec/eqref for each cluster
        pass

    def set_equivalent_properties():
        # takes map clusters -> equivalents?
        pass

    def set_passive_params():
        # first half of substitute_fold_equivalents
        pass

    def set_conductances():
        # second half of substitute_fold_equivalents
        pass


def calc_fold_equivalents(
        reduction,
        target_Y_secs,
        i_pass,
        allsecrefs
    ):
    """
    Do collapse operations: calculate equivalent Section properties for each collapse.

    @return         list of Cluster objects with properties of equivalent
                    Section for each set of collapsed branches.
    """

    # Get additional parameters
    glist = reduction.gbar_names
    gleak_name = reduction.gleak_name
    f_lambda = reduction.get_reduction_param('f_lambda')

    # Flag all sections as unvisited
    for secref in allsecrefs:

        if not hasattr(secref, 'absorbed'):
            secref.absorbed = [False] * secref.sec.nseg
            secref.visited = [False] * secref.sec.nseg
        
        secref.zip_labels = [None] * secref.sec.nseg

    # Create Clusters: collapse (zip) each Y section up to length of first branch point
    clusters = []
    for j_zip, par_ref in enumerate(target_Y_secs):
        par_sec = par_ref.sec
        child_secs = par_sec.children()

        # Function to determine which segment can be 'zipped'
        min_child_L = min(sec.L for sec in child_secs) # Section is unbranched cable
        eligfunc = lambda seg, jseg, ref: (ref.parent.same(par_sec)) and (seg.x*seg.sec.L <= min_child_L)
        
        # Name for equivalent zipped section
        # name_sanitized = par_sec.name().replace('[','').replace(']','').replace('.','_')
        name_sanitized = re.sub(r"[\[\]\.]", "", par_sec.name())
        alphabet_uppercase = [chr(i) for i in xrange(65,90+1)] # A-Z are ASCII 65-90
        zip_label = "zip{0}_{1}".format(alphabet_uppercase[i_pass], name_sanitized)
        zip_id = 1000*i_pass + j_zip

        # Function for processing zipped SectionRefs
        zipped_sec_gids = set()
        zipped_region_labels = set()
        def process_zipped_seg(seg, jseg, ref):
            # Tag segment with label of current zip operation
            ref.zip_labels[jseg] = zip_label
            # Save GIDs of original sections that are zipped/absorbed into equivalent section
            if ref.is_original:
                zipped_sec_gids.add(ref.gid)
                zipped_region_labels.add(ref.region_label)
            else:
                zipped_sec_gids.update(ref.zipped_sec_gids)
                zipped_region_labels.update(ref.zipped_region_labels)
        
        # Perform 'zip' operation
        far_bound_segs = [] # last (furthest) segments that are zipped
        eq_seq, eq_br = merge_seg_subtree(par_sec(1.0), allsecrefs, eligfunc, 
                            process_zipped_seg, far_bound_segs)

        logger.debug("Target Y-section for zipping: {0}".format(par_sec.name()))
        bounds_info = ["\n\tsegment {0} [{1}/{2}]".format(seg, seg_index(seg)+1, seg.sec.nseg) for seg in far_bound_segs]
        logger.debug("Zipping up to {0} boundary segments:{1}".format(len(far_bound_segs), "\n".join(bounds_info)))

        # Make Cluster object that represents collapsed segments
        cluster = Cluster(zip_label)
        cluster.eqL = eq_br.L_eq
        cluster.eqdiam = eq_br.diam_eq
        cluster.eq_area_sum = eq_br.L_eq * PI * eq_br.diam_eq
        cluster.eqri = eq_br.Ri_eq
        cluster.zipped_sec_gids = zipped_sec_gids
        cluster.zip_id = zip_id
        cluster.parent_seg = par_sec(1.0)
        cluster.bound_segs = far_bound_segs # Save boundaries (for substitution)

        # Calculate cluster statistics #########################################
        
        # Gather all cluster sections & segments
        clu_secs = [secref for secref in allsecrefs if (cluster.label in secref.zip_labels)]
        clu_segs = [seg for ref in clu_secs for jseg,seg in enumerate(ref.sec) if (
                        ref.zip_labels[jseg]==cluster.label)]

        # Calculate max/min path length
        clu_path_L = [ref.pathL_seg[j] for ref in clu_secs for j,seg in enumerate(ref.sec) if (
                        ref.zip_labels[j]==cluster.label)]
        cluster.orMaxPathL = max(clu_path_L)
        cluster.orMinPathL = min(clu_path_L)

        # Calculate min/max axial path resistance
        clu_path_ri = [ref.pathri_seg[j] for ref in clu_secs for j,seg in enumerate(ref.sec) if (
                        ref.zip_labels[j]==cluster.label)]
        cluster.orMaxpathri = max(clu_path_ri)
        cluster.orMinpathri = min(clu_path_ri)

        # Calculate area, capacitance, conductances
        cluster.or_area = sum(seg.area() for seg in clu_segs)
        cluster.or_cmtot = sum(seg.cm*seg.area() for seg in clu_segs)
        cluster.or_cm = cluster.or_cmtot / cluster.or_area
        cluster.or_gtot = dict((gname, 0.0) for gname in glist)
        for gname in glist:
            try:
                gval = getattr(seg, gname, 0.0)
            except NameError: # NEURON error if mechanism not inserted in section
                gval = 0.0
            cluster.or_gtot[gname] += sum(gval*seg.area() for seg in clu_segs)

        # Equivalent axial resistance
        clu_segs_Ra = [seg.sec.Ra for seg in clu_segs]
        if min(clu_segs_Ra) == max(clu_segs_Ra):
            cluster.eqRa = clu_segs_Ra[0]
        else:
            logger.warning("Sections have non-uniform Ra, calculating average "
                            "axial resistance per unit length, weighted by area")
            cluster.eqRa = PI*(cluster.eqdiam/2.)**2*cluster.eqri*100./cluster.eqL # eq. Ra^eq
        clusters.append(cluster)

        # Calculate electrotonic path length
        cluster.or_L_elec = sum(electro.seg_L_elec(seg, gleak_name, f_lambda) for seg in clu_segs)
        cluster.eq_lambda = electro.calc_lambda_AC(f_lambda, cluster.eqdiam, cluster.eqRa, cluster.or_cmtot/cluster.eq_area_sum)
        cluster.eq_L_elec = cluster.eqL/cluster.eq_lambda
        eq_min_nseg = electro.calc_min_nseg_hines(f_lambda, cluster.eqL, cluster.eqdiam, 
                                            cluster.eqRa, cluster.or_cmtot/cluster.eq_area_sum)

        # Debug
        print_attrs = ['eqL', 'eqdiam', 'or_area', 'eq_area_sum', 'or_L_elec', 'eq_L_elec']
        clu_info = ("- {0}: {1}".format(prop, getattr(cluster, prop)) for prop in print_attrs)
        logger.debug("Equivalent section for zipped Y-section has following properties:\n\t{0}".format(
                        "\n\t".join(clu_info)))
        logger.debug("Zip reduces L/lambda by {0:.2f} %; number of segments saved is {1} (Hines rule)\n".format(
                        cluster.eq_L_elec/cluster.or_L_elec, len(clu_segs)-eq_min_nseg))

    return clusters


def substitute_fold_equivalents(
        reduction,
        fold_pass,
        interp_prop='path_L', 
        interp_method='linear_neighbors', 
        gbar_scaling='area'
    ):
    """
    Substitute equivalent Section for each cluster into original cell.

    @see    docstring of function `reduce_bush_sejnowski.equivalent_sections()`

    @param  reduction       FoldReduction object

    @param  interp_prop     property used for calculation of path length (x of interpolated
                            x,y values), one of following:
                            
                            'path_L': path length (in micrometers)

                            'path_ri': axial path resistance (in Ohms)

                            'path_L_elec': electrotonic path length (L/lambda, dimensionless)

    @param  interp_method   how numerical values are interpolated, one of following:    
                            
                            'linear_neighbors':
                                linear interpolation of 'adjacent' segments in full model 
                                (i.e. next lower and higher electrotonic path length). 
                            
                            'linear_dist':
                                estimate linear distribution and interpolate it
                            
                            'left_neighbor', 'right_neighbor', 'nearest_neighbor':
                                extrapolation of 'neighoring' segments in terms of L/lambda
    """
    # List with equivalent section for each cluster
    eq_secs = []
    eq_refs = []

    # Create equivalent sections and passive electric structure
    for i, cluster in enumerate(fold_pass.clusters):

        # Create equivalent section
        if cluster.label in [sec.name() for sec in h.allsec()]:
            raise Exception('Section named {} already exists'.format(cluster.label))
        
        created = h("create %s" % cluster.label)
        if created != 1:
            raise Exception("Could not create section with name '{}'".format(cluster.label))
        
        eqsec = getattr(h, cluster.label)
        eqref = ExtSecRef(sec=eqsec)

        # Set passive properties
        eqsec.L = cluster.eqL
        eqsec.diam = cluster.eqdiam
        eqsec.Ra = cluster.eqRa

        # Save metadata
        for prop in ['zipped_sec_gids', 'zip_id', 'or_area']:
            setattr(eqref, prop, getattr(cluster, prop)) # copy some useful attributes
        eqref.is_original = False

        # Append to list of equivalent sections
        eq_secs.append(eqsec)
        eq_refs.append(eqref)

        # Connect to tree (need to trace path from soma to section)
        eqsec.connect(cluster.parent_seg, 0.0) # see help(sec.connect)

    
    gleak_name = reduction.gleak_name
    f_lambda = reduction.get_reduction_param('f_lambda')

    # Set active properties and finetune
    for i_clu, cluster in enumerate(fold_pass.clusters):
        logger.debug("Scaling properties of cluster %s ..." % fold_pass.clusters[i_clu].label)
        
        eqsec = eq_secs[i_clu]
        eqref = eq_refs[i_clu]

        # Scale passive electrical properties
        cluster.eq_area = sum(seg.area() for seg in eqsec) # should be same as cluster eqSurf
        area_ratio = cluster.or_area / cluster.eq_area
        logger.debug("Surface area ratio is %f" % area_ratio)

        # Scale Cm
        eq_cm = cluster.or_cmtot / cluster.eq_area # more accurate than cm * or_area/eq_area

        # Scale Rm
        or_gleak = cluster.or_gtot[gleak_name] / cluster.or_area
        eq_gleak = or_gleak * area_ratio # same as reducing Rm by area_new/area_old

        # Set number of segments based on rule of thumb electrotonic length
        eqsec.cm = eq_cm
        eqsec.nseg = electro.calc_min_nseg_hines(f_lambda, eqsec.L, eqsec.diam, eqsec.Ra, eq_cm)

        # Copy section mechanisms and properties
        absorbed_secs = redutils.find_secprops(reduction.orig_tree_props,
                                lambda sec: sec.gid in eqref.zipped_sec_gids)
        redutils.merge_sec_properties(absorbed_secs, eqsec, 
                        reduction.mechs_params_nogbar, check_uniform=True)

        # Save Cm and conductances for each section for reconstruction
        cluster.nseg = eqsec.nseg # save for reconstruction
        cluster.eq_gbar = dict((
            (gname, [float('NaN')]*cluster.nseg) for gname in reduction.active_gbar_names))
        cluster.eq_cm = [float('NaN')]*cluster.nseg

        # Set Cm and gleak (Rm) for each segment
        if gbar_scaling is not None:
            for j, seg in enumerate(eqsec):
                setattr(seg, 'cm', eq_cm)
                setattr(seg, gleak_name, eq_gleak)
                cluster.eq_cm[j] = eq_cm
                cluster.eq_gbar[gleak_name][j] = eq_gleak

        # Calculate path lengths in equivalent section
        redutils.sec_path_props(eqref, f_lambda, gleak_name)
        
        if interp_prop == 'path_L':
            seg_prop = 'pathL_seg'
        elif interp_prop == 'path_ri':
            seg_prop = 'pathri_seg'
        elif interp_prop == 'path_L_elec':
            seg_prop = 'pathL_elec'
        else:
            raise ValueError("Unknown path property '{}'".format(interp_prop))

        # Find conductances at same path length (to each segment midpoint) in original cell
        for j_seg, seg in enumerate(eqsec):
            
            # Get adjacent segments along interpolation path
            path_L = getattr(eqref, seg_prop)[j_seg]
            path_secs = reduction.get_interpolation_path_secs(eqref)
            bound_segs, bound_L = interp.find_adj_path_segs(interp_prop, path_L, path_secs)
            
            # DEBUG STATEMENTS:
            # bounds_info = "\n".join(("\t- bounds {0} - {1}".format(a, b) for a,b in bound_segs))
            # logger.debug("Found boundary segments at path length "
            #              "x={0:.3f}:\n{1}".format(path_L, bounds_info))

            # INTERPOLATE: Set conductances by interpolating neighbors
            for gname in reduction.active_gbar_names:
                
                if interp_method == 'linear_neighbors':
                    gval = interp.interp_gbar_linear_neighbors(path_L, gname, bound_segs, bound_L)
                
                else:
                    match_method = re.search(r'^[a-z]+', interp_method)
                    method = match_method.group() # should be nearest, left, or right
                    gval = interp.interp_gbar_pick_neighbor(path_L, gname, 
                                        bound_segs[0], bound_L[0], method)
                
                seg.__setattr__(gname, gval)
                cluster.eq_gbar[gname][j_seg] = gval

        # Re-scale gbar distribution to yield same total gbar (sum(gbar*area))
        if gbar_scaling is not None:
            for gname in reduction.active_gbar_names:
                
                eq_gtot = sum(getattr(seg, gname)*seg.area() for seg in eqsec)
                if eq_gtot <= 0.:
                    eq_gtot = 1.
                
                or_gtot = cluster.or_gtot[gname]
                
                for j_seg, seg in enumerate(eqsec):
                    
                    if gbar_scaling == 'area':
                        # conserves ratio in each segment but not total original conductance
                        scale = cluster.or_area/cluster.eq_area
                    
                    elif gbar_scaling == 'gbar_integral':
                        # does not conserve ratio but conserves gtot_or since: sum(g_i*area_i * or_area/eq_area) = or_area/eq_area * sum(gi*area_i) ~= or_area/eq_area * g_avg*eq_area = or_area*g_avg
                        scale = or_gtot/eq_gtot
                    
                    else:
                        raise Exception("Unknown gbar scaling method'{}'.".format(gbar_scaling))
                    
                    # Set gbar
                    gval = getattr(seg, gname) * scale
                    seg.__setattr__(gname, gval)
                    
                    cluster.eq_gbar[gname][j_seg] = gval # save for reconstruction

        # Check gbar calculation
        # for gname in active_glist:
        #   gtot_or = cluster.or_gtot[gname]
        #   gtot_eq_scaled = sum(getattr(seg, gname)*seg.area() for seg in eqsec)
        #   logger.debug("conductance %s : gtot_or = %.3f ; gtot_eq = %.3f",
        #                   gname, gtot_or, gtot_eq_scaled)

        # Debugging info:
        logger.debug("Created equivalent Section '%s' with \n\tL\tdiam\tcm\tRa\tnseg"
                     "\n\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n", fold_pass.clusters[i_clu].label, 
                     eqsec.L, eqsec.diam, eqsec.cm, eqsec.Ra, eqsec.nseg)
    
    # Substitute equivalent section into tree
    logger.info("\n###############################################################"
                "\nSubstituting equivalent sections...\n")

    orsecrefs = reduction.all_sec_refs
    for i_clu, cluster in enumerate(fold_pass.clusters):
        eqsec = eq_secs[i_clu]
        # Disconnect substituted segments and attach segment after Y boundary
        # Can only do this now since all paths need to be walkable before this
        logger.debug("Substituting zipped section {0}".format(eqsec))
        treeops.sub_equivalent_Y_sec(eqsec, 
                    cluster.parent_seg, cluster.bound_segs, 
                    orsecrefs, reduction.mechs_gbars_dict, delete_substituted=True)
        logger.debug("Substitution complete.\n")

    # build new list of valid SectionRef
    newsecrefs = [ref for ref in orsecrefs if not (ref.is_substituted or ref.is_deleted)]
    newsecrefs.extend(eq_refs)
    return eq_refs, newsecrefs