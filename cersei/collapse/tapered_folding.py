"""
Morphology reduction by constructing equivalent cylinders with variable
diameter.

@author Lucas Koelman
@date   19-01-2017
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
from common.treeutils import subtree_topology
from common.nrnutil import ExtSecRef, getsecref
from common.stdutil import isclose

import redutils
import tree_edit as treeedit
import interpolation as interp
from fold_algorithm import FoldingAlgorithm
import tapered_merging as taper

import logging, common.logutils as logutils
logger = logutils.getBasicLogger(
                    name='marasco', level=logging.DEBUG,
                    format="%(levelname)s@%(filename)s:%(lineno)s  %(message)s")


class TaperedFolder(FoldingAlgorithm):
    """
    Tapered folding procedure.

    Constructs equivalent cylinders with variable diameter by marching along
    parallel branches concurrently and, at each step, calculating an equivalent
    cylinder that is connected sequentially to the last.
    """

    def __init__(self, algorithm):
        """
        @note   FoldReduction class is responsible for maintaining
                bi-directional association
        """
        self.reduction = None
        self.impl_algorithm = algorithm


    def preprocess_reduction(self):
        """
        Preprocess cell for reduction.
        """
        # NOTE: all required peprocessing steps are done in FoldReduction.preprocess_cell()
        pass


    def fold_one_pass(self, i_pass, Y_criterion="exact_level"):
        """
        Collapse branches at branch points identified by given criterion.
        """
        # Find collapsable branch points
        target_Y_secs = treeedit.find_collapsable(
                                self.reduction.all_sec_refs, 
                                i_pass, Y_criterion, level=2)

        fold_pass = AdvancingFrontCollapse(
                        self.reduction, 
                        fold_roots=target_Y_secs,
                        interp_prop='path_L',
                        interp_method='linear_neighbors',
                        gbar_scaling='area')
        
        new_refs = fold_pass.do_folds() # do collapse operation at each branch points

        return new_refs


    def postprocess_reduction(self):
        """
        Postprocess cell after reduction procedure. Executed once.
        """
        pass


################################################################################
# Private functions
################################################################################


class AdvancingFrontCollapse(object):
    """
    Encapsulate data for collapse operation.
    """

    def __init__(
            self,
            reduction,
            fold_roots,
            interp_prop,
            interp_method,
            gbar_scaling
        ):
        self.reduction = reduction
        self.fold_roots = fold_roots
        self.interp_prop = interp_prop
        self.interp_method = interp_method
        self.gbar_scaling = gbar_scaling

        self.f_lambda = self.reduction.get_reduction_param('f_lambda')
        self.gleak_name = self.reduction.gleak_name


    def collapse(self):
        """
        Calculate folds, make equivalent sections, and insert them into tree.

        @return     list(SectionRef)
                    list of equivalent sections for folds
        """

        # Flag all sections as unvisited
        for secref in self.reduction.all_sec_refs:
            secref.is_original = True
            secref.is_substituted = False
            secref.is_deleted = False
            if not hasattr(secref, 'absorbed'):
                secref.absorbed = [False] * secref.sec.nseg
                secref.visited = [False] * secref.sec.nseg

        logger.info("\n###############################################################"
                    "\nCollapsing subtrees ..."
                    "\n###############################################################")

        # Calculate equivalent cylinders for each folding root
        subtree_eq_cyls = []
        for collapse_index, par_ref in enumerate(self.fold_roots):

            seq_cyls = self.collapse_subtree(par_ref, collapse_index)
            subtree_eq_cyls.append(seq_cyls)

        # substitute equivalents
        subtree_sec_refs = self.make_equivalent_secs(subtree_eq_cyls)

        logger.info("\n###############################################################"
                    "\nDetermining active electrical properties ..."
                    "\n###############################################################")

        # Set their properties
        self.set_specific_elec_params(subtree_sec_refs)

        if self.interp_method is not None:
            logger.debug("Using path interpolation for setting conductances.")
            self.set_conductances_interp(subtree_sec_refs)
            self.scale_conductances(subtree_sec_refs)
        else:
            logger.debug("NOT using path interpolation for setting conductances")

        logger.info("\n###############################################################"
                    "\nDelete substituted sections ..."
                    "\n###############################################################")

        self.disconnect_substituted_secs(delete=True)

        new_sec_refs = [ref for subtree in subtree_sec_refs for ref in subtree]
        return new_sec_refs


    def int_to_alphabet(i):
        """
        Convert integer to alphabetic character.
        """
        # NOTE: A-Z are ASCII characters 65-90
        num_char = 90-65+1
        q, r = divmod(i, num_char)
        letter = chr(65+r)
        return q*letter


    def collapse_subtree(self, par_ref, i_collapse):
        """
        Calculate one subtree and return equivalent cylinders

        @param  par_ref : SectionRef
                folding root (fork) whose children should be folded

        @param  i_collapse : int
                index (sequence number_ of collapse operation

        @return cluster: Cluster
                cluster object describing properties of equivalent section
        """
        par_sec = par_ref.sec
        allsecrefs = self.reduction.all_sec_refs

        logger.debug("Topology at folding root {0}:\n".format(par_sec.name()) + 
                        subtree_topology(par_sec, max_depth=2))
        
        # Collapse operation: calculate cylinders
        seq_cyls = taper.merge_cylinders_subtree(
                            par_sec(1.0), allsecrefs,
                            self.split_criterion,
                            self.distance_func)

        # Label this collapse operation
        zip_id = 1000 + i_collapse # usable as gid

        # Save metadata for each equivant cylinder
        for i_cyl, cyl in enumerate(seq_cyls):
            cyl.zip_id = zip_id
            cyl.label = "zip{0}{1}".format(self.int_to_alphabet(i_collapse), i_cyl)
            if i_cyl == 0:
                cyl.parent_seg = par_sec(1.0)
            else:
                cyl.parent_seg = None

        # Post-process subtree
        subtree_seclist = h.SectionList()
        par_sec.push(); subtree_seclist.subtree(); h.pop_section()
        for child_sec in subtree_seclist: # iterates CAS
            # Make sure every Section has been visited and fully absorbed (until 1-end)
            child_ref = getsecref(child_sec, allsecrefs)
            assert child_ref.visited and child_ref.absorbed

        return seq_cyls


    def make_equivalent_secs(self, subtree_eq_cyls):
        """
        Create equivalent section for each cylinder (only geometry and connections).

        @param  subtree_eq_cyls : list(list(Cylinder))
                For each collapsed subtree: list of equivalent sequential Cylinders
        """

        subtree_sec_refs = []
        for eq_cyls in subtree_eq_cyls:
            
            sequential_refs = []
            for cyl in eq_cyls:

                if h.section_exists(cyl.label):
                    raise Exception('Section named {} already exists'.format(cyl.label))
                
                created = h("create %s" % cyl.label)
                if created != 1:
                    raise Exception(
                            "Could not create section with name '{}'".format(cyl.label))
                
                eqsec = getattr(h, cyl.label)
                eqref = ExtSecRef(sec=eqsec)
                eqref.label = cyl.label

                # Set passive properties
                eqsec.L = cyl.L
                eqsec.diam = cyl.diam
                eqsec.Ra = cyl.Ra
                assert isclose(cyl.area, sum(seg.area() for seg in eqsec), rel_tol=1e-9)

                # Connect to tree (need to trace path from soma to section)
                if cyl.parent_seg is not None:
                    eqsec.connect(cyl.parent_seg, 0.0)
                else:
                    eqsec.connect(sequential_refs[-1].sec(1.0), 0.0)

                # Save info about merged properties
                eqref.orig_props = cyl.orig_props
                eqref.is_original = False
                sequential_refs.append(eqref)

            subtree_sec_refs.append(sequential_refs)

        return subtree_sec_refs


    def set_specific_elec_params(self, subtree_sec_refs):
        """
        Set passive parameters and insert mechanisms into equivalent sections.

        @param  subtree_sec_refs : list(list(Cylinder))
                For each collapsed subtree: list of sequential equivalent Sections

        WAYS TO SET GBAR
        ----------------

        TODO: let user choose to initialize with g_wavg or g_interp, make
              argument gbar_init_method and gbar_scale_method

        - set them to the area-weighted average gbar of all cylinders
          that have been merged into the equivalent sequential section, with
          no further scaling

            + i.e. g_i = g_wavg = sum(g_j*S_j) / sum(S_j) where j are absorbed cylinders.

            + => Gtot = sum(g_wavg * Snew_cyl)
            
            + this is likely to preserve the gradient/spatial profile since
              gbar is the weighted average at each distance
            
            + this does NOT compensate for area reduction, so the total
              integrated Gbar in an equivalent section will not be the same
              as that in the original cylinders
        
        - initialize to area-weighted gbar, then multiply the gbar in each
          equivalent section with the ration of original area over new area

            + i.e. g_i = g_wavg * Sorig_cyl / Snew_cyl

            + => Gtot = sum(g_wavg * Snew_cyl * Sorig_cyl / Snew_cyl)
                      = sum(g_wavg * Sorig_cyl)
                      ~= sum(Gorig_cyl)
                      ~= Gorig_subtree

            + since each cylinder has its own scaling factor, this will likely
              NOT preserve the gradient/spatial profile

            + this _approximately_ preserves Gbar in each cylinder and the
              entire subtree

        - initialize to area-weighted gbar, then multiply the gbar in each
          equivalent section with the ratio of integrated Gbar over that
          section vs over absorbed cylinders

            + i.e. g_i = g_wavg * Gorig_cyl / Gnew_cyl

            + => Gtot = sum(g_wavg * Snew_cyl * Gorig_cyl / Gnew_cyl)
                      = sum(Gorig_cyl)
                      = Gorig_subtree

            + since each cylinder has its own scaling factor, this will likely
              NOT preserve the gradient/spatial profile

            + this _exactly_ preserves total Gbar in each cylinder AND over,
              entire subtree so this has good area compensation

        - initialize to area-weighted gbar, then multiply the gbar in each
          equivalent section with the ratio of integrated Gbar over entire
          subtree

            + i.e. g_i = g_wavg * Gorig_subtree / Gnew_subtree

            + => Gtot = sum(g_wavg * Snew_cyl * Gorig_subtree / Gnew_subtree)
                      = Gorig_subtree / Gnew_subtree * sum(g_wavg * Snew_cyl)
                      = Gorig_subtree

            + this is likely to preserve the gradient/spatial profile since all
              the g_wavg are multiplied by constant factor

            + this preserves total Gbar over entire subtree, but NOT in each
              cylinder individually, which is a form of area compensation
        
        """
        
        # Set passive electrical properties
        new_sec_refs = [ref for subtree in subtree_sec_refs for ref in subtree]
        for eqref in new_sec_refs:
            
            eqsec = eqref.sec
            orig = eqref.orig_props
            assert eqsec.nseg == 1

            # Scaling of area-specific passive electrical properties
            eq_area = sum(seg.area() for seg in eqsec)
            # TODO: the scaling method here has to match that of conductances
            #       in order to preserve proportionality in differential equations
            eq_cm = orig.cmtot / eq_area # conserve cm _exactly_
            eq_gleak = orig.gtot[self.gleak_name] / eq_area # conserve gleak _exactly_
            logger.debug("Surface area ratio is %f" % (orig.area / eq_area))
            eqsec.cm = eq_cm
            setattr(eqsec, self.gleak_name, eq_gleak)

            # Copy section mechanisms and properties
            absorbed_secs = redutils.find_secprops(
                                    self.reduction.orig_tree_props,
                                    lambda sec: sec.gid in orig.merged_sec_gids)
            
            redutils.merge_sec_properties(
                            absorbed_secs, eqsec, 
                            self.reduction.mechs_params_nogbar, 
                            check_uniform=True)


    def init_passive_wavg(self, subtree_sec_refs):
        """
        TODO: see previous note, scaling method has to match that of conductances
        """
        pass


    def init_conductances_wavg(self, subtree_sec_refs):
        """
        TODO: check what Bush & Marasco do, plot conductance profiles and 
              responses for four scaling methods.
        """

        new_sec_refs = [ref for subtree in subtree_sec_refs for ref in subtree]
        for eqref in new_sec_refs:

            eqsec = eqref.sec
            orig = eqref.orig_props

            for gname in self.reduction.active_gbar_names:
                eq_gbar = orig.gtot[gname] / orig.area
                setattr(eqsec, gname, eq_gbar)


    def init_conductances_interp(self, subtree_sec_refs):
        """
        Set channel conductances by interpolating conductance values along
        a path of sections in the original cell model.

        This does not take into account the effect of a different surface area.

        @param  subtree_sec_refs : list(list(Cylinder))
                For each collapsed subtree: list of sequential equivalent Sections

        @post   for each conductance in reduction.active_gbar_names, all segments
                in the cluster are assigned a value true interpolation.
        """
        # Names of path-integrated properties stored on SectionRef
        path_prop_names = {
            'path_L': 'pathL_seg',
            'path_ri': 'pathri_seg',
            'path_L_elec': 'pathL_elec',
        }
        seg_prop = path_prop_names[self.interp_prop]

        new_sec_refs = [ref for subtree in subtree_sec_refs for ref in subtree]
        for eqref in new_sec_refs:
            
            logger.debug("Scaling properties of cylinder %s ..." % eqref.label)
            eqsec = eqref.sec

            # Calculate path-integrated properties
            redutils.sec_path_props(eqref, self.f_lambda, self.gleak_name)

            # Find path of original sections running through one of folded branches
            path_secs = self.reduction.get_interpolation_path_secs(eqref)

            # Find conductances at same path length (to each segment midpoint) in original cell
            for j_seg, seg in enumerate(eqsec):
                
                # Get adjacent segments along interpolation path
                path_L = getattr(eqref, seg_prop)[j_seg]
                bound_segs, bound_L = interp.find_adj_path_segs(self.interp_prop, path_L, path_secs)

                # INTERPOLATE: Set conductances by interpolating neighbors
                for gname in self.reduction.active_gbar_names:
                    if not hasattr(eqsec, gname):
                        continue
                    
                    # get interpolation function and use it to compute gbar
                    if self.interp_method == 'linear_neighbors':
                        gval = interp.interp_gbar_linear_neighbors(path_L, gname, bound_segs, bound_L)
                    else:
                        match_method = re.search(r'^[a-z]+', self.interp_method)
                        method = match_method.group() # should be nearest, left, or right
                        gval = interp.interp_gbar_pick_neighbor(path_L, gname, 
                                            bound_segs[0], bound_L[0], method)
                    
                    setattr(seg, gname, gval)


    def scale_conductances(self, subtree_sec_refs):
        """
        After interpolation of specific conductance values, scale values
        so that total non-specific conductance integrated over cluster is the same.

        @param  clusters_refs: dict(Cluster: SectionRef)
                clusters representing collapsed branches and their equivalent section

        @post   
        """
        new_sec_refs = [ref for subtree in subtree_sec_refs for ref in subtree]
        
        for eqref in new_sec_refs:
            eqsec = eqref.sec
            orig = eqref.orig_props

            # Re-scale gbar distribution to yield same total gbar (sum(gbar*area))
            for gname in self.reduction.active_gbar_names:
                if not hasattr(eqsec, gname):
                    continue
                
                # original and current total conductance
                gtot_orig = orig.gtot[gname]
                gtot_interp = sum(getattr(seg, gname, 0.0)*seg.area() for seg in eqsec)
                if gtot_interp <= 0.:
                    gtot_interp = 1.
                
                # TODO: check this calculation
                for seg in eqsec:
                    
                    if self.gbar_scaling == 'area':
                        # conserves ratio in each segment but not total original conductance
                        scale = cluster.or_area/cluster.eq_area
                    
                    elif self.gbar_scaling == 'gbar_integral':
                        # does not conserve ratio but conserves gtot_or since: sum(g_i*area_i * or_area/eq_area) = or_area/eq_area * sum(gi*area_i) ~= or_area/eq_area * g_avg*eq_area = or_area*g_avg
                        scale = or_gtot/eq_gtot
                    
                    else:
                        raise Exception("Unknown gbar scaling method'{}'.".format(self.gbar_scaling))
                    
                    # Set gbar
                    gval = getattr(seg, gname) * scale
                    setattr(seg, gname, gval)

            # Debugging info:
            logger.anal("Created equivalent Section '%s' with \n\tL\tdiam\tcm\tRa\tnseg"
                         "\n\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n", cluster.label, 
                         eqsec.L, eqsec.diam, eqsec.cm, eqsec.Ra, eqsec.nseg)



    def disconnect_substituted_secs(self, delete=True):
        """
        Disconnect substituted sections and delete them if requested.
        """
        def should_delete(sec_ref):
            """ Check if Section was substituted and should be deleted. """
            return sec_ref.is_original and sec_ref.absorbed

        for root_ref in self.fold_roots:
            treeedit.disconnect_subtree(
                        root_ref,
                        self.reduction.all_sec_refs,
                        should_disconnect=should_delete,
                        delete=True)