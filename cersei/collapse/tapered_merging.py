"""
Recursive merging procedures for cylindrical compartments

@author     Lucas Koelman
@date       17-01-2017
"""

# Python modules
import math
PI = math.pi

# Our modules
from common.treeutils import next_segs
from common.nrnutil import seg_xmin, seg_xmax
from common.electrotonic import seg_lambda
from numpy import interp

################################################################################
# Rewrite from formulas
################################################################################

class Cylinder(object):
    """
    Cylindrical compartment, the fundamental element of the cable
    representation of a neurite.
    """

    def __init__(self, nrn_seg=None, use_seg_x=None, diam=None, L=None, Ra=None):
        """
        Create new cylindrical compartment.

        @param  nrn_seg : nrn.Segment
                A NEURON segment to copy properties from. Provide either this
                or explicit properties.

        @param  use_seg_x : (optional) str
                If not None: split given segment depending on value:
                "x_min" use nrn_seg.x as its left boundary
                "x_max" use nrn_seg.x as its right boundary

        @param  diam : float
                Cylinder diameter in units of micron

        @param  L : float
                Cylinder length in units of micron

        @param  Ra : float
                Cytoplasmic resistivity in units of [Ohm * cm]
        """
        if nrn_seg is not None:

            self.diam = nrn_seg.diam
            self.Ra = nrn_seg.sec.Ra
            
            if use_seg_x is None:
                self.L = nrn_seg.sec.L / nrn_seg.sec.nseg
            elif use_seg_x == "x_min":
                self.L = nrn_seg.sec.L * (seg_xmax(nrn_seg) - nrn_seg.x)
            elif use_seg_x == "x_max":
                self.L = nrn_seg.sec.L * (nrn_seg.x - seg_xmin(nrn_seg))
            else:
                raise ValueError("Argument 'use_seg_x': invalid value {}".format(use_seg_x))
        else:
            self.diam = diam
            self.L = L
            self.Ra= Ra

            if any([nrn_seg, use_seg_x]):
                raise ValueError("Provide either a NEURON segment or cylinder properties "
                                 "but not both.")


    @property
    def area(self):
        return PI * self.diam * self.L


    @property
    def Ri(self):
        """
        Absolute axial resistance in units of Ohm
        """
        # units = ([Ohm*cm] * [um]) / ([um]^2)
        # units = ([Ohm] * [m]*1e-2 * [m]*1e-6) / ([m]*1e-6 * [m]*1e-6)
        # units = [Ohm] * [m] * 1e4
        # units = [Ohm] * [cm] * 1e2
        unit_factor = 1e2
        return (4.0 * self.Ra * self.L) / (PI * self.diam**2 * unit_factor)
    

def merge_cylinders_sequential(cyls):
    """
    Merge sequential cylinders
    
    @param      cyls: list(Cylinder)
                Cilinders to merge sequentially

    @return     Cylinder
    """
    # Trivial case: only a single cylinder
    if len(cyls) == 1:
        return cyls[0]

    # Intermediate calculations
    Ri_seq = sum((cyl.Ri for cyl in cyls))

    # Cylinder geometrical & electrical properties
    L_seq = sum((cyl.L for cyl in cyls))
    Ra_seq = sum((cyl.Ra for cyl in cyls)) / len(cyls)
    diam_seq = math.sqrt(Ra_seq*L_seq*4./PI/Ri_seq/100.) # ensures that Ri_seq will be the total absolute axial resistance of the equivalent cylinder

    return Cylinder(diam=diam_seq, L=L_seq, Ra=Ra_seq)


def merge_cylinders_parallel(cyls):
    """
    Merge cylinders that are connected in parallel, i.e. as siblings of the
    same parant cylinder at a branch point.

    @param      cyls: list(Cylinder)
                Cilinders to merge

    @return     Cylinder
    """
    # Trivial case: only a single cylinder
    if len(cyls) == 1:
        return cyls[0]

    # Cylinder geometrical & electrical properties
    L_br = sum((cyl.area*cyl.L for cyl in cyls)) / sum((cyl.area for cyl in cyls))
    Ra_br = sum((cyl.Ra for cyl in cyls)) / len(cyls)
    diam_br = math.sqrt(sum((cyl.diam**2 for cyl in cyls))) # ensures that absolute axial resistance Ri (also per unit length, Ri/L) is preserved

    return Cylinder(diam=diam_br, L=L_br, Ra=Ra_br)


def merge_until_distance(start_seg, stop_dist, distance_func):
    """
    Keep merging cylindrical compartments in subtree starting at start_seg until
    distance_func(end_seg) is equal to the desired stopping distance.

    @param  start_seg : nrn.Segment
            Segment to star merging procedure with x-value equal to 
            the last splitting point.
    
    @param  distance_func : callable(nrn.Segment) -> float
            Function that measures distance of a NEURON segment with an
            associated x-value, e.g. from the root of the tree

    @param  stop_dist: float
            The distance at which merging should be stopped, measured using
            distance_func

    @return tuple(Cylinder, list(nrn.Segment))
            Equivalent cylinder until stopping distance, and the list of
            stopping points
    """
    # Keep absorbing segments until the 1-end boundary is larger than distance
    start_x_max = start_seg.sec(seg_xmax(start_seg, inside=True))
    X_beg = distance_func(start_seg)
    X_end = distance_func(start_x_max)
    if stop_dist < X_beg:
        raise ValueError("Start segment is already farther than stopping distance")

    # Scenario 1: far boundary of starting segment is farther that stop_dist
    if stop_dist < X_end:
        # Interpolate distance values at start and end of segment
        x_interp = interp(
                    [stop_dist], 
                    [X_beg, X_end],
                    [seg_xmin(start_seg), seg_xmax(start_seg, inside=True)])
        # Truncate segment at stopping dist
        trunc_cyl = Cylinder(diam=start_seg.diam, Ra=start_seg.sec.Ra,
                             L=start_seg.sec.L*(x_interp-start_seg.x))
        term_seg = start_seg.sec(x_interp)
        return trunc_cyl, [term_seg]
    
    # Scenario 2: stopping distance is beyond far boundary of segment
    start_cyl = Cylinder(nrn_seg=start_seg, use_seg_x="x_min") # cylinder representing the starting segment

    # Scenario 2.1: no child segments
    child_segments = next_segs(start_seg, x_loc="min")
    if not any(child_segments):
        return start_cyl, []

    # Scenario 2.2: child segment can be merged
    # Get child segment(s), absorb their cylinders into starting cylinder
    parallel_cyls = []
    parallel_x_stop = []
    for child_seg in child_segments:
        child_eq_cyl, child_x_term = merge_until_distance(child_seg, stop_dist, distance_func)
        parallel_cyls.append(child_eq_cyl)
        parallel_x_stop.extend(child_x_term)

    # Merge parallel child cylinders (equivalents)
    next_seq_cyl = merge_cylinders_parallel(parallel_cyls)
    
    # Merge children equivalent sequentially into own cylinder
    sequential_cyls = [start_cyl, next_seq_cyl]
    subtree_eq_cyl = merge_cylinders_sequential(sequential_cyls)

    return subtree_eq_cyl, parallel_x_stop


def next_splitpoints_seg_dlambda(current_seg, lambda_start=None, fraction=0.1):
    """
    Ascend tree and return first segment along each path where lambda
    differs by more than given fraction from starting value.

    @param  start_segs: list(nrn.Segment)
            segments with x-values where last split occurred
    """
    # Merge up to first segment where lambda differs more than 10% from start
    # i.e. in each sibling: ascend until you find segment with d_lambda > 10 or branch point
    if lambda_start is None:
        lambda_start = seg_lambda(current_seg, None, 100.0)

    # Check current node
    lambda_current = seg_lambda(current_seg, None, 100.0)
    if not (1.0-fraction <= lambda_current/lambda_start <= 1.0+fraction):
        # lambda changed more than theshold: split before segment
        return [current_seg]

    # Check child nodes
    child_segments = next_segs(current_seg, x_loc="min")
    child_splitpoints = []
    for child_seg in child_segments:
        child_splitpoints.extend(next_splitpoints_seg_dlambda(
                                    child_seg,
                                    lambda_start=lambda_start,
                                    fraction=fraction))

    return child_splitpoints


def next_splitpoints(start_seg, **kwargs):
    """
    Return next cable splitting points encountered when ascending
    tree starting at given segment

    @param  split_criterion : str
    
            How cylinder boundaries should be determined, one of following:
            
            - "segment_dlambda"
                - merge until segment length constant has changed by given
                  fraction of starting value
            
            - (NOT IMPLEMENTED) "segment_ddiam"
                - merge until diam has changed by given fraction of starting value

            - (NOT IMPLEMENTED) "equivalent_lambda"
                - merge approximately until the point where the equivalent cylinder
                  has a length of 10% its electrotonic length

            - (NOT IMPLEMENTED) "path_dL"
                - merge until fixed increment in path length from root is reached

            - (NOT IMPLEMENTED) "path_dlambda"
                - merge until fixed increment in electrotonc path length is reached

    """
    split_criterion = kwargs.pop["split_criterion"]
    
    if split_criterion == "segment_dlambda":
        return next_splitpoints_seg_dlambda(start_seg, **kwargs)

    else:
        raise NotImplementedError("Splitting criterion {} not implemented".format(split_criterion))


def merge_cylinders_subtree(
        node,
        allsecrefs,
        split_criterion,
        distance_func
    ):
    """
    Merge all cylinders in subtree in one shot.

    NOTE: the reason that we write a wrapper around merge_until_distance() is
    that in the top-level loop, we don't want to do sequential merging since
    this does not preserve diameter tapering.

    @param  node : nrn.Segment
            Segment where children will be merged.
    
    @param  distance_func : callable(nrn.Segment) -> float
            Function that measures distance of a NEURON segment with an
            associated x-value, e.g. from the root of the tree

    @param  split_criterion : str
            See same argument @ next_splitpoints()

    TODO:   for each sequential cylinder: save original total area,
            and integrated conductances

    TODO:   make tapered_folding module based on marasco_folding

    TODO:   make variant where you step with a small step size, but merge
            sequential equivalent cylinders until you get approx L/lambda = 0.1.
            This can be done at the end as a postprocessing step.

            Or you could make a heuristic where you first calculate lambda in
            the next increment equivalent cylinder (step by small dL) and then
            step by a true dL = 0.1*lambda_est, assuming that area ratio
            will be approximately constant in that step. You can do this in 
            adaptive step fashion: first make small step to calculate local
            lambda, then add a step that would dL ~= 0.1*lambda, but then you
            may have to backtrack.

            Or take Stratford idea: walk 0.1*orig_lambda along each parallel branch,
            and make sure that the equivalent cylinder also has L=0.1*new_lambda.
            Problem is that the new lambda depends on new cm, which is scaled
            depending on new L (recursive dependence). When you solve this equation
            `L=0.1*new_lambda` for L, you obtain an expression for L which
            should be `L = (0.1*d_eq)^2 / (f*cm*S_old*Ra)` where S_old is the total
            surface of merge sections. However this is exclusive with the Bush
            & Sejnowski principle because that sets dimensions to determine axial
            resistance per unit length. For this approach you can make new functions
            merge_cylinder_xxx() that preserve L/lambda rather than ra.
    """

    # Initial cable splitting points
    this_x_split = [sec(0.0) for sec in node.sec.children()]
    sequential_cyls = []

    # Continue as long as there is any branch left where we haven't marched until terminal segment
    while len(this_x_split) > 0:
    
        # Look ahead: find next cable splitting points on each branch
        candidate_x_split = []
        for seg_x in this_x_split:
            branch_x_split = next_splitpoints(this_x_split, split_criterion=split_criterion)
            candidate_x_split.extend(branch_x_split)
        
        candidate_dists = [distance_func(x_seg) for x_seg in candidate_x_split]
        dist_x_split = min(candidate_dists)

        # On each sibling branch: march up tree until split distance,
        # and return the equivalent cylinder until that point, as well as 
        # the new splitting points (at end of merge operation)
        parallel_x_stop = []
        parallel_cyls = []
        for seg_x in this_x_split:
            branch_eq_cyl, branch_x_stop = merge_until_distance(
                                                seg_x, dist_x_split, distance_func)
            parallel_cyls.append(branch_eq_cyl) # always one cylinder
            parallel_x_stop.extend(branch_x_stop) # split points only if end not reached

        # Merge the equivalent cylinders in parallel
        seq_cyl = merge_cylinders_parallel(parallel_cyls)
        sequential_cyls.append(seq_cyl)
        
        # Update cable splitting points (new starting points)
        this_x_split = parallel_x_stop

    # TODO: postprocess by merging sequential cylinders?
    return sequential_cyls

