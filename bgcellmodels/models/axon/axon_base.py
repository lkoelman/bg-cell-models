"""
Base class for axon builders.
"""

from __future__ import division # float division for int literals, like Hoc
import math
import logging

import numpy as np
import neuron

from bgcellmodels.morphology import morph_3d

h = neuron.h
PI = math.pi

def normvec(a):
    """
    Normalized vector pointing from a to b.
    """
    return a / np.sqrt(np.dot(a, a))


def veclen(a):
    """
    Vector length of a (Euclidean norm).
    """
    return np.sqrt(np.dot(a, a))


class AxonBuilder(object):
    """
    Base class for axon model

    The attributes that must be set are described below. You can either set
    them in a method or define them as class attributes.

    Attributes
    ----------

    @attr   'initial_comp_sequence' : list(str)
            Sequence of compartment types that define initial structure of axon

    @attr   'repeating_comp_sequence' : list(str)
            Sequence of compartment types that define repeating structure of axon

    @attr   'nodal_compartment_type' : str
            Name of compartment type representing node of Ranvier

    @attr   'compartment_defs' : dict[str, dict[str, object]]
            Dictionary mapping names of compartment types to the keys
            - 'mechanisms' : dict[<mechanism name> , <dict of parameters>]
            - 'passive' : dict[<parameter name>, <value>]
            - 'morphology': dict[<parameter name>, <value>]
    """

    def __init__(self, logger=None, without_extracellular=False):
        """
        Define compartments types that constitute the axon model.

        @post   attributes 'compartment_defs' and 'repeating_comp_sequence'
                have been set.
        """
        self.logger = logger
        self.without_extracellular = without_extracellular


    def estimate_num_sections(self):
        """
        Estimate number of of Sections needed to build axon along streamline.
        """
        tck_length_mm = np.sum(np.linalg.norm(np.diff(self.streamline_pts, axis=0), axis=1))
        rep_length_um = sum((self.compartment_defs[sec]['morphology']['L'] 
                                    for sec in self.repeating_comp_sequence))
        return 1e3 * tck_length_mm / rep_length_um * len(self.repeating_comp_sequence)


    def get_initial_diameter(self):
        """
        Get diameter of first axonal section
        """
        if len(self.initial_comp_sequence) > 0:
            comp_type = self.initial_comp_sequence[0]
        else:
            comp_type = self.repeating_comp_sequence[0]
        return self.compartment_defs[comp_type]['morphology']['diam']


    def _set_comp_attributes(self, sec, sec_attrs):
        """
        Create compartment from properties.
        """
        # Insert mechanisms and set their parameters
        for mech_name, mech_params in sec_attrs['mechanisms'].items():
            if mech_name == 'extracellular' and self.without_extracellular:
                continue
            sec.insert(mech_name)
            for pname, pval in mech_params.items():
                if mech_name == 'extracellular':
                    for i in range(2): # default nlayer = 2
                        getattr(sec, pname)[i] = pval
                else:
                    setattr(sec, '{}_{}'.format(pname, mech_name), pval)
        
        # Set passive parameters
        for pname, pval in sec_attrs['passive'].items():
                setattr(sec, pname, pval)

        # Number of segments (discretization)
        sec.nseg = sec_attrs['morphology'].get('nseg', 1)
        return sec


    def _next_node_dist(self, i_sequence, measure_from=1):
        """
        Distance to next node in mm.
        """
        if measure_from==1 and i_sequence==len(self.repeating_comp_sequence)-1:
            return 0.0
        i_start = i_sequence+1 if measure_from==1 else i_sequence
        dist_microns = sum((self.compartment_defs[t]['morphology']['L']
                                for t in self.repeating_comp_sequence[i_start:]))
        return 1e-3 * dist_microns


    def _set_streamline_length(self):
        """
        Calculate length of streamline and set array of distances
        from start of streamline to each intermediate point.
        """
        # length = np.sum(np.linalg.norm(np.diff(self.streamline_pts, axis=0), axis=1))
        dvecs = np.diff(self.streamline_pts, axis=0)
        dvecs_norms = np.sqrt(np.sum(dvecs*dvecs, axis=1))
        self.streamline_length = np.sum(dvecs_norms)
        self.streamline_segment_lengths = np.concatenate(([0.0], dvecs_norms))


    def _get_remaining_arclength(self):
        """
        Get remaining length of axon to be built by counting length
        alone streamline from the last compartment endpoint.
        """
        if self.num_passed >= self.num_streamline_pts:
            return 0.0
        dist_stop2next = veclen(self.last_coord - self.streamline_pts[self.num_passed])
        if self.num_passed >= self.num_streamline_pts-1:
            return dist_stop2next
        dist_next2end = np.sum(self.streamline_segment_lengths[self.num_passed+1:])
        return dist_stop2next + dist_next2end


    def _get_segment_vec(self, i):
        """
        Get the vector (b - a) of two consecutive streamline points
        starting at index i.

        For the last point the vector between the previous point is returned.
        """
        if i == len(self.streamline_pts) - 1:
            i = i - 1
        return np.diff(self.streamline_pts[i:i+2], axis=0).reshape((-1,))


    def _walk_arclength(self, dist):
        """
        Walk for given distance along streamline.

        If you walk past a streamline point, the arclength between
        start and stop point will be larger than the cartestian distance.
        Hence the actual compartment length will be shorter than intended.
        This shouldn't be a problem though since action potentials will just
        be slightly less attenuated.

        Returns
        -------

        @return     num_passed : int
                    Number of streamline points passed during walk.

        @return     stop_coord : np.array[float] (3x1)
                    Coordinates of endpoint of walk.

        @return     next_tangent : np.array[float] (3x1)
                    Unit tangent vector at endpoint of walk
        """
        remaining_pts = self.streamline_pts[self.num_passed:]
        last_walk_coord = self.last_coord
        dist_walked = 0.0
        num_passed = 0

        # Keep walking until we've walked for <dist> mm
        while dist_walked < dist:
            # As long as we are not at the end, set waypoint to next streamline coordinate
            if num_passed < len(remaining_pts):
                waypoint_coord = remaining_pts[num_passed]
            else:
                # walking distance extends beyond end of streamline
                print('Extending axon beyond streamline endpoint for {} mm'.format(
                       dist-dist_walked))
                waypoint_coord = last_walk_coord + 1000.0 * self.last_tangent
            
            # Walk up to next stopping point
            path_vec = waypoint_coord - last_walk_coord
            dist_waypoint = veclen(path_vec) # distance to next streamline point

            # Choose to stop before or after next route point
            if dist_walked + dist_waypoint < dist:
                # end of walk is beyond next streamline point
                last_walk_coord = waypoint_coord
                dist_walked += dist_waypoint
                num_passed += 1
            else:
                # end of walk is before next streamline point
                tangent_vec = path_vec / dist_waypoint
                stop_pt = last_walk_coord + (dist - dist_walked) * tangent_vec
                break

        return num_passed, stop_pt, tangent_vec


    def _walk_cartesian_length(self, dist):
        """
        Walk along streamline until cartesian distance to last stopping point
        is equal to <dist>.

        Returns
        -------

        @return     stop_coord : np.array[float] (3x1)
                    Coordinates of endpoint of walk.

        @return     next_tangent : np.array[float] (3x1)
                    Unit tangent vector at endpoint of walk

        TODO: fix bug where num_passed not updated correctly (case 1 & 2)
        """
        if self.num_passed >= self.num_streamline_pts:
            remaining_pts = []
        else:
            remaining_pts = self.streamline_pts[self.num_passed:]
        
        start_walk_coord = self.last_coord
        walk_passed = 0 # streamline points passed during this walk

        # Walk along remaining points until next point is farther than walking distance
        i_pre = -1  # index (offset) of next point _closer_ than distance
        i_post = -1 # index (offset) of next point _farther_ than distance
        for i, waypoint in enumerate(remaining_pts):
            dist_waypoint = veclen(waypoint - start_walk_coord)
            if dist_waypoint <= dist:
                i_pre = i
            else:
                i_post = i
                break

        # Identify one of four cases
        colinear = False
        if (i_pre >= 0) and (i_post >= 0):
            # We are moving from one line segment to one of the following
            # line segments.
            colinear = False
            walk_passed = i_pre + 1
            p1 = remaining_pts[i_pre]
            p2 = remaining_pts[i_post]
        elif i_post == 0:
            # Stopping point is on the current line segment
            colinear = True
            walk_passed = 0
            tangent = normvec(self._get_segment_vec(self.num_passed))
        elif (i_pre == -1) and (i_post == -1):
            # We are past the end of the streamline (extending it)
            assert self.num_passed == self.num_streamline_pts
            colinear = True
            walk_passed = 0
            tangent = self.last_tangent
        elif (i_post == -1) and (i_pre == 0):
            # We are on last line segment and stopping point is beyond endpoint
            # of streamline.
            assert self.num_passed == self.num_streamline_pts - 1
            colinear = True
            walk_passed = 1
            tangent = normvec(self._get_segment_vec(self.num_streamline_pts-1))
        elif (i_pre != -1) and (i_post == -1):
            # We are moving from a line segment to beyond the endpoint of
            # the streamline
            colinear = False
            streamline_i_pre = self.num_passed + i_pre + 1
            assert streamline_i_pre == self.num_streamline_pts - 1
            walk_passed = i_pre + 1
            tangent = normvec(self._get_segment_vec(streamline_i_pre))
            p1 = remaining_pts[i_pre]
            p2 = p1 + 100.0 * tangent
        else:
            assert False, "Unknown condition, should not occur."

        # Choose the interpolation method based on the case
        if colinear:
            # Stopping point is colinear with the current line segment
            stop_pt = start_walk_coord + dist * tangent

        else:
            # Stopping point lies on [p1, p2) and is not necessarily colinear
            # with the line segment
            p0 = start_walk_coord

            # Find point on [p1, p2) that yields cartesian distance to p0
            # TODO: check calculation
            u12 = normvec(p2 - p1)
            v01 = p1 - p0
            coefficients = [                # a*x^2 + b*x + c
                np.dot(u12, u12),           # a
                2 * np.dot(u12, v01),       # b
                np.dot(v01, v01) - dist**2, # c
            ]
            roots_all = np.roots(coefficients)
            roots_real = roots_all[np.isreal(roots_all)]
            alpha = np.max(roots_real)
            assert alpha > 0

            # Calculate stop point
            stop_pt = p1 + alpha * u12
            tangent = u12

        return walk_passed, stop_pt, tangent


    def _connect_axon(self, parent_cell, parent_sec, connection_method,
                      tolerance_mm=1e-3):
        """
        Ensure axon is connected to parent cell, both geometrically (in 3D space)
        and electrically (in NEURON).

        @see    build_along_streamline.
        """
        # Get connection point on parent cell (assume last 3D point)
        n3d = int(h.n3d(sec=parent_sec))
        parent_coords = np.array([h.x3d(n3d-1, sec=parent_sec),
                                  h.y3d(n3d-1, sec=parent_sec),
                                  h.z3d(n3d-1, sec=parent_sec)])

        # Connect axon according to method
        if connection_method == 'orient_coincident':
            # Check which end of streamline is coincident with connection
            # point and build in appropriate direction
            if np.allclose(parent_coords, self.streamline_pts[-1], atol=tolerance_mm):
                self.streamline_pts = self.streamline_pts[::-1] # reverse
            elif not np.allclose(parent_coords, self.streamline_pts[0], atol=tolerance_mm):
                raise ValueError("Start or end of streamline must be coincident"
                        "with endpoint of parent section ({})".format(
                            parent_coords))

        elif connection_method.startswith('translate_axon'):
            # Translate axon start or end to connection point
            if connection_method.endswith('start'):
                streamline_origin = self.streamline_pts[0]
            elif connection_method.endswith('end'):
                streamline_origin = self.streamline_pts[-1]
            else:
                raise ValueError(connection_method)

            translate_vec = parent_coords - streamline_origin
            self.streamline_pts = self.streamline_pts - translate_vec # broadcasts

        elif connection_method.startswith('translate_cell'):
            # Translate cell so that connection point is coincident with
            # start or end of streamline
            if connection_method.endswith('start'):
                target_pt = self.streamline_pts[0]
            elif connection_method.endswith('end'):
                target_pt = self.streamline_pts[-1]
            else:
                raise ValueError(connection_method)

            translate_vec = target_pt - parent_coords
            translate_mat = np.eye(4)
            translate_mat[:3,3] = translate_vec
            morph_3d.transform_sections(parent_cell.all, translate_mat)


    def build_along_streamline(self, streamline_coords, terminate='nodal_cutoff',
                               tolerance_mm=1e-6, interp_method='cartesian',
                               parent_cell=None, parent_sec=None,
                               connection_method='translate_axon',
                               raise_if_existing=True):
        """
        Build NEURON axon along a sequence of coordinates.

        Arguments
        ---------

        @param  terminate : str

                - 'any' to terminate as soon as last compartment extends beyond
                  endpoint of streamline

                - 'nodal_extend' to terminate axon with nodal compartment
                   extended beyond streamline endpoint if necessary

                - 'nodal_cutoff' to terminate axon with nodal compartment
                   before streamline endpoint is reached


        @param  tolerance_mm : float

                Tolerance on length of compartments caused by discrepancy
                between streamline arclength and length of compartments spanning
                a streamline node.

        @param  connection_method : str
                
                Method for connecting the reconstructed acon to the parent
                sections. One of the following:

                'orient_coincident': find streamline end that connects to the 
                parent section and start building from this point. Raise
                exception if not coincident.

                'translate_axon_<start/end>': translate starting or endpoint
                of axon to endpoint of parent section.

                'translate_cell_<start/end>': Translate cell so that connection 
                point is coincident with start or end of streamline

        Returns
        -------

        @return     sections : list[nrn.Section]
                    List of axonal sections
        """
        # Compartments have fixed distances, so need to advance between streamline
        # points using interpolation.
        

        self.streamline_pts = np.array(streamline_coords)
        if parent_sec is not None:
            self._connect_axon(parent_cell, parent_sec, connection_method,
                               tolerance_mm=1e-3)

        # Save streamline info
        self.num_streamline_pts = len(self.streamline_pts)
        self._set_streamline_length()
        
        # State variables for building algorithm
        self.interp_pts = []        # interpolated points
        self.i_compartment = 0      # index in sequence of compartments
        self.num_passed = 1         # index of last passed streamline point
                                    # (we already 'passed' starting point)
        
        self.interp_pts = [self.streamline_pts[0]]        # interpolated points
        self.last_coord = self.streamline_pts[0]
        self.last_tangent = normvec(self.streamline_pts[1] - streamline_coords[0])
        self.built_length = 0.0

        # Walk to its endpoint
        if interp_method == 'arclength':
            walk_func = self._walk_arclength
        elif interp_method == 'cartesian':
            walk_func = self._walk_cartesian_length
        else:
            raise ValueError('Unknown inerpolation method: ', interp_method)
        
        sec_by_type = {sec_type: [] for sec_type in self.compartment_defs.keys()}
        sec_ordered = []
        # num_ax_sec = 0
        # if parent_cell is not None:
        #     num_ax_sec = len(list(parent_cell.axonal))

        n_repeating = len(self.repeating_comp_sequence)
        MAX_NUM_COMPARTMENTS = int(1e9)
        est_num_comp = self.estimate_num_sections()
        self.debug("Estimated number of compartments to build axon: {}".format(est_num_comp))
        if est_num_comp > MAX_NUM_COMPARTMENTS:
            raise ValueError('Streamline too long (estimated number of '
                'compartments needed is {}'.format(est_num_comp))

        for i_compartment in xrange(MAX_NUM_COMPARTMENTS):

            # Look up section type for current point in the chain
            self.i_compartment = i_compartment
            if i_compartment < len(self.initial_comp_sequence):
                # We are in the initial section of the axon (non-repeating structure)
                i_sequence = i_compartment
                sec_type = self.initial_comp_sequence[i_sequence]
            else:
                # We are in the repeating part of the axon
                i_sequence = i_compartment % n_repeating
                sec_type = self.repeating_comp_sequence[i_sequence]
            sec_attrs = self.compartment_defs[sec_type]
            sec_L_mm = sec_attrs['morphology']['L'] * 1e-3 # um to mm

            # Create the compartment
            sec_name = "{:s}[{:d}]".format(sec_type, len(sec_by_type[sec_type]))
            # sec_name = "axon[{:d}]".format(num_ax_sec)
            # num_ax_sec += 1
            if parent_cell is None:
                ax_sec = h.Section(name=sec_name)
            else:
                # see module neuron.__init__
                ax_sec = h.Section(name=sec_name, cell=parent_cell)
            
            # Set its properties and connect it
            self._set_comp_attributes(ax_sec, sec_attrs)
            if parent_sec is not None:
                ax_sec.connect(parent_sec(1.0), 0.0)
            
            parent_sec = ax_sec
            sec_by_type[sec_type].append(ax_sec)
            sec_ordered.append(ax_sec)
            
            # Find section endpoint by walking along streamline for sec.L
            num_passed, stop_coord, next_tangent = walk_func(sec_L_mm)
            self.interp_pts.append(stop_coord)
            
            # Check compartment length vs tolerance
            real_length = veclen(stop_coord - self.last_coord)
            if not np.isclose(real_length, sec_L_mm, atol=tolerance_mm):
                self.logger.warning(
                    'WARNING: exceed length tolerance ({}) '
                    ' in compartment compartment {} : L = {}'.format(
                        tolerance_mm, ax_sec, real_length))

            # Add the 3D start and endpoint
            sec_endpoints = self.last_coord, stop_coord
            h.pt3dclear(sec=ax_sec)
            for coords_mm in sec_endpoints:
                coords_um = coords_mm * 1e3
                x, y, z = coords_um
                h.pt3dadd(x, y, z, sec_attrs['morphology']['diam'], sec=ax_sec)

            # Update state variables
            self.built_length += real_length
            self.num_passed += num_passed
            self.last_coord = stop_coord
            self.last_tangent = next_tangent

            # If terminating axon with nodal compartment: either cutoff or extrapolate
            # remaining_length = self._get_remaining_arclength() # FIXME: fix bug
            remaining_length = self.streamline_length - self.built_length

            if terminate == 'any_extend':
                if self.num_passed >= len(self.streamline_pts):
                    break
            elif terminate == 'any_cutoff':
                next_type = self.repeating_comp_sequence[i_compartment % n_repeating]
                next_length = self.compartment_defs[next_type]['morphology']['L'] * 1e-3
                if remaining_length - next_length <= 0:
                    break
            elif terminate == 'nodal_cutoff' and sec_type == self.nodal_compartment_type:
                next_node_dist = self._next_node_dist(i_sequence, measure_from=1)
                if next_node_dist > remaining_length:
                    break
            elif terminate == 'nodal_extend' and sec_type == self.nodal_compartment_type:
                if self.num_passed >= len(self.streamline_pts):
                    break

            # Sanity check: is axon too long?
            if i_compartment >= MAX_NUM_COMPARTMENTS-1:
                raise ValueError("Axon too long.")
            elif i_compartment >= 1.1 * est_num_comp:
                self.warning("Created {}-th section, more than estimate {}".format(
                                i_compartment, est_num_comp))
        
        self.debug("Created %i axonal compartments", len(sec_ordered))

        # Add to parent cell
        if parent_cell is not None:
            # NOTE: refs to sections are not kept alive by appending them
            # to one of the the instantiated template's (icell's) SectionList.
            # There does not seem a way to store newly created sections
            # on the instantiated template (icell) from within python in a way
            # that references are kept alive. You cannot assign python
            # lists to the icell, and appending to its SectionLists
            # also does not work.

            # for sec_type, seclist in sec_by_type.items():
            #     existing = getattr(parent_cell, sec_type, None)
            #     if existing is None:
            #         setattr(parent_cell, sec_type, seclist)
            #     elif raise_if_existing:
            #         raise Exception(
            #             "Parent cell {} has existing section list '{}': {}".format(
            #              parent_cell, sec_type, existing))
            #     else:
            #         old_secs = list(existing)
            #         old_secs.extend(seclist)
            #         setattr(parent_cell, sec_type, seclist)

            # Add to axonal SectionList in order of connection
            append_to = ['all', 'axonal']
            for seclist_name in append_to:
                seclist = getattr(parent_cell, seclist_name, None)
                if seclist is not None:
                    # DOES NOT KEEP ALIVE REFS:
                    for ax_sec in sec_ordered:
                        seclist.append(sec=ax_sec)
                    self.debug("Updated SectionList '%s' of %s", seclist_name, parent_cell)


        return sec_by_type


# Make logging functions
for level in logging.INFO, logging.DEBUG, logging.WARN:
    def make_func(level=level):
        # inner function solves late binding issue
        def log_func(self, *args, **kwargs):
            if self.logger is None:
                return
            self.logger.log(level, *args, **kwargs)
        return log_func
    level_name = logging.getLevelName(level).lower()
    setattr(AxonBuilder, level_name, make_func(level))