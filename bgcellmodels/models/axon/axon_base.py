"""
Base class for axon builders.
"""

from __future__ import division # float division for int literals, like Hoc
import math

import numpy as np
import neuron
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

    @attr   'compartment_sequence' : list(str)
            Sequence of compartment types that define repeating structure of axon

    @attr   'nodal_compartment_type' : str
            Name of compartment type representing node of Ranvier

    @attr   'compartment_defs' : dict[str, dict[str, object]]
            Dictionary mapping names of compartment types to the keys
            - 'mechanisms' : dict[<mechanism name> , <dict of parameters>]
            - 'passive' : dict[<parameter name>, <value>]
            - 'morphology': dict[<parameter name>, <value>]
    """

    def __init__(self, logger=None):
        """
        Define compartments types that constitute the axon model.

        @post   attributes 'compartment_defs' and 'compartment_sequence'
                have been set.
        """
        raise NotImplementedError('Implement __init__ in subclass to set axon properties.')


    def _set_comp_attributes(self, sec, sec_attrs):
        """
        Create compartment from properties.
        """
        # Insert mechanisms and set their parameters
        for mech_name, mech_params in sec_attrs['mechanisms'].items():
            sec.insert(mech_name)
            for pname, pval in mech_params.items():
                setattr(sec, '{}_{}'.format(pname, mech_name), pval)
        # Set passive parameters
        for pname, pval in sec_attrs['passive'].items():
            if pname in ('xraxial', 'xg', 'xc'):
                for i in range(2): # default nlayer = 2
                    getattr(sec, pname)[i] = pval
            else:
                setattr(sec, pname, pval)
        return sec


    def _next_node_dist(self, i_sequence, measure_from=1):
        """
        Distance to next node in mm.
        """
        if measure_from==1 and i_sequence==len(self.compartment_sequence)-1:
            return 0.0
        i_start = i_sequence+1 if measure_from==1 else i_sequence
        dist_microns = sum((self.compartment_defs[t]['morphology']['L']
                                for t in self.compartment_sequence[i_start:]))
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

        TODO: does not work if last_coord extrapolated beyond streamline
        """
        dist_stop2next = veclen(self.last_coord - self.streamline_pts[self.i_streamline_pt+1])
        dist_next2end = np.sum(self.streamline_segment_lengths[self.i_streamline_pt+2:])
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

        Returns
        -------

        @return     num_passed : int
                    Number of streamline points passed during walk.

        @return     stop_coord : np.array[float] (3x1)
                    Coordinates of endpoint of walk.

        @return     next_tangent : np.array[float] (3x1)
                    Unit tangent vector at endpoint of walk
        """
        remaining_pts = self.streamline_pts[self.i_streamline_pt+1:]
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
                return num_passed, stop_pt, tangent_vec


    def _walk_cartesian_length(self, dist):
        """
        Walk along streamline until cartesian distance to last stopping point
        is equal to <dist>.

        Returns
        -------

        @return     num_passed : int
                    Number of streamline points passed during walk.

        @return     stop_coord : np.array[float] (3x1)
                    Coordinates of endpoint of walk.

        @return     next_tangent : np.array[float] (3x1)
                    Unit tangent vector at endpoint of walk
        """
        remaining_pts = self.streamline_pts[self.i_streamline_pt+1:]
        start_walk_coord = self.last_coord
        num_passed = 0

        # Find between which streamline points the stopping point lies
        i_pre = -1
        i_post = -1
        for i, waypoint in enumerate(remaining_pts):
            dist_waypoint = veclen(waypoint - start_walk_coord)
            if dist_waypoint <= dist:
                i_pre = i
            else:
                i_post = i
                break

        # Identify one of four cases
        colinear = False
        if i_post == 0:
            # Stopping point is on the current line segment
            colinear = True
            num_passed = 0
            tangent = normvec(self._get_segment_vec(self.i_streamline_pt))
        elif (i_post == -1) and (i_pre == 0):
            # We are on last line segment and stopping point is beyond endpoint
            # of streamline.
            assert len(remaining_pts) == 1
            colinear = True
            num_passed = 1
            tangent = normvec(self._get_segment_vec(self.num_streamline_pts-1))
        elif (i_pre != -1) and (i_post != -1):
            # We are moving from one line segment to one of the following
            # line segments.
            colinear = False
            num_passed = i_pre + 1
            p1 = remaining_pts[i_pre]
            p2 = remaining_pts[i_post]
        elif (i_pre != -1) and (i_post == -1):
            # We are moving from a line segment to beyond the endpoint of
            # the streamline
            colinear = False
            streamline_i_pre = self.i_streamline_pt + i_pre + 1
            assert streamline_i_pre == len(self.streamline_pts) - 1
            num_passed = i_pre + 1
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

        return num_passed, stop_pt, tangent



    def build_along_streamline(self, streamline_xyz, terminate='nodal_cutoff',
                               tolerance_mm=1e-6, interp_method='cartesian',
                               parent_cell=None, parent_sec=None):
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

        Returns
        -------

        @return     sections : list[nrn.Section]
                    List of axonal sections
        """
        # Compartments have fixed distances, so need to advance between streamline
        # points using interpolation.
        n_repeating = len(self.compartment_sequence)

        # Save streamline info
        self.streamline_pts = streamline_xyz
        self.num_streamline_pts = len(self.streamline_pts)
        self._set_streamline_length()
        
        # State variables for building algorithm
        self.interp_pts = []        # interpolated points
        self.i_compartment = 0      # index in sequence of compartments
        self.i_streamline_pt = 0    # index of last passed streamline point
        
        self.interp_pts = [self.streamline_pts[0]]        # interpolated points
        self.last_coord = self.streamline_pts[0]
        self.last_tangent = normvec(streamline_xyz[1] - streamline_xyz[0])
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

        MAX_NUM_COMPARTMENTS = int(1e9)
        for i_compartment in xrange(MAX_NUM_COMPARTMENTS):

            # What kind of section must we create?
            self.i_compartment = i_compartment
            i_sequence = i_compartment % n_repeating
            sec_type = self.compartment_sequence[i_sequence]
            sec_attrs = self.compartment_defs[sec_type]
            sec_L_mm = sec_attrs['morphology']['L'] * 1e-3 # um to mm

            # Create the compartment
            sec_name = "{:s}[{:d}]".format(sec_type, len(sec_by_type[sec_type]))
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
                print('WARNING: tolerance of {} on compartment length not met '
                      'for compartment {} (L={})'.format(tolerance_mm, ax_sec, real_length))

            # Add the 3D start and endpoint
            sec_endpoints = self.last_coord, stop_coord
            h.pt3dclear(sec=ax_sec)
            for coords_mm in sec_endpoints:
                coords_um = coords_mm * 1e3
                x, y, z = coords_um
                h.pt3dadd(x, y, z, sec_attrs['morphology']['diam'], sec=ax_sec)

            # Update state variables
            self.built_length += sec_L_mm
            self.i_streamline_pt += num_passed
            self.last_coord = stop_coord
            self.last_tangent = next_tangent

            # If terminating axon with nodal compartment: either cutoff or extrapolate
            remaining_length = self._get_remaining_arclength()
            if terminate == 'any_extend':
                if self.i_streamline_pt >= len(streamline_xyz)-1:
                    break
            elif terminate == 'any_cutoff':
                next_type = self.compartment_sequence[i_compartment % n_repeating]
                next_length = self.compartment_defs[next_type]['morphology']['L'] * 1e-3
                if remaining_length - next_length <= 0:
                    break
            elif terminate == 'nodal_cutoff' and sec_type == self.nodal_compartment_type:
                next_node_dist = self._next_node_dist(i_sequence, measure_from=1)
                if next_node_dist > remaining_length:
                    break
            elif terminate == 'nodal_extend' and sec_type == self.nodal_compartment_type:
                if self.i_streamline_pt >= len(streamline_xyz)-1:
                    break

        if i_compartment >= MAX_NUM_COMPARTMENTS-1:
            raise ValueError("Axon too long.")

        # Add to parent cell
        if parent_cell is not None:
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
            axonal = getattr(parent_cell, 'axonal', None)
            if axonal is not None:
                for ax_sec in sec_ordered:
                    axonal.append(sec=ax_sec)


        return sec_by_type
