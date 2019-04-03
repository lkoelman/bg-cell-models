"""
Base class for axon builders.
"""

from __future__ import division # float division for int literals, like Hoc
import math

import numpy as np
import neuron
h = neuron.h


PI = math.pi

def vecnorm(a, b):
    """
    Normalized vector pointing from a to b.
    """
    return (b - a) / np.sqrt(np.dot(a, b))


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

    def __init__(self):
        """
        Define compartments types that constitute the axon model.

        @post   attributes 'compartment_defs' and 'compartment_sequence'
                have been set.
        """
        raise NotImplementedError('Implement __init__ in subclass to set axon properties.')


    def _make_compartment(self, comp_attrs, name):
        """
        Create compartment from properties.
        """
        sec = h.Section(name=name)
        # Insert mechanisms and set their parameters
        for mech_name, mech_params in comp_attrs['mechanisms'].items():
            sec.insert(mech_name)
            for pname, pval in mech_params.items():
                setattr(sec, '{}_{}'.format(pname, mech_name), pval)
        # Set passive parameters
        for pname, pval in comp_attrs['passive'].items():
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
        dvecs = np.diff(self.streamline_pts, axis=0)
        dvecs_norms = np.sqrt(np.sum(dvecs*dvecs, axis=1))
        self.streamline_length = np.sum(dvecs_norms)
        self.streamline_vecnorms = np.concatenate(([0.0], dvecs_norms))


    def _get_remaining_arclength(self):
        """
        Get remaining length of axon to be built by counting length
        alone streamline from the last compartment endpoint.

        TODO: does not work if last_endpt extrapolated beyond streamline
        """
        comp_to_tck = veclen(self.last_endpt-self.streamline_pts[self.i_streamline_pt+1])
        tck_to_end = np.sum(self.streamline_vecnorms[self.i_streamline_pt+2:])
        return comp_to_tck + tck_to_end


    def _walk_along_interp(self, dist):
        """
        Walk for distance 'dist' starting from coordinate 'last_xyz' along
        sequence of following coordinates 'next_xyz_seq' using interpolation.
        """
        remaining_pts = self.streamline_pts[self.i_streamline_pt+1:]
        last_pt = self.last_endpt
        dist_walked = 0.0
        num_passed = 0
        while dist_walked < dist:
            if num_passed < len(remaining_pts):
                target_pt = remaining_pts[num_passed]
            else:
                # walking distance extends beyond end of streamline
                print('Extending axon beyond streamline endpoint for {} mm'.format(
                       dist-dist_walked))
                target_pt = last_pt + 1000.0 * self.last_tangent
            
            # Walk up to next stopping point
            path_vec = target_pt - last_pt
            pt_dist = np.sqrt(np.dot(path_vec, path_vec)) # distance to next streamline point
            if dist_walked + pt_dist < dist:
                # end of walk is beyond next streamline point
                dist_walked += pt_dist
                last_pt = target_pt
                num_passed += 1
            else:
                # end of walk is before next streamline point
                tangent_vec = path_vec / pt_dist
                stop_pt = last_pt + (dist-dist_walked) * tangent_vec
                return num_passed, stop_pt, tangent_vec


    def build_along_streamline(self, streamline_xyz, terminate='nodal_cutoff',
                               tol_comp_len=0.05):
        """

        @param  terminate : str
                - 'any' to terminate as soon as last compartment extends beyond
                  endpoint of streamline
                - 'nodal_extend' to terminate axon with nodal compartment
                   extended beyond streamline endpoint if necessary
                - 'nodal_cutoff' to terminate axon with nodal compartment
                   before streamline endpoint is reached

        @param  tol_comp_len : float
                Tolerance on length of compartments caused by discrepancy
                between streamline arclength and length of compartments spanning
                a streamline node.
        """
        # Compartments have fixed distances, so need to advance between streamline
        # points using interpolation.
        n_repeating = len(self.compartment_sequence)

        # State variables for building algorithm
        self.i_compartment = 0 # index in sequence of compartments
        self.streamline_pts = streamline_xyz
        self._set_streamline_length()
        self.i_streamline_pt = 0 # index of last passed streamline point
        
        self.last_endpt = self.streamline_pts[0]
        self.last_tangent = vecnorm(streamline_xyz[0], streamline_xyz[1])
        self.built_length = 0.0
        
        axon_sections = []

        while True:
            # What kind of section must we create?
            i_sequence = self.i_compartment % n_repeating
            comp_type = self.compartment_sequence[i_sequence]
            comp_attrs = self.compartment_defs[comp_type]
            comp_length = comp_attrs['morphology']['L'] * 1e-3 # um to mm

            # Create the compartment
            ax_sec = self._make_compartment(comp_attrs, 
                        '{}_{}'.format(comp_type, self.i_compartment))
            if len(axon_sections) > 0:
                ax_sec.connect(axon_sections[-1](1.0), 0.0)
            axon_sections.append(ax_sec)
            
            # Walk to its endpoint
            num_passed, next_endpt, next_tangent = self._walk_along_interp(
                                                    comp_length)
            
            # Check compartment length vs tolerance
            real_length = veclen(next_endpt - self.last_endpt)
            if not (1.0-tol_comp_len <= real_length/comp_length <= 1.0+tol_comp_len):
                print('WARNING: tolerance of {} on compartment length not met '
                      'for compartment {} (L={})'.format(tol_comp_len, ax_sec, real_length))

            # Add the 3D start and endpoint
            h.pt3dclear(sec=ax_sec)
            for x, y, z in self.last_endpt, next_endpt:
                h.pt3dadd(x, y, z, comp_attrs['morphology']['diam'], sec=ax_sec)

            # Update state variables
            self.built_length += comp_length
            self.i_compartment += 1
            self.i_streamline_pt += num_passed
            self.last_endpt = next_endpt
            self.last_tangent = next_tangent

            # If terminating axon with nodal compartment: either cutoff or extrapolate
            remaining_length = self._get_remaining_arclength()
            if terminate == 'any_extend':
                if self.i_streamline_pt >= len(streamline_xyz)-1:
                    break
            elif terminate == 'any_cutoff':
                next_type = self.compartment_sequence[self.i_compartment % n_repeating]
                next_length = self.compartment_defs[next_type]['morphology']['L'] * 1e-3
                if remaining_length - next_length <= 0:
                    break
            elif terminate == 'nodal_cutoff' and comp_type == self.nodal_compartment_type:
                next_node_dist = self._next_node_dist(i_sequence, measure_from=1)
                if next_node_dist > remaining_length:
                    break
            elif terminate == 'nodal_extend' and comp_type == self.nodal_compartment_type:
                if self.i_streamline_pt >= len(streamline_xyz)-1:
                    break

        return axon_sections
