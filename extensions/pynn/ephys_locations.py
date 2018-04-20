"""
Module for working with BluePyOpt Ephys locations in PyNN.

@author     Lucas Koelman

@date       14/02/2018
"""

import numpy as np
from pyNN.neuron import h, state as nrn_state

from common import nrnutil

from bluepyopt.ephys.locations import Location, EPhysLocInstantiateException
from bluepyopt.ephys.serializer import DictMixin


rng_structural_variability = h.Random(
    nrn_state.mpi_rank + nrn_state.native_rng_baseseed)


# class SynapseLocation(Location, DictMixin):
#     """
#     Random segment between min and max distance from soma.
#     """

#     SERIALIZED_FIELDS = (
#         'name',
#         'comment',
#         'syn_mech_names')


#     def __init__(
#             self,
#             name,
#             syn_mech_names=None,
#             comment=''):
#         """
#         Constructor

#         @param      name : str
#                     Name of this object

#         @param      syn_mech_names : list(str)
#                     List of NEURON synapse mechanism names that are supported
#                     by this location
#         """

#         super(SynapseLocation, self).__init__(name, comment)

#         self.syn_mech_names = syn_mech_names # could also be set after creation

#         # Attributes to be set by owner icell
#         self.sim = None
#         self.icell = None


#     def __getattr__(self, name):
#         """
#         Override so synapse objects can be requested as attributes.

#         @note   __getattr__ is only called as a last resort i.e. if there are no
#                 attributes in the instance that match the name.
#         """
#         if name in self.syn_mech_names:
#             return self.get_synapse(name)
#         else:
#             raise AttributeError("Synapse model '{}' not supported by "
#                     "this location".format(name))


#     def get_synapse(self, mechanism_name):
#         """
#         Get a synapse with given mechanism at this location.

#         If a synapse of the given type (mechanism_name) is already present,
#         return that synapse. If not, make a new synapse and return it.

#         @return     synapse : nrn.HocObject
#                     Instantiated Neuron POINT_PROCESS mechanism
#         """
#         iseg = self.instantiate(sim=self.sim, icell=self.icell)
#         seg_pps = iseg.point_processes()

#         target_syn = next((syn for syn in seg_pps if 
#                         nrnutil.get_mod_name(syn)==mechanism_name), None)
        
#         if target_syn is None:
#             constructor = getattr(h, mechanism_name)
#             target_syn = constructor(iseg)

#         # TODO: retrieve synapse parameters from somewhere, e.g. parent cell param space or see connection creation
#         return target_syn


class SomaDistanceRangeLocation(Location, DictMixin):
    """
    Random segment between min and max distance from soma.
    """
    
    SERIALIZED_FIELDS = (
        'name',
        'comment',
        #'syn_mech_names',
        'seclist_name',
        'min_soma_distance',
        'max_soma_distance')


    def __init__(
            self,
            name,
            seclist_name=None,
            min_distance=None,
            max_distance=None,
            #syn_mech_names=None,
            comment=''):
        """
        Constructor

        @param      name : str
                    Name of this object

        @param      seclist_name : str
                    Name of Neuron section list (ex: 'apical')

        @param      syn_mech_names : list(str)
                    List of NEURON synapse mechanism names
        """

        super(SomaDistanceRangeLocation, self).__init__(
            name,
            #syn_mech_names=syn_mech_names,
            comment=comment)

        self.min_soma_distance = min_distance
        self.max_soma_distance = max_distance
        self.seclist_name = seclist_name

        # Attributes to be set by owner icell
        self.rng = None


    def instantiate(self, sim=None, icell=None):
        """
        Find the instantiate compartment
        """

        soma = icell.soma[0]

        sim.neuron.h.distance(0, 0.5, sec=soma)

        iseclist = getattr(icell, self.seclist_name)

        # Gather all sections and their x-ranges that satisfy the distance criterion
        eligible_sections = []
        sec_eligible_ranges = [] # eligible x-interval within section (0,1)
        sec_sample_distance = [] # total accumulated distance to section (for sampling)
        sample_distance = 0.0 # total accumulated distance

        for isec in iseclist:
            start_distance = sim.neuron.h.distance(1, 0.0, sec=isec)
            end_distance = sim.neuron.h.distance(1, 1.0, sec=isec)

            sec_min_distance = min(start_distance, end_distance)
            sec_max_distance = max(start_distance, end_distance)

            if (sec_max_distance > self.min_soma_distance and 
                sec_min_distance < self.max_soma_distance):

                sec_min_eligible_dist = max(self.min_soma_distance, sec_min_distance)

                sec_x_lo = (float(sec_min_eligible_dist - sec_min_distance) /
                            (sec_max_distance - sec_min_distance))

                sec_max_eligible_dist = min(self.max_soma_distance, sec_max_distance)
                
                sec_x_hi = (float(sec_max_eligible_dist - sec_min_distance) /
                            (sec_max_distance - sec_min_distance))

                eligible_sections.append(isec)
                sec_eligible_ranges.append((sec_x_lo, sec_x_hi))

                sample_distance += isec.L * abs(sec_x_hi - sec_x_lo)
                sec_sample_distance.append(sample_distance)

        if len(eligible_sections) == 0:
            raise EPhysLocInstantiateException(
                'No comp in distance range {}-{} um from soma'.format(
                self.min_soma_distance, self.max_soma_distance))

        # Pick one segment at random from eligible ones
        global rng_structural_variability
        fraction = rng_structural_variability.uniform(0, 1)

        pick_distance = fraction * sample_distance
        pick_index = next(
            (i for (i,dist) in enumerate(sec_sample_distance) if (
                (i==0 and pick_distance <= sec_sample_distance[i]) or
                (sec_sample_distance[i-1] < pick_distance <= sec_sample_distance[i]))), 
            None)

        if pick_index is None:
            # Could only happed due to numerical error e.g. if x-interval too small
            # pick_index = 0
            raise EPhysLocInstantiateException('Error in picking random segment')

        # Interpolate the picked section
        if pick_index == 0:
            dist_lo, dist_hi = 0.0, sec_sample_distance[0]
        else:
            dist_lo, dist_hi = sec_sample_distance[pick_index-1:pick_index+1]
        x_lo, x_hi = sec_eligible_ranges[pick_index]
        
        sec_x_pick = np.interp(pick_distance, [dist_lo, dist_hi], [x_lo, x_hi])
        icomp = eligible_sections[pick_index](sec_x_pick)

        return icomp


    def __str__(self):
        """
        String representation of location.
        """
        return 'Distance range {}-{} micron from soma in {}'.format(
            self.min_soma_distance, self.max_soma_distance, self.seclist_name)


class SomaDistanceDiamLocation(Location, DictMixin):
    """
    All sections in a SectionList with diameter in a given range
    that lie in a given distance range from the soma compartment.
    """
    
    SERIALIZED_FIELDS = (
        'name',
        'comment',
        'seclist_name',
        'distance_range',
        'diameter_range')

    def __init__(
            self,
            name,
            seclist_name=None,
            distance_range=None,
            diameter_range=None,
            comment=''):
        """
        Constructor

        @param      name : str
                    Name of this object

        @param      seclist_name : str
                    Name of Neuron section list (ex: 'apical')

        @param      distance_range : tuple(float, float)

        @param      diameter_range : tuple(float, float)
        """

        super(SomaDistanceDiamLocation, self).__init__(
            name,
            comment=comment)

        self.distance_range = distance_range
        self.diameter_range = diameter_range
        self.seclist_name = seclist_name


    def instantiate(self, sim=None, icell=None):
        """
        Find the instantiate compartment

        - for sec in seclist:
          - get sections on path to soma
          - filter the ones that are in same seclist
          - if diam & distance constraint satisfied + parent/child constraints
            satisfied: return section
        """
        # see functions:
        #   - common/treeutils/path_segments()
        #   - clustering: recursive functions

        # !!! you have to know the parent category (cannot go higher than parent
        #     diam rating) so we have to do a complete traversal starting from soma
        # - find all subtree roots for sectionlist
        # - do traversal starting from each subtree root:
        # - first do a lookahead: check diameter of four next descendants along 
        #   each subtree path
        #   - descend + pass int (level) + save stack, if level is max, 
        #     save four top items on stack
        #   - assign diam rating based on diam & lookahead
        #   - save sec if dist range and diam range is OK
        #   - repeat recursively

        # calculate temp rating (based on interval)
        for sec in dfs_iter_tree_stack(subtree_root):
            paths_ahead, stack = [], []
            lookahead_diams(sec, 0, 4, paths_ahead, stack)
            for path in paths_ahead:
                ratings = [diam_rating(sec) for sec in path]
            # if 3/4 would get rating <= temp, assign temp, else asssign parent


    def __str__(self):
        """
        String representation of location.
        """
        return 'Distance range {}-{} micron from soma in {} with diameter {}-{} um'.format(
            self.distance_range[0], self.distance_range[1],
            self.seclist_name,
            self.diameter_range[0], self.diameter_range[1])


def lookahead_diams(sec, parent_dist, max_dist, paths, stack):
    """

    Usage
    -----
    
    >>> paths, stack = [], []
    >>> lookahead_diams(sec, 0, 4, paths, stack)
    
    """
    if parent_dist > max_dist:
        paths.append([stack.pop() for i in range(4)])
        return

    for child in getattr(sec, 'chidlren', []):
        stack.append(child)
        lookahead_diams(child, parent_dist+1, max_dist, paths, stack)