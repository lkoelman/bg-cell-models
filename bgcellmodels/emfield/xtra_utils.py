"""
Tools for working with extracellular layers in NEURON.

@author     Lucas Koelman
@date       26-04-2019
"""

from __future__ import division
import math
import logging

from neuron import h
import numpy as np

PI = math.pi
sqrt = math.sqrt
logger = logging.getLogger('emfield.xtra_utils')


def set_transfer_impedances(seclist, impedance_lookup_func):
    """
    Set transfer resistances of 'xtra' mechanism using impedance
    lookup function for segment coordinates stored in 'xtra'.

    @param  seclist : neuron.SectionList
            Sections where impedances will be set

    @param  impedance_lookup_function : callable(x, y, x)
            Function returning transfer impedance for with spatial coordinates.
    """
    for sec in seclist:
        for seg in sec:
            x, y, z = (seg.x_xtra, seg.y_xtra, seg.z_xtra)
            R_transfer = impedance_lookup_func(x, y, z)
            seg.rx_xtra = R_transfer


def set_transfer_impedances_nearest(seclist, Z_coords, Z_values,
                                    max_dist, warn_dist, min_electrode_dist,
                                    electrode_coords, Z_intersect=1e12):
    """
    Set transfer impedances using nearest neighbor interpolation, or using
    matching coordinates (set max_dist=eps).

    Uses KD-tree from scipy.spatial for fast lookups.
    
    @param  Z_coords : array_like of shape (N x 3)
            Coordinates of transfer impedances in Z_values (micron)

    @param  Z_values : arrayLike of length N
            Transfer impedances at coordinates in Z_coords
    """
    import scipy.spatial

    # Construct KD-tree using coordinates of transfer impedance values
    tree = scipy.spatial.KDTree(Z_coords)

    for sec in seclist:
        if not h.ismembrane('xtra', sec=sec):
            logger.debug("Skipping section '%s', no mechanism 'xtra' found.", sec.name())
            continue

        # Get coordinates of compartment centers
        num_samples = int(h.n3d(sec=sec))
        nseg = sec.nseg

        # Get 3D sample points for section
        xx = h.Vector([h.x3d(i, sec=sec) for i in xrange(num_samples)])
        yy = h.Vector([h.y3d(i, sec=sec) for i in xrange(num_samples)])
        zz = h.Vector([h.z3d(i, sec=sec) for i in xrange(num_samples)])

        # Length in micron from start of section to sample i
        pt_locs = h.Vector([h.arc3d(i, sec=sec) for i in xrange(num_samples)])
        L = pt_locs.x[num_samples-1]

        # Normalized location of 3D sample points (0-1)
        pt_locs.div(L)

        # Normalized locations of nodes (0-1)
        node_locs = h.Vector(nseg + 2)
        node_locs.indgen(1.0 / nseg)
        node_locs.sub(1.0 / (2 * nseg))
        node_locs.x[0] = 0.0
        node_locs.x[nseg+1] = 1.0

        # Now calculate 3D locations of nodes (segment centers + 0 + 1)
        # by interpolating 3D locations of samples
        node_xlocs = h.Vector(nseg+2)
        node_ylocs = h.Vector(nseg+2)
        node_zlocs = h.Vector(nseg+2)
        node_xlocs.interpolate(node_locs, pt_locs, xx)
        node_ylocs.interpolate(node_locs, pt_locs, yy)
        node_zlocs.interpolate(node_locs, pt_locs, zz)
        node_coords = np.array(zip(node_xlocs, node_ylocs, node_zlocs))

        # Query KD tree
        nn_dists, nn_idx = tree.query(node_coords, k=1, distance_upper_bound=max_dist)
        for i, c in enumerate(node_locs):
            node_xyz = node_coords[i]
            if np.linalg.norm(node_xyz - electrode_coords) <= min_electrode_dist:
                # Node is too close to electride
                Z_node = Z_intersect
            elif nn_dists[i] > max_dist:
                # Node is too far from nearest neighbor
                raise ValueError("No transfer impedance value found within distance "
                                 "{} of point {}".format(max_dist, node_coords[i]))
            elif nn_dists[i] > warn_dist:
                # Node is too far from nearest neighbor
                Z_node = Z_values[nn_idx[i]]
                logger.debug("Transfer impedance at point {} exceeds warning "
                             "distance of {} um".format(node_coords[i], warn_dist))
            else:
                # No issues, nearest neighbor and electrode distance OK
                Z_node = Z_values[nn_idx[i]]
            
            # Assign transfer impedance
            sec(c).rx_xtra = Z_node


def transfer_resistance_pointsource(seg, seg_coords, source_coords, rho):
    """
    Analytical transfer resistance between electrical point source
    and target coordinate in isotropic non-dispersive extraceullular medium.

    @param  rho : float
            Resistivity of extracullular medium (Ohm * cm).
    """
    x1, y1, z1 = seg_coords
    x2, y2, z2 = source_coords
    dist = sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    if dist == 0:
        dist = seg.diam / 2

    # 0.01 converts rho's cm to um and ohm to megohm
    return (rho / 4 / PI) * (1 / dist) * 0.01
