"""
Tools for working with extracellular layers in NEURON.

@author     Lucas Koelman
@date       26-04-2019
"""

from __future__ import division
import math

PI = math.pi
sqrt = math.sqrt

def xtra_set_transfer_impedances(seclist, impedance_lookup_func):
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