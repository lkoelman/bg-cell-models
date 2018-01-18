"""
Utilities for dealing with NEURON

@author Lucas Koelman
"""

import re

import neuron
h = neuron.h


class ExtSection(neuron.hclass(h.Section)):
    """ Extension of Section to allow modifying attributes """
    pass


class ExtSecRef(neuron.hclass(h.SectionRef)):
    """ Extension of SectionRef to allow modifying attributes """
    pass


# Create equivalent section
def create_hoc_section(secname):
    """
    Create Section with given name on global Hoc object

    @post   Section will be available as attribute on global Hoc object (neuron.h).
            This will ensure that the Section is not destroyed, even though
            no Python reference to it exists.

    @return         tuple(Section, SectionRef)
    """

    if secname in [sec.name() for sec in h.allsec()]:
        raise Exception('Section named {} already exists'.format(secname))
    
    created = h("create %s" % secname)
    if created != 1:
        raise Exception("Could not create section with name '{}'".format(secname))
    
    eqsec = getattr(h, secname)
    eqref = ExtSecRef(sec=eqsec)
    return eqsec, eqref


def getsecref(sec, refs):
    """
    Look for SectionRef pointing to Section sec in enumerable of SectionRef

    @return     <SectionRef/NoneType> 
                first SectionRef in refs with same section name as sec
    """
    if sec is None: return None
    # Section names are unique, but alternatively use sec.same(ref.sec)
    return next((ref for ref in refs if (ref.exists() and ref.sec.name()==sec.name())), None)


def contains_sec(seclist, sec):
    """
    Check if enumerable contains given section
    """
    return any([sec_b.same(sec) for sec_b in seclist])


def get_mod_name(hobj):
    """
    Get NEURON mechanism name of given synapse object

    @param  hobj        HocObject: synapse POINT_PROCESS
    """
    match_mod = re.search(r'^[a-zA-Z0-9]+', hobj.hname())
    modname = match_mod.group()
    return modname


def seg_index(tar_seg):
    """ Get index of given segment on Section """
    # return next(i for i,seg in enumerate(tar_seg.sec) if seg.x==tar_seg.x) # DOES NOT WORK if you use a seg object obtained using sec(my_x)
    seg_dx = 1.0/tar_seg.sec.nseg
    seg_id = int(tar_seg.x/seg_dx) # same as tar_seg.x // seg_dx
    return min(seg_id, tar_seg.sec.nseg-1)


def seg_at_index(sec, iseg):
    """
    Get the i-th segment of Section

    @return     nrn.Segment with x-value equal to segment midpoint
    """
    xmid = (2.*(iseg+1)-1.)/(2.*sec.nseg) # See NEURON book p. 8
    # return next(seg for i,seg in enumerate(sec) if i==iseg)
    return sec(xmid)


def seg_xmid(seg):
    """
    x-value at segment midpoint
    """
    nseg = seg.sec.nseg
    iseg = min(int(seg.x * nseg), nseg-1)
    xmid = (2.*(iseg+1)-1.)/(2.*nseg) # See NEURON book p. 8
    return xmid


def seg_xleft(seg):
    """
    x-value at left boundary of segment (towards 0-end)
    """
    nseg = seg.sec.nseg
    iseg = min(int(seg.x * nseg), nseg-1)
    return (1.0/nseg) * iseg

seg_xmin = seg_xleft


def seg_xright(seg, inside=False):
    """
    x-value at right boundary of segment (towards 1-end)

    @param  inside: bool
            if True: x-location will be inside same segment, i.e. not true
            boundary since this will yield the next segment if used for 
            addressing the Section
    """
    nseg = seg.sec.nseg
    iseg = min(int(seg.x * nseg), nseg-1)
    x_max = (1.0/nseg) * (iseg + 1)
    
    if inside:
        return x_max - 1e-12
    else:
        return x_max

seg_xmax = seg_xright


def same_seg(seg_a, seg_b):
    """
    Check whether both segments are in same Section and their x location
    maps to the same segment.
    """
    if not(seg_a.sec.same(seg_b.sec)):
        return False
    seg_dx = 1.0/seg_a.sec.nseg
    # check if x locs map to same section
    return int(seg_a.x/seg_dx) == int(seg_b.x/seg_dx)


def get_range_var(seg, varname, default=0.0):
    """
    Get RANGE variable at segment regardless whether the mechanism exists
    or not.
    """
    try:
        return getattr(seg, varname, default)
    except NameError: # mechanisms is known but not inserted
        return default