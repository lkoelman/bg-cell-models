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
    if hasattr(hobj, 'htype'):
        hoc_name = hobj.htype.hname() # for wrapped HocObject, mechanism name is in htype attribute
    else:
        hoc_name = hobj.hname()
    match_mod = re.search(r'^[a-zA-Z0-9]+', hoc_name)
    modname = match_mod.group()
    return modname


def seg_index(tar_seg):
    """
    Get index of given segment on Section
    """
    seg_dx = 1.0/tar_seg.sec.nseg
    seg_id = int(tar_seg.x/seg_dx) # same as tar_seg.x // seg_dx
    return min(seg_id, tar_seg.sec.nseg-1)

    # NOTE: == operator compares actual segments
    # if tar_seg.x == 0.0: # start node
    #     return 0
    # elif tar_seg.x == 1.0: # end node
    #     return tar_seg.sec.nseg-1
    # else: # internal node
    #     return next(i for i,seg in enumerate(tar_seg.sec) if seg==tar_seg)
    

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
    iseg = seg_index(seg)
    xmid = (2.*(iseg+1)-1.)/(2.*nseg) # See NEURON book p. 8
    return xmid


def seg_xmin(seg, side=None):
    """
    x-value at left boundary of segment (towards 0-end)

    @param  side : str
            Relative location of return x-value to the exact segment boundary
                - 'inside' : inside given segment
                - 'outside': in previous segment or 0-end node
                - 'boundary' or None: exactly on segment boundary, no guarantee 
                   whether this is inside or outside given segment
    """
    nseg = seg.sec.nseg
    iseg = seg_index(seg)
    x_lo = (1.0/nseg) * iseg

    if side=='inside':
        x_lo += 1e-12
    elif side=='outside':
        x_lo -= 1e-12
    elif (side=='boundary') or (side is None):
        return x_lo
    else:
        raise ValueError(side)

    return max(0.0, x_lo)

seg_xleft = seg_xmin


def seg_xmax(seg, side=None):
    """
    x-value at right boundary of segment (towards 1-end)

    @param  side : str
            Relative location of return x-value to the exact segment boundary
                - 'inside' : inside given segment
                - 'outside': in previous segment or 0-end node
                - 'boundary' or None: exactly on segment boundary, no guarantee 
                   whether this is inside or outside given segment
    """
    nseg = seg.sec.nseg
    iseg = seg_index(seg)
    x_hi = (1.0/nseg) * (iseg + 1)

    if side=='inside':
        x_hi -= 1e-12
    elif side=='outside':
        x_hi += 1e-12
    elif (side=='boundary') or (side is None):
        return x_hi
    else:
        raise ValueError(side)

    return min(1.0, x_hi)

seg_xright = seg_xmax


def same_seg(seg_a, seg_b):
    """
    Check whether both segments are in same Section and their x location
    maps to the same segment.
    """
    if not(seg_a.sec.same(seg_b.sec)):
        return False
    # check if x locs map to same section
    return seg_index(seg_a) == seg_index(seg_b)


def get_range_var(seg, varname, default=0.0):
    """
    Get RANGE variable at segment regardless whether the mechanism exists
    or not.
    """
    try:
        return getattr(seg, varname, default)
    except NameError: # mechanisms is known but not inserted
        return default


def copy_ion_styles(src_sec, tar_sec, ions=None):
    """
    Copy ion styles from source to target section

    NOTE:

    oldstyle = ion_style("name_ion")

    oldstyle = int:
        int( 1*c_style + 4*cinit + 8*e_style + 32*einit + 64*eadvance )
        c_style:    0, 1, 2, 3  (2 bits)
        e_style:    0, 1, 2, 3  (2 bits)
        einit:      0, 1        (1 bits)
        eadvance:   0, 1        (1 bits)
        cinit:      0, 1        (1 bits)

    ion_style("name_ion", c_style, e_style, einit, eadvance, cinit)

    """
    if ions is None:
        ions = ['na', 'k', 'ca']

    # Get ion style for each ion species
    src_sec.push()
    styles = dict(((ion, h.ion_style(ion+'_ion')) for ion in ions))
    
    # Copy to target Section
    set_ion_styles(tar_sec, **styles)

    h.pop_section()


def get_ion_styles(src_sec, ions=None):
    """
    Get ion styles as integer for each ion.

    @return     dict({str:int}) with ion species as keys and integer
                containing bit flags signifying ion styles as values
    """
    if not isinstance(ions, list):
        ions = ['na', 'k', 'ca']

    # Get ion style for each ion species
    src_sec.push()
    styles = {}
    for ion in ions:
        pname = ion + '_ion'
        if hasattr(src_sec(0.5), pname):
            styles[ion] = h.ion_style(pname)
    h.pop_section()

    return styles


def set_ion_styles(tar_sec, **kwargs):
    """
    Set ion styles from integer containing bit flags.

    @param  kwargs      keyword arguments ion_name: style_int
    """
    # Copy to target Section
    tar_sec.push()
    for ion, style in kwargs.iteritems():

        # Decompose int into bit flags
        c_style = int(style) & (1+2)
        cinit = (int(style) & 4) >> 2
        e_style = (int(style) & (8+16)) >> 3
        einit = (int(style) & 32) >> 5
        eadvance = (int(style) & 64) >> 6

        # Copy to new section
        h.ion_style(ion+'_ion', c_style, e_style, einit, eadvance, cinit)
    
    h.pop_section()


def ion_styles_bits_to_dict(style):
    """
    Convert a float representing the styles of one ion to a dictionary
    containing values for all the style flags.

    @param  style : float
            Result of call to h.ion_style(ion) for the CAS

    @return styles: dict
            Names of styles flags and their values
    """
    # Decompose int into bit flags
    styles = {}
    styles['c_style'] = int(style) & (1+2)
    styles['cinit'] = (int(style) & 4) >> 2
    styles['e_style'] = (int(style) & (8+16)) >> 3
    styles['einit'] = (int(style) & 32) >> 5
    styles['eadvance'] = (int(style) & 64) >> 6

    return styles


def test_segment_boundaries():
    """
    Test functions related to segment x-values
    """
    from stdutil import isclose

    for nseg in range(1,11):
        sec = h.Section()
        sec.nseg = nseg
        dx = 1.0 / nseg

        # Assign true index to segment property
        sec.insert('hh')
        for i, seg in enumerate(sec):
            seg.gnabar_hh = float(i)

        # Test each segment
        for i, seg in enumerate(sec):

            # test midpoint of segment
            x_mid = seg_xmid(seg)
            assert isclose(seg.x, x_mid, abs_tol=1e-9)
            assert seg_index(seg) == i

            # test low boundary of segment
            x_min = seg_xmin(seg, side='inside')
            lo_seg = sec(x_min)
            assert seg_index(lo_seg) == i
            assert lo_seg.gnabar_hh == float(i)
            assert isclose(x_min, x_mid-dx/2, abs_tol=1e-9)

            if i != 0:
                x_min = seg_xmin(seg, side='outside')
                prev_seg = sec(x_min)
                assert seg_index(prev_seg) == i-1
                assert prev_seg.gnabar_hh == float(i-1)
                assert isclose(x_min, x_mid-dx/2, abs_tol=1e-9)

            # test high boundary of segment
            x_max = seg_xmax(seg, side='inside')
            hi_seg = sec(x_max)
            assert seg_index(hi_seg) == i
            assert hi_seg.gnabar_hh == float(i)
            assert isclose(x_max, x_mid+dx/2, abs_tol=1e-9)

            if i != nseg-1:
                x_max = seg_xmax(seg, side='outside')
                next_seg = sec(x_max)
                assert seg_index(next_seg) == i+1
                assert next_seg.gnabar_hh == float(i+1)
                assert isclose(x_max, x_mid+dx/2, abs_tol=1e-9)


    print("Test 'test_segment_boundaries' passed.")