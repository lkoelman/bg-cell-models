"""
Cell reduction helper functions.


@author Lucas Koelman
@date   03-11-2016
@note   must be run from script directory or .hoc files not found

"""


import logging
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s @%(filename)s:%(lineno)s', level=logging.DEBUG)
logname = "redops" # __name__
logger = logging.getLogger(logname) # create logger for this module

from neuron import h

from common.nrnutil import getsecref, seg_index, same_seg
from common.treeutils import subtreeroot
from common.electrotonic import seg_lambda, sec_lambda

# Load NEURON function libraries
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

################################################################################
# Electrotonic structure
################################################################################


def sec_path_L_elec(secref, f, gleak_name):
    """ Calculate electrotonic path length up to but not including soma
        section (the topmost root section).

    ALGORITHM
    - walk each segment from root section (child of top root) to the given
      section and sum L/lambda for each segment

    @return     tuple pathLelec0, pathLelec1
    @post       pathLelec0 and pathLelec1 are available as attributes on secref

    FIXME: in root node, start walking segments only from midpoint
    """
    rootsec = subtreeroot(secref)
    rootparent = rootsec.parentseg()
    if rootparent is None:
        secref.pathLelec0 = 0.0
        secref.pathLelec1 = 0.0
        return 0.0, 0.0 # if we are soma/topmost root: path length is zero

    # Get path from soma (not including) up to and including this section
    calc_path = h.RangeVarPlot('v')
    rootsec.push()
    calc_path.begin(0.5)
    secref.sec.push()
    calc_path.end(0.5)
    root_path = h.SectionList() # SectionList structure to store path
    calc_path.list(root_path) # copy path sections to SectionList
    h.pop_section()
    h.pop_section()

    # Compute electrotonic path length
    secref.pathLelec1 = 0.0 # path length from root sec to 1 end of this sec
    secref.pathLelec0 = 0.0 # path length from root sec to 0 end of this sec
    path_secs = list(root_path)
    path_len = len(path_secs)
    for i, psec in enumerate(path_secs):
        L_seg = psec.L/psec.nseg # segment length
        for seg in psec:
            lamb_seg = seg_lambda(seg, gleak_name, f)
            L_elec = L_seg/lamb_seg
            secref.pathLelec1 += L_elec
            if i < path_len-1:
                secref.pathLelec0 += L_elec

    return secref.pathLelec0, secref.pathLelec1


def sec_path_props(secref, f, gleak_name, linearize_gating=False, init_cell=None):
    """
    Assign path properties to start and end of section, and to all internal segments

    @post   the given SectionRef will have the following attributes:

            - pathL0, pathL1    path length to start and end of section
            - pathL_seg         path length to END of each segment

            - pathri0, pathri1  axial path resistance to start and end of section
            - pathri_seg        axial path resistance to START of each segment

            - pathLelec0            electrotonic path length to START of SECTION
            - pathLelec1            electrotonic path length to END of SECTION
            
            - seg_path_Lelec0       electrotonic path length to START of each SEGMENT
            - seg_path_Lelec1       electrotonic path length to END of each SEGMENT
    """
    rootsec = subtreeroot(secref) # first child section of absolute root

    # Create attribute for storing path resistance
    secref.pathL_seg        = [0.0] * secref.sec.nseg
    secref.pathri_seg       = [0.0] * secref.sec.nseg
    secref.seg_path_Lelec0  = [0.0] * secref.sec.nseg
    secref.seg_path_Lelec1  = [0.0] * secref.sec.nseg
    secref.seg_lambda       = [0.0] * secref.sec.nseg
    # Aliases
    secref.pathL_elec = secref.seg_path_Lelec0
    
    # Initialize path length calculation
    ## Get path from root section to end of given section
    calc_path = h.RangeVarPlot('v')
    
    rootsec.push()
    calc_path.begin(0.0) # x doesn't matter since we only use sections
    
    secref.sec.push()
    calc_path.end(1.0) # x doesn't matter (idem)
    
    root_path = h.SectionList() # SectionList structure to store path
    calc_path.list(root_path) # copy path sections to SectionList
    
    h.pop_section()
    h.pop_section()


    # Initialize variables
    path_L = 0.0
    path_L_elec = 0.0
    path_ri = 0.0

    # Compute path length
    path_secs = list(root_path)
    path_len = len(path_secs)
    
    for isec, psec in enumerate(path_secs):
        arrived_sec = (isec==path_len-1) # alternative: use sec.same()
        
        # Start at 0-end of section
        for j_seg, seg in enumerate(psec):
            
            # store path length to start of segment
            if arrived_sec:
                secref.pathri_seg[j_seg] = path_ri
                secref.seg_path_Lelec0[j_seg] = path_L_elec
                if j_seg==0:
                    secref.pathL0 = path_L
                    secref.pathLelec0 = path_L_elec
                    secref.pathri0 = path_ri

            # Update path lengths to end
            seg_L = psec.L/psec.nseg
            seg_lamb = seg_lambda(seg, gleak_name, f)
            seg_L_elec = seg_L/seg_lamb
            path_L += seg_L
            path_L_elec += seg_L_elec
            path_ri += seg.ri()

            # store path length to end of segment
            if arrived_sec:
                secref.pathL_seg[j_seg] = path_L
                secref.seg_path_Lelec1[j_seg] = path_L_elec
                secref.seg_lambda[j_seg] = seg_lamb
                if (j_seg==psec.nseg-1):
                    secref.pathL1 = path_L
                    secref.pathLelec1 = path_L_elec
                    secref.pathri1 = path_ri


def seg_path_L_elec(endseg, f, gleak_name):
    """ 
    Calculate electrotonic path length from after root section up to
    start of given segment.

    ALGORITHM
    - walk each segment from root section (child of top root) to the given
      segment and sum L/lambda for each segment

    @return     electrotonic path length
    """
    secref = h.SectionRef(sec=endseg.sec)
    rootsec = subtreeroot(secref)
    j_endseg = seg_index(endseg)
    rootparent = rootsec.parentseg()
    if rootparent is None:
        return 0.0 # if we are soma/topmost root: path length is zero

    # Get path from soma (not including) up to and including this section
    calc_path = h.RangeVarPlot('v')
    rootsec.push()
    calc_path.begin(0.5)
    secref.sec.push()
    calc_path.end(0.5)
    root_path = h.SectionList() # SectionList structure to store path
    calc_path.list(root_path) # copy path sections to SectionList
    h.pop_section()
    h.pop_section()

    # Compute electrotonic path length
    path_L_elec = 0.0
    path_secs = list(root_path)
    assert(endseg.sec in path_secs)
    for i, psec in enumerate(path_secs):
        L_seg = psec.L/psec.nseg # segment length
        for j_seg, seg in enumerate(psec):
            # Check if we have reached our target segment
            if seg.sec.same(endseg.sec) and j_seg==j_endseg:
                assert same_seg(seg, endseg)
                return path_L_elec
            lamb_seg = seg_lambda(seg, gleak_name, f)
            L_elec = L_seg/lamb_seg
            path_L_elec += L_elec

    raise Exception('End segment not reached')


def seg_path_L(endseg, to_end):
    """
    Calculate path length from center of root section to given segment

    @param to_end   if True, return distance to end of segment, else
                    return distance to start of segment
    """
    secref = h.SectionRef(sec=endseg.sec)
    rootsec = subtreeroot(secref)
    j_endseg = seg_index(endseg)
    rootparent = rootsec.parentseg()
    if rootparent is None:
        return 0.0 # if we are soma/topmost root: path length is zero
    
    # Get path from root section to endseg
    calc_path = h.RangeVarPlot('v')
    rootsec.push()
    calc_path.begin(0.0) # x doesn't matter since we only use path sections
    endseg.sec.push()
    calc_path.end(endseg.x)
    root_path = h.SectionList() # SectionList structure to store path
    calc_path.list(root_path) # copy path sections to SectionList
    h.pop_section()
    h.pop_section()

    # Compute path length
    path_secs = list(root_path)
    path_L = 0.0
    for isec, psec in enumerate(path_secs):
        arrived = bool(psec.same(endseg.sec))
        seg_L = psec.L/psec.nseg 
        for j_seg, seg in enumerate(psec):
            if arrived and j_seg==j_endseg:
                # assert same_seg(seg, endseg)
                if to_end:
                    return path_L + seg_L
                else:
                    return path_L
            path_L += seg_L


class SecProps(object):
    """
    Equivalent properties of merged sections

    NOTE: this is the 'Bunch' recipe from the python cookbook
          al alternative would be `myobj = type('Bunch', (object,), {})()`

    ATTRIBUTES
    ----------

        L           <float> section length
        Ra          <float> axial resistance
        nseg        <int> number of segments
        seg         <list(dict)> RANGE properties for each segment, including
                    'diam' and 'cm'
        children    <list(EqProps)> children attached to 1-end of section
    
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

EqProps = SecProps # alias


def get_sec_properties(src_sec, mechs_pars):
    """
    Get RANGE properties for each segment in Section.

    @return     list(dict()) of length nseg containing property names and 
                values for each segment in Section
    """
    props = []
    for seg in src_sec:
        pseg = {}
        pseg['diam'] = seg.diam
        pseg['cm'] = seg.cm
        for mech in mechs_pars.keys():
            for par in mechs_pars[mech]:
                prop = par+'_'+mech
                pseg[prop] = getattr(seg, prop)
        props.append(pseg)
    return props


def get_sec_props_ref(
        secref, 
        mechs_pars, 
        seg_assigned=None, 
        sec_assigned=None):
    """
    Get both RANGE properties and assigned properties for each segment.

    The properties are stored on an struct-like object named EqProps.


    @pre        all listed properties in seg_assigned and sec_assigned must
                be already computed and stored on the SectionRef object


    @param  secref      SectionRef object to Section for which you want to
                        save the properties

    @param  seg_assigned    list of SectionRef attributes you wish to save
                            (properties per segment)

    @param  sec_assigned    list of SectionRef attributes you wish to save
                            (properties of entire Section)


    @return     EqProps object with requested properties stored as attributes
    """
    if seg_assigned is None:
        seg_assigned = []
    if sec_assigned is None:
        sec_assigned = []

    # Store section properties (non-Range)
    sec_props = EqProps(
                    L=secref.sec.L, 
                    Ra=secref.sec.Ra, 
                    nseg=secref.sec.nseg)
    
    for prop in sec_assigned:
        setattr(sec_props, prop, getattr(secref, prop))

    # Initialize segment RANGE properties
    sec_props.seg = [dict() for i in xrange(secref.sec.nseg)]
    bprops = [par+'_'+mech for mech,pars in mechs_pars.iteritems() for par in pars]
    
    # Store segment RANGE properties
    for j_seg, seg in enumerate(secref.sec):
        
        # Store built-in properties
        for prop in bprops:
            if hasattr(seg, prop):
                sec_props.seg[j_seg][prop] = getattr(seg, prop)
        
        # Store self-assigned properties (stored on SectionRef)
        for prop in seg_assigned:
            sec_props.seg[j_seg][prop] = getattr(secref, prop)[j_seg]
    
    return sec_props

# Aliases
get_sec_props_obj = get_sec_props_ref


def get_sec_props(sec, mechs_pars):
    """
    Get Section properties and save in new SecProps object.

    @param  sec             <nrn.Section> NEURON section

    @param  mechs_pars      dict mechanism_name -> [parameter_names] with segment 
                            properties to save. To include diam and cm, use a
                            key-value pair {'' : ['diam', 'cm']}

    @return                 <SecProps> object
    """
    # Store section properties (non-Range)
    sec_props = SecProps(
                    L=sec.L, 
                    Ra=sec.Ra,
                    nseg=sec.nseg)

    # Initialize dicts with RANGE properties
    sec_props.seg = [dict() for i in xrange(sec.nseg)]
    parnames = [par+'_'+mech for mech, pars in mechs_pars.iteritems() for par in pars]
    
    # Store segment RANGE properties
    for j_seg, seg in enumerate(sec):
        for pname in parnames:
            sec_props.seg[j_seg][pname] = getattr(seg, pname)

    return sec_props


def merge_sec_properties(src_props, tar_sec, mechs_pars, check_uniform=True):
    """
    Merge section properties from multiple SecProps objects into target Section.

    WARNING: properties are queried at center segment and assigned to the
    target section like sec.ppty = val (i.e. not on a per-segment basis)

    @param  src_props   list(SecProps), each with attribute 'seg' containing one dict
                        of segment attributes per segment

    @param  mechs_pars  dict mechanism_name -> [parameter_names] with segment 
                        properties (RANGE properties) to copy
    """
    

    # keep track of assigned parameters
    assigned_params = {par+'_'+mech: None for mech,pars in mechs_pars.iteritems() 
                                            for par in pars}

    for src_sec in src_props:
        nseg = src_sec.nseg
        segs = src_sec.seg
        for mechname, parnames in mechs_pars.iteritems():
            
            # Check if any segment in source has the mechanism
            if any((p.startswith(mechname) for i in range(nseg)
                    for p in segs[i].keys())):
                tar_sec.insert(mechname)
            else:
                continue
            
            # Copy parameter values
            for pname in parnames:
                fullpname = pname+'_'+mechname

                # Check that parameter values are uniform in section
                mid_val = segs[int(nseg)/2].get(fullpname, None)
                
                if check_uniform and any(
                    (mid_val!=segs[i].get(fullpname, None) for i in range(nseg))):
                    raise Exception("Parameter {} is not uniform in source section {}."
                                    "Cannot assign non-uniform value to Section.".format(
                                        fullpname, src_sec))

                # Check that parameter value is same as in other source sections
                prev_val = assigned_params[fullpname]
                
                if check_uniform and (prev_val is not None) and mid_val!=prev_val:
                    raise Exception("Parameter {} is not uniform between source sections."
                                    "Cannot assign non-uniform value to Section.".format(
                                        fullpname))

                # Copy value to entire section
                setattr(tar_sec, fullpname, mid_val)
                assigned_params[fullpname] = mid_val


def copy_sec_properties(src_sec, tar_sec, mechs_pars):
    """
    Copy section properties from source to target Section
    """
    # Number of segments and mechanisms
    tar_sec.nseg = src_sec.nseg
    for mech in mechs_pars.keys():
        if hasattr(src_sec(0.5), mech):
            tar_sec.insert(mech)

    # Geometry and passive properties
    tar_sec.L = src_sec.L
    tar_sec.Ra = src_sec.Ra

    # copy RANGE properties
    for seg in src_sec:
        tar_sec(seg.x).diam = seg.diam # diameter
        tar_sec(seg.x).cm = seg.cm # capacitance
        for mech in mechs_pars.keys():
            for par in mechs_pars[mech]:
                prop = par+'_'+mech
                setattr(tar_sec(seg.x), prop, getattr(seg, prop))

    # ion styles
    copy_ion_styles(src_sec, tar_sec)


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
    if ions is None:
        ions = ['na', 'k', 'ca']

    # Get ion style for each ion species
    src_sec.push()
    styles = dict(((ion, h.ion_style(ion+'_ion')) for ion in ions))
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


def save_tree_properties(node_sec, mechs_pars):
    """
    Save properties of all Sections in subtree of given Section

    @param      mechs_pars      dict mechanism_name -> [parameter_names] with segment 
                                properties to save

    @param      save_ion_styles list(str) containing ion names: ion styles you want to save
    
    @return     EqProps tree with requested properties stored as attributes
    """
    # Create SecProps object for current node
    sec_props = get_sec_props(node_sec, mechs_pars)

    # Call for each child and add to children
    sec_props.children = []
    for child_sec in node_sec.children(): # Depth-first tree traversal
        sec_props.children.append(save_tree_properties(child_sec, mechs_pars))

    return sec_props


def save_tree_properties_ref(
        node_ref,
        all_refs,
        mechs_pars,
        sec_assigned_props=None,
        seg_assigned_props=None,
        save_ion_styles=None
    ):
    """
    Save properties of all Sections in subtree of given SectionRef

    @param      mechs_pars      dict mechanism_name -> [parameter_names] with segment 
                                properties to save

    @param      save_ion_styles list(str) containing ion names: ion styles you want to save
    
    @return     EqProps tree with requested properties stored as attributes
    """
    # Create SecProps object for current node
    sec_props = get_sec_props_ref(
                    node_ref,
                    mechs_pars,
                    sec_assigned=sec_assigned_props,
                    seg_assigned=seg_assigned_props)

    if save_ion_styles is not None:
        sec_props.ion_styles = get_ion_styles(node_ref.sec, ions=save_ion_styles)

    # Call for each child and add to children
    sec_props.parent = None
    sec_props.children = []
    for child_sec in node_ref.sec.children(): # Depth-first tree traversal
        child_ref = getsecref(child_sec, all_refs)
        child_props = save_tree_properties_ref(
                        child_ref, all_refs, mechs_pars, 
                        sec_assigned_props=sec_assigned_props,
                        seg_assigned_props=seg_assigned_props,
                        save_ion_styles=save_ion_styles)
        child_props.parent = sec_props
        sec_props.children.append(child_props)

    return sec_props


def subtree_assign_attributes(noderef, allsecrefs, attr_dict):
    """
    Assign attributes to all SectionRefs in subtree of given section

    @param attr_dict    dictionary of key-value pairs (attribute_name, attribute_value)
    """
    # Assign current node
    for aname, aval in attr_dict.iteritems():
        setattr(noderef, aname, aval)

    childsecs = noderef.sec.children()
    childrefs = [getsecref(sec, allsecrefs) for sec in childsecs]
    for childref in childrefs:
        subtree_assign_attributes(childref, allsecrefs, attr_dict)


def subtree_assign_gids_dfs(node_ref, all_refs, parent_id=0):
    """
    Label nodes with int identifier by doing depth-first tree traversal
    and increment the id upon each node visit.
    """
    if node_ref is None:
        return

    highest = node_ref.gid = parent_id + 1

    childsecs = node_ref.sec.children()
    childrefs = [getsecref(sec, all_refs) for sec in childsecs]
    for childref in childrefs:
        highest = subtree_assign_gids_dfs(childref, all_refs, highest)

    return highest


def dfs_iter_tree_recursive(node):
    """
    Return generator that does depth-first tree traversal starting at given node.

    @note   makes new generator for each child, not as elegant
    """
    yield node

    for child_node in getattr(node, 'children', []):
        for cn in dfs_iter_tree_recursive(child_node):
            yield cn


def dfs_iter_tree_stack(start_node):
    """
    Return generator that does depth-first tree traversal starting at given node.

    @note   non-recusrive, avoids making a new generator per descent
    """
    stack = [start_node]
    while stack:
        node = stack.pop()
        yield node
        for child in getattr(node, 'children', []):
            stack.append(child)


def find_secprops(node, filter_fun, find_all=True):
    """
    Find SecProps object in tree satisfying given filter function

    @param  find_all    True if all nodes satisfying filter function should
                        be returned

    @return             list(SecProps) that match the filter function (may be
                        empty)
    """
    nodes_gen = dfs_iter_tree_stack(node)

    if find_all:
        return list(filter(filter_fun, nodes_gen))
    else:
        try:
            match = next((n for n in nodes_gen if filter_fun(n)))
            return [match]
        except StopIteration:
            return []


def find_roots_at_level(level, root_ref, all_refs):
    """
    Find all root Sections art a particular level.

    @param  level       (int) level of branch points to return

    @pre                ref.level and ref.end_branchpoint attributes have been set, 
                        e.g. via a call to cluster.assign_topology_attrs().
    """
    return [ref for ref in all_refs if (ref.level==level and ref.end_branchpoint)]





if __name__ == '__main__':
    print("__main__ @ redutils.py")
