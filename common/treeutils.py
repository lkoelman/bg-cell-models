"""
Utilities for dealing with NEURON cell models

@author Lucas Koelman
"""

import re

import neuron
from neuron import h


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

	@post	Section will be available as attribute on global Hoc object (neuron.h).
			This will ensure that the Section is not destroyed, even though
			no Python reference to it exists.

	@return			tuple(Section, SectionRef)
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

	@return		first SectionRef in refs with same section name as sec
	"""
	if sec is None: return None
	# Section names are unique, but alternatively use sec.same(ref.sec)
	return next((ref for ref in refs if (ref.exists() and ref.sec.name()==sec.name())), None)


def contains_sec(seclist, sec):
	"""
	Check if enumerable contains given section
	"""
	return any([sec_b.same(sec) for sec_b in seclist])


def prev_seg(curseg):
	"""
	Get segment preceding seg: this can be on same or parent Section
	"""
	# NOTE: cannot use seg.next() since this changed content of seg
	allseg = reversed([seg for seg in curseg.sec if seg_index(seg) < seg_index(curseg)])
	return next(allseg, curseg.sec.parentseg())


def next_segs(curseg):
	"""
	Get child segments of given segment
	"""
	cursec = curseg.sec
	i_rootseg = seg_index(curseg)
	child_segs = []

	# find next segments
	if i_rootseg < cursec.nseg-1: # Case 1: not end segment
		child_segs.append(next((seg for seg in cursec if seg_index(seg)>i_rootseg)))
	else: # Case 2: end segment
		child_segs = [next((seg for seg in sec)) for sec in cursec.children()]
	
	return child_segs


def next_segs_dx(curseg, dx):
	"""
	Get next segments (in direction 0 to 1 end) with discretization step x
	"""
	x = curseg.x

	if x + dx <= 1.0:
		return [curseg.sec(x + dx)]
	else:
		xb = x + dx - 1.0
		return [sec(xb) for sec in curseg.sec.children()]


def next_segs_dL(curseg, dL):
	"""
	Get next segments (in direction 0 to 1 end) with step size dL [um]
	"""
	aL = curseg.x * curseg.sec.L
	bL = aL + dL

	if bL <= curseg.sec.L:
		return [curseg.sec(bL/curseg.sec.L)]
	else:
		sL = bL - curseg.sec.L
		return [sec(sL/sec.L) for sec in curseg.sec.children()]


def seg_at_index(sec, iseg):
	""" Get the i-th segment of Section """
	xmid = (2.*(iseg+1)-1.)/(2.*sec.nseg) # See NEURON book p. 8
	# return next(seg for i,seg in enumerate(sec) if i==iseg)
	return sec(xmid)


def seg_index(tar_seg):
	""" Get index of given segment on Section """
	# return next(i for i,seg in enumerate(tar_seg.sec) if seg.x==tar_seg.x) # DOES NOT WORK if you use a seg object obtained using sec(my_x)
	seg_dx = 1.0/tar_seg.sec.nseg
	seg_id = int(tar_seg.x/seg_dx) # same as tar_seg.x // seg_dx
	return min(seg_id, tar_seg.sec.nseg-1)


def interp_seg(seg, a, b):
	"""
	Interpolate values [a,b] at segment boundaries using segment's x-loc
	"""
	seg_dx = 1.0/seg.sec.nseg
	iseg = min(int(seg.x/seg_dx), seg.sec.nseg-1)
	x_a = iseg * seg_dx
	return a + (seg.x - x_a)/seg_dx * (b-a)


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


def sameparent(secrefA, secrefB):
	""" Check if sections have same parent section """
	if not (secrefA.has_parent() and secrefB.has_parent()):
		return False
	apar = secrefA.parent # changes CAS
	bpar = secrefB.parent # changes CAS
	h.pop_section()
	h.pop_section()
	return apar.same(bpar)


def subtreeroot(secref):
	"""
	Find the root section of the tree that given sections belongs to.
	I.e. the first section after the root of the entire cell.
	"""
	# Get root section of tree
	orig = secref.root # changes the cas
	h.pop_section()

	for root in orig.children():
		# Get subtree of the current root
		roottree = h.SectionList()
		
		# Fill SectionList with subtree of CAS
		root.push()
		roottree.subtree()
		h.pop_section()

		# Check if given section in in subtree
		if secref.sec in roottree:
			return root
	
	return orig


def subtree_secs(rootsec):
	"""
	Get all Sections in subtree of but not including rootsec.
	"""
	tree_secs = h.SectionList()
	rootsec.push()
	tree_secs.subtree() # includes rootsec itself
	h.pop_section()

	return [sec for sec in tree_secs if not rootsec.same(sec)]


def wholetree_secs(sec):
	"""
	Get all Sections in the same cell (i.e. that have a path to the given section)
	"""
	tree_secs = h.SectionList()

	sec.push()
	tree_secs.wholetree()
	h.pop_section()

	return list(tree_secs)


def get_mod_name(hobj):
	"""
	Get NEURON mechanism name of given synapse object

	@param	hobj		HocObject: synapse POINT_PROCESS
	"""
	match_mod = re.search(r'^[a-zA-Z0-9]+', hobj.hname())
	modname = match_mod.group()
	return modname


seg_builtin_attrs = ['area', 'cm', 'diam', 'hh', 'na_ion', 'k_ion', 'ca_ion', 'next', 'node_index', 'point_processes', 'ri', 'sec', 'v', 'x']

pp_builtin_attrs = ['allsec', 'baseattr', 'cas', 'get_loc', 'has_loc', 'loc', 'hname', 'hocobjptr', 'next', 'ref', 'same', 'setpointer', 'Section']


def print_pp_info(cell_sec=None, mechs_params=None):
	"""
	Print information about all POINT_PROCESS mechanisms.

	WARNING: this doesn't seem to work in Python ?!
	"""
	if cell_sec:
		all_cell_secs = wholetree_secs(cell_sec)
	else:
		all_cell_secs = [sec for sec in h.allsec()]
	
	for sec in all_cell_secs:
		for seg in sec:
			for pp in seg.point_processes():

				print '\nInfo for point process {} @ {}'.format(pp, seg)
				
				mech_name = get_mod_name(pp)
				if mechs_params and mech_name in mechs_params:
					param_names = mechs_params[mech_name]
				else:
					param_names = [attr for attr in dir(pp) if (not attr.startswith('__') and (attr not in pp_builtin_attrs))]

				for param_name in param_names:
					print '{} : {}'.format(param_name, getattr(pp, param_name, 'not found'))


