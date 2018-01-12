"""
Utilities for dealing with NEURON cell models

@author Lucas Koelman
"""

from neuron import h

from nrnutil import seg_index
import StringIO


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


def interp_seg(seg, a, b):
	"""
	Interpolate values [a,b] at segment boundaries using segment's x-loc
	"""
	seg_dx = 1.0/seg.sec.nseg
	iseg = min(int(seg.x/seg_dx), seg.sec.nseg-1)
	x_a = iseg * seg_dx
	return a + (seg.x - x_a)/seg_dx * (b-a)



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


def wholetreeroot(secref, allsecrefs):
	"""
	Find absolute root of tree.

	@return     nrn.Section
				absolute root of tree
	"""
	root_sec = allsecrefs[0].root; h.pop_section() # pushes CAS
	return root_sec


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


def subtree_topology(sub_root, max_depth=1e9):
	"""
	Like h.topology() but for subtree of section.

	@see	nrnhome/nrn/src/nrnoc/solve.c::nrnhoc_topology()
	"""

	buff = StringIO.StringIO()

	def dashes(sec, offset, lead_char, dist=0):
		"""
		@param	dist	distance from first section (subtree root)
		"""

		orient = int(h.section_orientation(sec=sec))
		direc = "({}-{})".format(orient, 1-orient)

		# Print section in format -----| with one dash per segment
		buff.write(" " * offset)
		buff.write(lead_char)
		if dist == max_depth+1:
			# truncate end sections
			buff.write("-..")
		else:
			buff.write("-" * sec.nseg)
		
		# Print termination symbol and section description
		buff.write("|       %s%s\n" % (sec.name(), direc))

		for child_sec in reversed(sec.children()): # reversed since NEURON uses stack + pop
			# get index of segment where child connects to parent
			con_seg_idx = seg_index(child_sec.parentseg())
			# buff.write(" ")
			dashes(child_sec, con_seg_idx+offset+3, "`", dist+1)

	
	dashes(sub_root, 0, ".|")

	buff_string = buff.getvalue()
	buff.close()
	return buff_string



# seg_builtin_attrs = ['area', 'cm', 'diam', 'hh', 'na_ion', 'k_ion', 'ca_ion', 'next', 'node_index', 'point_processes', 'ri', 'sec', 'v', 'x']

# pp_builtin_attrs = ['allsec', 'baseattr', 'cas', 'get_loc', 'has_loc', 'loc', 'hname', 'hocobjptr', 'next', 'ref', 'same', 'setpointer', 'Section']


# def print_pp_info(cell_sec=None, mechs_params=None):
# 	"""
# 	Print information about all POINT_PROCESS mechanisms.

# 	WARNING: this doesn't seem to work in Python ?!
# 	"""
# 	if cell_sec:
# 		all_cell_secs = wholetree_secs(cell_sec)
# 	else:
# 		all_cell_secs = [sec for sec in h.allsec()]
	
# 	for sec in all_cell_secs:
# 		for seg in sec:
# 			for pp in seg.point_processes():

# 				print '\nInfo for point process {} @ {}'.format(pp, seg)
				
# 				mech_name = get_mod_name(pp)
# 				if mechs_params and mech_name in mechs_params:
# 					param_names = mechs_params[mech_name]
# 				else:
# 					param_names = [attr for attr in dir(pp) if (not attr.startswith('__') and (attr not in pp_builtin_attrs))]

# 				for param_name in param_names:
# 					print '{} : {}'.format(param_name, getattr(pp, param_name, 'not found'))


