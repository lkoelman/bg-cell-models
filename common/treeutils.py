"""
Utilities for dealing with NEURON cell models

@author Lucas Koelman
"""

from neuron import h

from nrnutil import seg_index, seg_xmin, seg_xmax
import StringIO


def prev_seg(curseg):
	"""
	Get segment preceding seg: this can be on same or parent Section
	"""
	# NOTE: cannot use seg.next() since this changed content of seg
	allseg = reversed([seg for seg in curseg.sec if seg_index(seg) < seg_index(curseg)])
	return next(allseg, curseg.sec.parentseg())


def next_segs(curseg, x_loc='mid'):
	"""
	Get child segments of given segment

	@param	x_loc: str
			
			Which x-value to associate with each returned segment:
			- 'mid': center of the segment
			- 'min': minimum x-value inside the segment
			- 'max': maximum x-value inside the segment
	"""
	cursec = curseg.sec
	i_rootseg = seg_index(curseg)
	child_segs = []

	# find next segments
	if i_rootseg < cursec.nseg-1:
		# not end segment -> return next segment of same Section
		child_segs.append(next((seg for seg in cursec if seg_index(seg)>i_rootseg)))
	else:
		# end segment -> return first segment of each child section
		child_segs = [next((seg for seg in sec)) for sec in cursec.children()]

	# Adjust x-loc if required
	if x_loc == 'min':
		child_segs = [seg.sec(seg_xmin(seg, side='inside')) for seg in child_segs]
	elif x_loc == 'max':
		child_segs = [seg.sec(seg_xmax(seg, side='inside')) for seg in child_segs]
	elif x_loc != 'mid':
		raise ValueError("Invalid value {} for argument 'x-loc'".format(x_loc))
	
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
			num_child = len(subtree_secs(sec))
			buff.write("-..       %s%s + %i children\n" % (sec.name(), direc, num_child))
			return # end of recursion
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


def check_tree_constraints(sections):
	"""
	Check unbranched cable assumption and orientation constrained.

	@return		a tuple (unbranched, oriented, branched, misoriented) of type 
				tuple(bool, bool, list(Section), list(Section)) 
				with following entries:

				bool: all Sections are unbranched
				bool: all sections are correctly oriented, i.e. the 0-end is
				 	  connected to the 1-end of the parent, except if the parent
				 	  is the root section in which case a connection to the 0-end
				 	  is permitted.
				list: all branched sections
				list: all misoriented sections
	"""
	# check both connect(child(x), parent(y))
	# parent_y: for all sections in whole tree: sec.parentseg().x must be 1.0
	# 			EXCEPT for sections connected to root
	# child_x:  for all sections in whole tree: sec.orientation() (h.section_orientation(sec=sec)) must 0.0

	is_unbranched = True
	is_oriented = True
	branched = set()
	misoriented = set()

	first_ref = h.SectionRef(sec=sections[0])
	tree_root = first_ref.root; h.pop_section() # pushes CAS

	for sec in sections:

		parent_sec = sec.parentseg().sec
		parent_y = sec.parentseg().x
		orient_parent_ok = parent_y==1.0 or (parent_y==0.0 and parent_sec.same(tree_root))
		branch_parent_ok = parent_y==1.0 or parent_y==0.0

		self_x = sec.orientation() # see h.section_orientation()
		orient_self_ok = self_x==0.0
		branch_self_ok = self_x==0.0 or self_x==1.0

		is_unbranched = is_unbranched and branch_parent_ok and branch_self_ok
		is_oriented = is_oriented and orient_parent_ok and orient_self_ok


		if not orient_parent_ok:
			misoriented.update(parent_sec)
		if not orient_self_ok:
			misoriented.update(sec)
		
		if not branch_parent_ok:
			branched.update(parent_sec)
		if not branch_self_ok:
			branched.update(sec)

	return is_unbranched, is_oriented, list(branched), list(misoriented)



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


