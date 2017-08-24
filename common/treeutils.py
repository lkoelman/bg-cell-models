"""
Utilities for dealing with NEURON cell models

@author Lucas Koelman
"""

import neuron
from neuron import h

class ExtSection(neuron.hclass(h.Section)):
	""" Extension of Section to allow modifying attributes """
	pass

class ExtSecRef(neuron.hclass(h.SectionRef)):
	""" Extension of SectionRef to allow modifying attributes """
	pass

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
	""" Get segment preceding seg: this can be on same or parent Section """
	# NOTE: cannot use seg.next() since this changed content of seg
	allseg = reversed([seg for seg in curseg.sec if seg_index(seg) < seg_index(curseg)])
	return next(allseg, curseg.sec.parentseg())

def next_segs(curseg):
	""" Get child segments of given segment """
	cursec = curseg.sec
	i_rootseg = seg_index(curseg)
	child_segs = []

	# find next segments
	if i_rootseg < cursec.nseg-1: # Case 1: not end segment
		child_segs.append(next((seg for seg in cursec if seg_index(seg)>i_rootseg)))
	else: # Case 2: end segment
		child_segs = [next((seg for seg in sec)) for sec in cursec.children()]
	return child_segs

def seg_at_index(sec, iseg):
	""" Get the i-th segment of Section """
	xmid = (2.*(iseg+1)-1.)/(2.*sec.nseg) # See NEURON book p. 8
	# return next(seg for i,seg in enumerate(sec) if i==iseg)
	return sec(xmid)

def seg_index(tar_seg):
	""" Get index of given segment on Section """
	# return next(i for i,seg in enumerate(tar_seg.sec) if seg.x==tar_seg.x) # DOES NOT WORK if you use a seg object obtained using sec(my_x)
	seg_xwidth = 1.0/tar_seg.sec.nseg
	seg_id = int(tar_seg.x/seg_xwidth)
	return min(seg_id, tar_seg.sec.nseg-1)

def same_seg(seg_a, seg_b):
	"""
	Check whether both segments are in same Section and their x location
	maps to the same segment.
	"""
	if not(seg_a.sec.same(seg_b.sec)):
		return False
	seg_xwidth = 1.0/seg_a.sec.nseg
	# check if x locs map to same section
	return int(seg_a.x/seg_xwidth) == int(seg_b.x/seg_xwidth)

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
	orig = secref.root # changes the cas
	h.pop_section()
	for root in orig.children():
		# Get subtree of the current root
		roottree = h.SectionList()
		root.push()
		roottree.subtree()
		h.pop_section()
		# Check if given section in in subtree
		if secref.sec in roottree:
			return root
	return orig