"""
Cell reduction helper functions.


@author Lucas Koelman
@date	03-11-2016
@note	must be run from script directory or .hoc files not found

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import re

import logging
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module

import neuron
from neuron import h

# Load NEURON function libraries
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

################################################################################
# Electrotonic structure
################################################################################

def lambda_DC(sec, gleak):
	""" Compute electrotonic length of section in units of micron [um]"""
	# Convert membrane resistance to same units as Ra
	# R_m = 1./(gleak*math.pi*sec.diam*1e-4) # r_m = R_m [Ohm*cm^2] /(pi*d) [Ohm*cm]
	R_m = 1./gleak # units [Ohm*cm^2]
	return 1e2 * math.sqrt(sec.diam*R_m/(4*sec.Ra)) # units: ([um]*[Ohm*cm^2]/(Ohm*cm))^1/2 = [um*1e2]

def lambda_AC(sec, f):
	""" Compute electrotonic length (taken from stdlib.hoc) """
	return 1e5 * math.sqrt(sec.diam/(4*math.pi*f*sec.Ra*sec.cm))

def electrotonic_length(sec, gleak, f):
	if f <= 0:
		return lambda_DC(sec, gleak)
	else:
		return lambda_AC(sec, f)

def seg_lambda(seg, gleak, f):
	""" Compute length constant of segment """
	Ra = seg.sec.Ra # Ra is section property
	if f <= 0:
		if isinstance(gleak, str):
			Rm = 1./getattr(seg, gleak)
		else:
			Rm = 1./gleak # units [Ohm*cm^2]
		return 1e2 * math.sqrt(seg.diam*Rm/(4*Ra)) # units: ([um]*[Ohm*cm^2]/(Ohm*cm))^1/2 = [um*1e2]
	else:
		return 1e5 * math.sqrt(seg.diam/(4*math.pi*f*Ra*seg.cm))

def min_nseg_hines(sec, f=100.):
	""" Minimum number of segments based on electrotonic length """
	return int(sec.L/(0.1*lambda_AC(sec, f))) + 1

def calc_min_nseg_hines(f, L, diam, Ra, cm):
	lamb_AC = 1e5 * math.sqrt(diam/(4*math.pi*f*Ra*cm))
	return int(L/(0.1*lamb_AC)) + 1

def inputresistance_inf(sec, gleak, f):
	""" Input resistance for semi-infinite cable in units of [Ohm*1e6] """
	lamb = electrotonic_length(sec, gleak, f)
	R_m = 1./gleak # units [Ohm*cm^2]
	return 1e2 * R_m/(math.pi*sec.diam*lamb) # units: [Ohm*cm^2]/[um^2] = [Ohm*1e8]

def inputresistance_sealed(sec, gleak, f):
	""" Input resistance of finite cable with sealed end in units of [Ohm*1e6] """
	x = sec.L/electrotonic_length(sec, gleak, f)
	return inputresistance_inf(sec, gleak, f) * (math.cosh(x)/math.sinh(x))

def inputresistance_leaky(sec, gleak, f, R_end):
	""" Input resistance of finite cable with leaky end in units of [Ohm*1e6]
	@param R_end	input resistance of connected cables at end of section
					in units of [Ohm*1e6]
	"""
	R_inf = inputresistance_inf(sec, gleak, f)
	x = sec.L/electrotonic_length(sec, gleak, f)
	return R_inf * (R_end/R_inf*math.cosh(x) + math.sinh(x)) / (R_end/R_inf*math.sinh(x) + math.cosh(x))

def inputresistance_tree(rootsec, f, glname):
	""" Compute input resistance to branching tree """
	childsecs = rootsec.children()
	gleak = getattr(rootsec, glname)

	# Handle leaf sections
	if not any(childsecs):
		return inputresistance_sealed(rootsec, gleak, f)

	# Calc input conductance of children
	g_end = 0.
	for childsec in childsecs:
		g_end += 1./inputresistance_tree(childsec, f, glname)
	return inputresistance_leaky(rootsec, gleak, f, 1./g_end)

def sec_path_ri(secref, store_seg_ri=False):
	""" Calculate axial path resistance from root to 0 and 1 end of each section

	@effect		calculate axial path resistance from root to 0/1 end of sections
				and set as properties pathri0/pathri1 on secref

	@return		tuple pathri0, pathri1
	"""
	rootsec = subtreeroot(secref)
	rootsec = subtreeroot(secref)
	rootparent = rootsec.parentseg()
	if rootparent is None:
		return 0.0 # if we are soma/topmost root: path length is zero

	# Get path from root node to this sections
	calc_path = h.RangeVarPlot('v')
	rootsec.push()
	calc_path.begin(0.5)
	secref.sec.push()
	calc_path.end(0.5)
	root_path = h.SectionList()
	calc_path.list(root_path) # store path in list
	h.pop_section()
	h.pop_section()

	# Store axial path resistance on section
	if store_seg_ri and not(hasattr(secref, pathri_seg)):
		secref.pathri_seg = [0.0] * secref.sec.nseg

	# Compute axial path resistances
	pathri0 = 0. # axial path resistance from root to start of target Section
	pathri1 = 0. # axial path resistance from root to end of target Section
	path_secs = list(root_path)
	path_len = len(path_secs)
	for isec, psec in enumerate(path_secs):
		arrived = bool(psec.same(secref.sec))
		for jseg, seg in enumerate(psec):
			# Axial path resistance to start of current segment
			if store_seg_ri and arrived:
				secref.pathri_seg[jseg] = pathri1
			# Axial path resistance to end of current segment
			pathri1 += seg.ri()
			# Axial path resistance to start of target section
			if isec < path_len-1:
				pathri0 = pathri1

	secref.pathri0 = pathri0
	secref.pathri1 = pathri1

	return pathri0, pathri1

def seg_path_ri(endseg, f, gleak_name):
	""" 
	Calculate axial path resistance from start of segment up to but 
	not including soma section (the topmost root section).

	@return		electrotonic path length
	"""
	secref = h.SectionRef(sec=endseg.sec)
	rootsec = subtreeroot(secref)
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
	pathri0 = 0. # axial path resistance from root sec to 0 end of this sec
	path_secs = list(root_path)
	assert(endseg.sec in path_secs)
	for i, psec in enumerate(path_secs): # walk sections
		for seg in psec:				 # walk segments
			if seg.sec.same(endseg.sec) and (seg.x == endseg.x): # reached end segment
				pathri1 = pathri0 + seg.ri()
				return pathri0, pathri1
			pathri0 += seg.ri()

	raise Exception('End segment not reached')

def sec_path_L_elec(secref, f, gleak_name):
	""" Calculate electrotonic path length up to but not including soma
		section (the topmost root section).

	ALGORITHM
	- walk each segment from root section (child of top root) to the given
	  section and sum L/lambda for each segment

	@return		tuple pathL0, pathL1
	@post		pathL0 and pathL1 are available as attributes on secref

	FIXME: in root node, start walking segments only from midpoint
	"""
	rootsec = subtreeroot(secref)
	rootparent = rootsec.parentseg()
	if rootparent is None:
		secref.pathL0 = 0.0
		secref.pathL1 = 0.0
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
	secref.pathL1 = 0.0 # path length from root sec to 1 end of this sec
	secref.pathL0 = 0.0 # path length from root sec to 0 end of this sec
	path_secs = list(root_path)
	path_len = len(path_secs)
	for i, psec in enumerate(path_secs):
		L_seg = psec.L/psec.nseg # segment length
		for seg in psec:
			lamb_seg = seg_lambda(seg, gleak_name, f)
			L_elec = L_seg/lamb_seg
			secref.pathL1 += L_elec
			if i < path_len-1:
				secref.pathL0 += L_elec

	return secref.pathL0, secref.pathL1

def seg_path_L_elec(endseg, f, gleak_name):
	""" Calculate electrotonic path length from start of segment up to but 
		not including soma section (the topmost root section).

	ALGORITHM
	- walk each segment from root section (child of top root) to the given
	  segment and sum L/lambda for each segment

	@return		electrotonic path length
	"""
	secref = h.SectionRef(sec=endseg.sec)
	rootsec = subtreeroot(secref)
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
		for seg in psec:
			# Check if we have reached our target segment
			if seg.sec.same(endseg.sec) and (seg.x == endseg.x):
				return path_L_elec
			lamb_seg = seg_lambda(seg, gleak_name, f)
			L_elec = L_seg/lamb_seg
			path_L_elec += L_elec

	raise Exception('End segment not reached')


################################################################################
# Clustering & Topology
################################################################################

class ExtSection(neuron.hclass(h.Section)):
	""" Extension of Section to allow modifying attributes """
	pass

class ExtSecRef(neuron.hclass(h.SectionRef)):
	""" Extension of SectionRef to allow modifying attributes """
	def __repr__(self):
		multiline = False
		if not multiline:
			desc = super(ExtSecRef, self).__repr__()
			printable = ['sec', 'strahlernumber', 'order']
			for ppty in printable:
				if hasattr(self, ppty):
					desc += '/{0}:{1}'.format(ppty, getattr(self, ppty))
		else:
			desc = super(ExtSecRef, self).__repr__()
			desc += '\n\t|- hocname: ' + self.sec.hoc_internal_name()
			printable = ['sec', 'strahlernumber', 'order', 'cluster_label', 
						 'secri', 'pathri0', 'pathri1', 'secSurf', 'mrgL',
						 'mrgdiam', 'mrgri', 'mrgSurf', 'visited', 'doMerge']
			for ppty in printable:
				if hasattr(self, ppty):
					desc += '\n\t|- {0}: {1}'.format(ppty, getattr(self, ppty))
		return desc

def getsecref(sec, refs):
	"""
	Look for SectionRef pointing to Section sec in enumerable of SectionRef

	@return		first SectionRef in refs with same section name as sec
	"""
	if sec is None: return None
	# Section names are unique, but alternatively use sec.same(ref.sec)
	return next((ref for ref in refs if ref.sec.name()==sec.name()), None)

def prev_seg(curseg):
	""" Get segment preceding seg: this can be on same or parent Section """
	# NOTE: cannot use seg.next() since this changed content of seg
	allseg = reversed([seg for seg in curseg.sec if round(seg.x,3) < round(curseg.x,3)])
	return next(allseg, curseg.sec.parentseg())

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
	""" Find the root section of the tree that given sections belongs to.
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

def split_section(src_sec, mechs_pars, delete_src=False):
	"""
	Split section by deleting it and adding two sections in series

	@param mechs_pars	dictionary of mechanism name -> [parameter names]
						that need to be copied
	"""
	# Create two halves
	halfA_name = src_sec.name() + "_A"
	h("create %s" % halfA_name)
	secA = getattr(h, halfA_name)

	halfB_name = src_sec.name() + "_B"
	h("create %s" % halfB_name)
	secB = getattr(h, halfB_name)

	# Copy properties
	for tar_sec in [secA, secB]:
		# Copy passive properties
		tar_sec.L = src_sec.L / 2.
		tar_sec.Ra = src_sec.Ra
		if src_sec.nseg % 2 == 0:
			tar_sec.nseg = src_sec.nseg / 2
		else:
			logger.warning("Splitting section with uneven number of segments")
			tar_sec.nseg = int(src_sec.nseg) / 2 + 1 # don't lose accuracy

		# Copy mechanisms
		for mechname in mechs_pars.keys():
			if hasattr(src_sec(0.5), mechname):
				tar_sec.insert(mechname)

		# Copy range variables
		for tar_seg in tar_sec:
			src_seg = src_sec(tar_seg.x/2.) # segment that it maps to
			tar_seg.diam = src_seg.diam
			tar_seg.cm = src_seg.cm
			for mech in mechs_pars.keys():
				for par in mechs_pars[mech]:
					prop = par+'_'+mech
					setattr(tar_seg, prop, getattr(src_seg, prop))

		# Copy ion styles
		copy_ion_styles(src_sec, tar_sec)

	# Connect A to B
	secB.connect(secA, 1, 0)

	# Connect A to parent of src_sec
	secA.connect(src_sec.parentseg().sec, src_sec.parentseg().x, 0)

	# Connect children of src_sec to B
	for childsec in src_sec.children():
		xloc = childsec.parentseg().x
		# NOTE: connecting section disconnects it from previous parent
		if xloc >= 0.5:
			childsec.connect(secB, 2*(xloc-0.5), 0)
		else:
			childsec.connect(secA, 2*xloc, 0)

	# Disconnect and delete src_sec
	src_sec.push()
	h.disconnect()
	if delete_src:
		h.delete_section()
	h.pop_section()

def dupe_secprops(src_sec, tar_sec, mechs_pars):
	""" Copy section properties """
	# Number of segments and mechanisms
	tar_sec.nseg = src_sec.nseg
	for mech in mechs_pars.keys():
		if hasattr(src_sec(0.5), mech):
			tar_sec.insert(mech)

	# Geometry and passive properties
	tar_sec.L = src_sec.L
	tar_sec.Ra = src_sec.Ra
	# tar_sec.cm = src_sec.cm

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

def copy_ion_styles(src_sec, tar_sec):
	""" Copy ion styles from source to target section """
	src_sec.push()
	ions = ['na', 'k', 'ca']
	styles = [h.ion_style(ion+'_ion') for ion in ions]
	tar_sec.push()
	for i, ion in enumerate(ions):
		style = styles[i]
		c_style = int(style) & (1+2)
		cinit = (int(style) & 4) >> 2
		e_style = (int(style) & (8+16)) >> 3
		einit = (int(style) & 32) >> 5
		eadvance = (int(style) & 64) >> 6
		h.ion_style(ion+'_ion', c_style, e_style, einit, eadvance, cinit)
	h.pop_section()
	h.pop_section()

def dupe_subtree(rootsec, mechs_pars, tree_copy):
	""" Duplicate tree of given section
	@param rootsec		root section of the subtree
	@param mechs_pars	dictionary mechanism_name -> parameter_name
	@param tree_copy	out argument: list to be filled
	"""
	# Copy current root node
	copyname = 'copyof_' + rootsec.name()
	i = 0
	while h.issection(copyname):
		if i > 1000:
			raise Exception('Too many copies of this section!')
		i += 1
		copyname = ('copy%iof' % i) + rootsec.name()
	h("create %s" % copyname)
	root_copy = getattr(h, copyname)
	dupe_secprops(rootsec, root_copy, mechs_pars)
	tree_copy.append(root_copy)

	# Copy children
	for childsec in rootsec.children():
		child_copy = dupe_subtree(childsec, mechs_pars, tree_copy)
		child_copy.connect(root_copy, childsec.parentseg().x, 0)

	return root_copy

def assign_electrotonic_length(rootref, allsecrefs, f, gleak_name, allseg=False):
	""" 
	Assign length constant (lambda) and electrotonic path length (L/lambda)
	to 0- and 1-end for each section in subtree of given section.

	@type	rootref		SectionRef
	@param	rootref		Section reference to current node

	@type	allsecrefs	list(SectionRef)
	@param	allsecrefs	references to all sections in the cell

	@post				all section references have the following attributes:
						- 'f_lambda': frequency at which length constant is computed
						- 'lambda_f': section's length constant at given frequency
						- 'pathL0': electrotonic path length to 0-end
						- 'pathL1': Electrotonic path length to 1-end
	"""
	if rootref is None:
		return

	# Compute length constant
	gleak = sum([getattr(seg, gleak_name) for seg in rootref.sec])
	gleak /= rootref.sec.nseg # average gleak of segments in section
	lambda_f = electrotonic_length(rootref.sec, gleak, f)
	rootref.lambda_f = lambda_f
	rootref.f_lambda = f

	# Compute electrotonic path length
	if allseg:
		rootref.pathL_elec = [0.0]*rootref.sec.nseg
		for i, seg in enumerate(rootref.sec):
			# visit every segment expcent 0/1 end zero-area segments
			pathL = seg_path_L_elec(seg, f, gleak_name)
			rootref.pathL_elec[i] = pathL
	pathL0, pathL1 = sec_path_L_elec(rootref, f, gleak_name)
	rootref.pathL0 = pathL0
	rootref.pathL1 = pathL1

	# Assign to children
	for childsec in rootref.sec.children():
		childref = getsecref(childsec, allsecrefs)
		assign_electrotonic_length(childref, allsecrefs, f, gleak_name, allseg=allseg)


if __name__ == '__main__':
	print("Main of reduction_tools.py: TODO: execute tests")