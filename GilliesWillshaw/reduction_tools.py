"""
Cell reduction helper functions.


@author Lucas Koelman
@date	03-11-2016
@note	must be run from script directory or .hoc files not found

"""

import numpy as np
import matplotlib.pyplot as plt
import math

import logging
logger = logging.getLogger(__name__) # create logger for this module
logger.setLevel(logging.DEBUG)

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

def calc_path_ri(secref):
	""" Calculate axial path resistance from root to 0 and 1 end of each section

	@effect		calculate axial path resistance from root to 0/1 end of sections
				and set as properties pathri0/pathri1 on secref

	@return		tuple pathri0, pathri1
	"""
	# Get path from root node to this sections
	rootsec = treeroot(secref)
	calc_path = h.RangeVarPlot('v')
	rootsec.push()
	calc_path.begin(0.5)
	secref.sec.push()
	calc_path.end(0.5)
	root_path = h.SectionList()
	calc_path.list(root_path) # store path in list
	h.pop_section()
	h.pop_section()

	# Compute axial path resistances
	secref.pathri1 = 0 # axial path resistance from root sec to 1 end of this sec
	secref.pathri0 = 0 # axial path resistance from root sec to 0 end of this sec
	path_secs = list(root_path)
	path_len = len(path_secs)
	for i, psec in enumerate(path_secs):
		for seg in psec:
			secref.pathri1 += seg.ri()
			if i < path_len-1:
				secref.pathri0 += seg.ri()

	return secref.pathri0, secref.pathri1

def path_L_electrotonic(secref, f, gleak_name):
	""" Calculate electrotonic path length from root to 0 and 1 end of section.

	ALGORITHM
	- walk each segment from root section (e.g. soma) to the given
	  section and sum L/lambda for each segment

	@return		tuple pathL0, pathL1
	@post		pathL0 and pathL1 are available as attributes on secref

	FIXME: in root node, start walking segments only from midpoint
	"""

	# Get path from root node to this sections
	rootsec = treeroot(secref)
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
	secref.pathL1 = 0 # path length from root sec to 1 end of this sec
	secref.pathL0 = 0 # path length from root sec to 0 end of this sec
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
			printable = ['sec', 'strahlernumber', 'order', 'clusterlabel', 
						 'secri', 'pathri0', 'pathri1', 'secSurf', 'mrgL',
						 'mrgdiam', 'mrgri', 'mrgSurf', 'visited', 'doMerge']
			for ppty in printable:
				if hasattr(self, ppty):
					desc += '\n\t|- {0}: {1}'.format(ppty, getattr(self, ppty))
		return desc

def getsecref(sec, refs):
	""" Return SectionRef in refs pointing to sec with same name as sec """
	if sec is None: return None
	return next((ref for ref in refs if ref.sec.name()==sec.name()), None)

def sameparent(secrefA, secrefB):
	""" Check if sections have same parent section """
	return secrefA.has_parent() and secrefB.has_parent() and (
		secrefA.parent is secrefB.parent)

def treeroot(secref):
	""" Find the root section of the tree that given sections belongs to.
		I.e. the first section after the root of the entire cell.
	"""
	orig = secref.root
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

def clusterroot(secref, allsecrefs):
	""" Find the highest parent/ancestor of given section that is still
		in the same cluster """
	if not secref.has_parent():
		return secref
	else:
		parref = getsecref(secref.parent, allsecrefs)
		if parref is None: # case where parent is not in allsecrefs
			return secref
		elif parref.cluster_label != secref.cluster_label:
			return secref
		else:
			return clusterroot(parref)

class Cluster(object):
	""" A cluster representing merged sections """

	def __init__(self, label):
		self.label = label

	def __repr__(self):
		return '{0} ({1})'.format(self.label, super(Cluster, self).__repr__())

	# def __repr__(self):
	# 	desc = super(Cluster, self).__repr__()
	# 	printable = ['label', 'eqL', 'eqdiam', 'eqSurf', 'totsurf', 'surfSum']
	# 	for ppty in printable:
	# 		if hasattr(self, ppty):
	# 			desc += '\n\t|- {0}: {1}'.format(ppty, getattr(self, ppty))
	# 	return desc

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
	tar_sec.cm = src_sec.cm

	# copy RANGE properties
	for seg in src_sec:
		tar_sec(seg.x).diam = seg.diam # diameter
		for mech in mechs_pars.keys():
			for par in mechs_pars[mech]:
				prop = par+'_'+mech
				setattr(tar_sec(seg.x), prop, getattr(seg, prop))

	# ion styles
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

def assign_strahler_order(noderef, secrefs, par_order):
	""" Assign strahler's numbers and order/distance from root
		(infer topology from parent/child relationships)

	@type	noderef		SectionRef
	@param	noderef		Section reference to current node

	@type	secrefs		list(SectionRef)
	@param	secrefs		references to all sections in the cell

	@type	par_order	int
	@param	par_order	order of parent sections (distance in #sections from soma)
	"""
	if noderef is None: return

	# assign order
	noderef.order = par_order + 1

	# Leaf nodes get Strahler's number 1
	childsecs = noderef.sec.children()
	if not any(childsecs):
		noderef.strahlernumber = 1
		return

	# Non-leaf nodes: assign children first
	childiter = iter(childsecs)
	leftref = getsecref(next(childiter, None), secrefs)
	rightref = getsecref(next(childiter, None), secrefs)
	for childref in leftref, rightref:
		assign_strahler_order(childref, secrefs, noderef.order)

	# Assign based on children
	if rightref is None:
		# one child: inhert
		noderef.strahlernumber = leftref.strahlernumber
	elif leftref.strahlernumber != rightref.strahlernumber:
		# nonzero and unequal: max
		noderef.strahlernumber = max(leftref.strahlernumber, rightref.strahlernumber)
	else:
		# nonzero and equal: increment
		noderef.strahlernumber = leftref.strahlernumber + 1

def assign_electrotonic_length(noderef, allsecrefs, f, gleak_name):
	""" Assign length constant and electrotonic path length to 0-end and
		1-end for each section in tree, starting from the given parent section

	@type	noderef		SectionRef
	@param	noderef		Section reference to current node

	@type	allsecrefs	list(SectionRef)
	@param	allsecrefs	references to all sections in the cell

	@post				all section references have the following attributes:
						- 'f_lambda': frequency at which length constant is computed
						- 'lambda_f': section's length constant at given frequency
						- 'pathL0': electrotonic path length to 0-end
						- 'pathL1': Electrotonic path length to 1-end
	"""
	if noderef is None:
		return

	# Compute length constant
	gleak = sum([getattr(seg, gleak_name) for seg in noderef.sec])
	gleak /= noderef.sec.nseg # average gleak of segments in section
	lambda_f = electrotonic_length(noderef.sec, gleak, f)
	noderef.lambda_f = lambda_f
	noderef.f_lambda = f

	# Compute electrotonic path length
	path_L_electrotonic(noderef, f, gleak_name) # assigns attributed pathL0 and pathL1

	# Assign to children
	for childsec in noderef.sec.children():
		childref = getsecref(childsec, allsecrefs)
		assign_electrotonic_length(childref, allsecrefs, f, gleak_name)

def clusterize_electrotonic(noderef, allsecrefs, thresholds, clusterlist, labelsuffix='', parent_pos=1.0):
	""" Cluster dendritic tree based on length divided by length constant
		(i.e. length in terms of its electrotonic length) measured from
		soma section.

	@type	noderef		SectionRef
	@param	noderef		Section reference to current node

	@type	allsecrefs	list(SectionRef)
	@param	allsecrefs	references to all sections in the cell

	@type	thresholds	tuple(float)
	@param	thresholds	one or two thresholds on electrotonic path length 
						to distinguish between trunk/smooth/spiny sections

	@type	clusterlist	list(Cluster)
	@param	clusterlist	list of existing Clusters

	@pre				all section references must have their length constant
						set using assign_electrotonic_length
	"""
	if noderef is None:
		return

	# Cluster based on electronic path length at midpoint
	L_mid = noderef.pathL0 + (noderef.pathL1 - noderef.pathL0)/2.
	labels = ['trunk', 'smooth', 'spiny']
	if L_mid <= thresholds[0]:
		noderef.clusterlabel = 'trunk' + labelsuffix
	elif len(thresholds) > 1 and L_mid <= thresholds[1]:
		noderef.clusterlabel = 'smooth' + labelsuffix
	else:
		noderef.clusterlabel = 'spiny' + labelsuffix

	# Determine relation to parent cluster
	parref = getsecref(noderef.parent, allsecrefs)
	if parref.cluster_label != noderef.cluster_label:
		noderef.parent_label = parref.cluster_label
	else:
		noderef.parent_label = parref.parent_label
	noderef.parent_pos = parent_pos # by default at end of section

	# If first section belonging to this cluster, add new Cluster object
	if not any(clu.label==noderef.cluster_label for clu in clusterlist):
		newclu = Cluster(noderef.cluster_label)
		parclu = next(clu for clu in clusterlist if clu.label==noderef.parent_label)
		newclu.parent_label = noderef.parent_label
		newclu.parent_pos = noderef.parent_pos
		newclu.order = parclu.order + 1
		clusterlist.append(newclu)

	# Cluster children (iteratively)
	for childsec in noderef.sec.children():
		childref = getsecref(childsec, allsecrefs)
		clusterize_electrotonic(childref, allsecrefs, thresholds, clusterlist, labelsuffix)

def clusterize_custom(noderef, allsecrefs, clusterfun, clusterlist, labelsuffix='', parent_pos=1.0):
	""" Cluster dendritic tree based on custom criterion

	@type	noderef		SectionRef
	@param	noderef		Section reference to current node

	@type	allsecrefs	list(SectionRef)
	@param	allsecrefs	references to all sections in the cell

	@type	par_order	int
	@param	par_order	order of parent sections (distance in #sections from soma)

	@type	clusterfun	function(SectionRef) -> string
	@param	clusterfun	function mapping a SectionRef to cluster label

	@type	clusterlist	list(Cluster)
	@param	clusterlist	list of existing Clusters
	"""
	if noderef is None: return

	# Cluster section based on custom criterion
	noderef.cluster_label = clusterfun(noderef) + labelsuffix

	# Determine relation to parent cluster
	parref = getsecref(noderef.parent, allsecrefs)
	if parref.cluster_label != noderef.cluster_label:
		noderef.parent_label = parref.cluster_label
	else:
		noderef.parent_label = parref.parent_label
	noderef.parent_pos = parent_pos # by default at end of section

	# Add new Cluster object if necessary
	if not any(clu.label==noderef.cluster_label for clu in clusterlist):
		newclu = Cluster(noderef.cluster_label)
		parclu = next(clu for clu in clusterlist if clu.label==noderef.parent_label)
		newclu.parent_label = noderef.parent_label
		newclu.parent_pos = noderef.parent_pos
		newclu.order = parclu.order + 1
		clusterlist.append(newclu)

	# Cluster children (iteratively)
	childiter = iter(noderef.sec.children())
	leftref = getsecref(next(childiter, None), allsecrefs)
	rightref = getsecref(next(childiter, None), allsecrefs)
	clusterize_custom(leftref, allsecrefs, clusterfun, clusterlist, labelsuffix)
	clusterize_custom(rightref, allsecrefs, clusterfun, clusterlist, labelsuffix)

def clusterize_strahler(noderef, allsecrefs, thresholds, clusterlist, labelsuffix='', parent_pos=1.0):
	""" Cluster a tree based on strahler numbers alone

	@param noderef		any section starting from but not equal to soma

	@param parent_pos	position on parent cluster of noderef

	@effect			assign label 'trunk'/'smooth'/spiny' to sections
					based on their Strahler number
	"""
	if noderef is None: return

	# Clustering thresholds
	if thresholds is None:
		thresholds = (3,5)

	# Cluster label based on strahler's number
	if noderef.strahlernumber <= thresholds[0]: # default: <= 3
		noderef.cluster_label = 'spiny' + labelsuffix
	elif noderef.strahlernumber <= thresholds[1]: # default: <= 5
		noderef.cluster_label = 'smooth' + labelsuffix
	else:
		noderef.cluster_label = 'trunk' + labelsuffix

	# Parent cluster
	parref = getsecref(noderef.parent, allsecrefs)
	if parref.cluster_label != noderef.cluster_label:
		noderef.parent_label = parref.cluster_label
	else:
		noderef.parent_label = parref.parent_label
	noderef.parent_pos = parent_pos # by default at end of section

	# Add new cluster
	if not any(clu.label==noderef.cluster_label for clu in clusterlist):
		newclu = Cluster(noderef.cluster_label)
		parclu = next(clu for clu in clusterlist if clu.label==noderef.parent_label)
		newclu.parent_label = noderef.parent_label
		newclu.parent_pos = noderef.parent_pos
		newclu.order = parclu.order + 1
		clusterlist.append(newclu)

	# Cluster children (iteratively)
	childiter = iter(noderef.sec.children())
	leftref = getsecref(next(childiter, None), allsecrefs)
	rightref = getsecref(next(childiter, None), allsecrefs)
	clusterize_strahler(leftref, allsecrefs, thresholds, clusterlist, labelsuffix)
	clusterize_strahler(rightref, allsecrefs, thresholds, clusterlist, labelsuffix)

def clusterize_strahler_trunks(allsecrefs, thresholds=None):
	""" Cluster a tree based on strahler numbers and depending
		on the trunk section it is attached to (trunk sections
		are large dendritic sections attached to soma)

	@param allsecrefs	list of SectionRef (mutable) with first element
						a ref to root/soma section
	@param thresholds	spiny: i <= tresholds[0] (default: 3)
						smooth: i <= thresholds[1] (default: 5)
						trunk: i > thresholds[1]

	ALGORITHM: this is the algorithm used in Marasco (2012)
	- each trunk section (neighbouring soma) gets its own cluster
	- for each trunk, two clusters are created: one for smooth and
	  one for spiny sections attached to that trunk
	"""
	# Clustering thresholds
	if thresholds is None:
		thresholds = (3,5)

	# Cluster soma
	somaref = allsecrefs[0]
	somaref.cluster_label = 'soma'
	somaref.parent_label = 'soma'
	somaref.parent_pos = 0.0

	# Cluster each trunk and its subtree
	for i, trunksec in enumerate(somaref.sec.children()):
		trunkref = getsecref(trunksec, allsecrefs)
		clusterize_strahler(trunkref, allsecrefs, thresholds, labelsuffix='_'+str(i))
		logger.debug("Using suffix '_%i' for subtree of section %s", i, trunksec.name())

def cluster_topology(rootref, allsecrefs, relations):
	""" Determine cluster topology from list of clustered section references
	@param relations	a container for relations of the form (parentlabel, childlabel)
						if this is a set, the entries are guaranteed to be unique
	"""
	# Depth-first recursion of tree
	for childsec in rootref.sec.children():
		childref = getsecref(childsec, allsecrefs)
		# Add parent-child relationship
		if childref.cluster_label != rootref.cluster_label:
			relations.add((rootref.cluster_label, childref.cluster_label))
		# determine topology of subtree
		cluster_topology(childref, allsecrefs, relations)

if __name__ == '__main__':
	plotconductances(treestruct()[1], 1, loadgstruct('gcaT_CaT'), includebranches=[1,2,5])
	# plotchanneldist(0, 'gcaL_HVA')
	# dend0tree, dend1tree = treechannelstruct()
	# gtstruct = loadgeotopostruct(0)