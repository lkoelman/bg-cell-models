"""
Reduce Gillies & Willshaw (2006) STN neuron model using Marasco 2012/2013
reduction method


@author Lucas Koelman
@date	28-11-2016
"""
# Python modules
from collections import OrderedDict
from operator import attrgetter
import math
PI = math.pi

# NEURON modules
import neuron
h = neuron.h

# Own modules
import reducemodel

class ExtSection(neuron.hclass(h.Section)):
	""" Extension of Section to allow modifying attributes """
	pass

class ExtSecRef(neuron.hclass(h.SectionRef)):
	""" Extension of SectionRef to allow modifying attributes """
	pass

def getsecref(sec, refs):
	""" Return SectionRef in refs pointing to sec with same name as sec """
	if sec is None: return None
	return next(ref in refs if ref.sec.name() == sec.name(), None)

def sameparent(secrefA, secrefB):
	""" Check if sections have same parent section """
	return secrefA.has_parent() and secrefB.has_parent() and 
			(secrefA.parent is secrefB.parent)

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


################################################################################
# Clustering
################################################################################

def assign_strahler_order(noderef, secrefs, par_order):
	""" Assign strahler's numbers and order/distance from root
		(infer topology from parent/child relationships)

	@noderef	SectionRef to current node
	@secrefs	list of mutable SectionRef
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

def clusterize_strahler_order(noderef, secrefs):
	""" Cluster a tree that has been assigned Strahler numbers

	@return	list of cluster names
	"""

	if noderef is None: return

	if noderef.strahlernumber <= 2:
		noderef.clusterlabel = 'spiny'
	elif noderef.strahlernumber <= 3:
		noderef.clusterlabel = 'smooth'
	else:
		noderef.clusterlabel = 'soma'

	childiter = iter(noderef.child)
	leftref = getsecref(next(childiter, None), secrefs)
	rightref = getsecref(next(childiter, None), secrefs)

	clusterize_strahler_order(leftref, secrefs)
	clusterize_strahler_order(rightref, secrefs)


################################################################################
# Merging
################################################################################

def calc_mrgRiSurf(secref):
	""" Calculate axial resistance and surface of section
		based on mutable its 'mrg' properties """
	# textbook definition of axial resistance
	mrgri = (secref.sec.Ra*secref.mrgL)/(PI*secref.mrgdiam**2/4*100)
	# cylinder surface based on merged L and diam
	mrgsurf = secref.mrgL*secref.mrgdiam*PI
	return mrgri, mrgsurf

def prep_merge(secrefs):
	""" Prepare sections for merging procedure by computing
		and storing metrics needed in the merging operation
	"""
	for secref in secrefs: # mark all as unvisited and unmerged
		sec = secref.sec

		# geometrical properties
		secref.secSurf = 0 # total area (diam/area is range var)
		secref.secri = 0 # total axial resistance from 0 to 1 end (seg.ri is between segments)
		for seg in sec:
			secref.secSurf += seg.area()
			secref.secri += seg.ri()

		# Get path from root node to this sections
		rootsec = treeroot(secref)
		calc_path = h.RangeVarPlot('v')
		rootsec.push()
		calc_path.begin(0.5)
		sec.push()
		calc_path.end(0.5)
		root_path = h.SectionList()
		calc_path.list(root_path) # store path in list
		h.pop_section()
		h.pop_section()

		# Compute axial path resistances
		secref.pathri1 = 0 # axial path resistance from root sec to 1 end of this sec
		secref.pathri0 = 0 # axial path resistance from root sec to 0 end of this sec
		path_secs = list(root_path)
		path_len = len(root_path)
		for i, psec in enumerate(path_secs):
			for seg in psec:
				secref.pathri1 += seg.ri()
				if i < path_len-1:
					secre.pathri0 += seg.ri()
		
		# mutable properties for merging
		secref.mrgL = abs(sec.L)
		secref.mrgdiamSer = sec.diam
		secref.mrgdiamPar = sec.diam**2
		secref.mrgdiam = math.sqrt(sec.Ra*sec.L*4./secref.secri/100/PI)
		secref.mrgri = secref.secri
		secref.mrgri2 = secref.secri
		secref.mrgSurf = secref.secSurf
		
		# properties for merging iteration
		secref.visited = False
		secref.merged = False

def find_mergeable(secrefs, clusterlabel):
	""" Find next mergeable sections

	@param secrefs			list of mutable SectionRef containing all sections
	@param clusterlabel		label of the cluster to merge
	@return 				tuple of Section Ref (secA, secB, secP)
	"""

	# Get sections in current cluster and sort by order
	clustersecs = [sec in secrefs if sec.clusterlabel == clusterlabel]
	
	# Sort by order (nseg from soma), descending
	secsbyorder = clustersecs.sort(key=attrgetter('order'), reverse=True)

	# Keep looping until all sections visited
	while any(not sec.visited for sec in clustersecs):
		# get next unvisited section furthest from soma
		secA = next(sec in secsbyorder if not sec.visited)
		secA.visited = True

		# get its brother section (same parent) if available
		secB = next(sec in secsbyorder if (sec is not secA) and (not sec.visited) 
					and sameparent(sec, secA), None)
		if secB is not None: secB.visited = True

		# get their parent section if in same cluster
		secP = secA.parent
		secP = getsecref(secP, clustersecs) # None if not in cluster
		if secP is not None:
			if secA: secA.merged = True
			if secB: secB.merged = True

		# Return sec refs from generator
		yield secA, secB, secP

def mergeChildWithParent(refA, refP):
	""" Merge child into parent section """
	
	# Combine properties (call to shortnms)
	Amri, Amsurf = calc_mrgRiSurf(refA)
	Pmri, Pmsurf = calc_mrgRiSurf(refP)

	# Combine (call to mergingSerialMethod())
	newL = refA.mrgL + refP.mrgL
	newRa = (refA.sec.Ra+refP.sec.Ra)/2
	newri = Amri + Pmri
	newSurf = max(Amsurf, Pmsurf)
	newdiam = math.sqrt(newRa*newL*4/newri/PI/100.)

	# Update parent properties
	refP.mrgL = newL
	refP.mrgri = newri
	refP.mrgri2 = newri
	refP.mrgSurf = newSurf
	refP.mrgdiam = newdiam
	# FIXME: unassigned/unused
	# refP.mrgdiamSer = refA.mrgdiamSer + refP.mrgdiamSer
	# refP.mrgdiamPar = refA.mrgdiamPar + refP.mrgdiamPar

def mergeYWithParent():
	""" Merge Y (two branched children) into parent """
	# NOTE: see Hoc mergingYSec()/mergingYMethod()/mergingParallelMethod()/mergingSerialMethod()

	# Current equivalent properties (call to shortnms)
	Amri, Amsurf = calc_mrgRiSurf(refA)
	Bmri, Bmsurf = calc_mrgRiSurf(refB)
	Pmri, Pmsurf = calc_mrgRiSurf(refP)

	# Combine properties of parallel sections (Call to mergingParallelMethod())
	L12 = (refA.mrgL*Amsurf+refB.mrgL*Bmsurf)/(Amsurf+Bmsurf) # lengths weighted by surfaces
	Ra12 = (refA.sec.Ra+refB.sec.Ra)/2
	diam12 = math.sqrt(refA.mrgdiam**2 + refB.mrgdiam**2)
	cross12 = PI*diam12**2/4 # cross-section area

	# Equivalent propties of merged parallel sections
	ri12 = Ra12*L12/cross12/100 # equivalent axial resistance
	surf12 = L12*diam12*PI # equivalent surface

	# Combine properties of serial sections (Call to mergingSerialMethod())
	newL = L12 + refP.mrgL
	newRa = (Ra12+refP.sec.Ra)/2
	newri = ri12 + Pmri
	newdiam = math.sqrt(newRa*newL*4/newri/PI/100.)
	# FIXME: unassigned/unused
	# newSurf = max(surf12, Pmsurf) # overwritten by newSurf in mergingYMethod()
	# newdiamSer = refA.mrgdiamSer + refP.mrgdiamSer
	# newdiamPar = refA.mrgdiamPar + refP.mrgdiamPar

	# Equivalent properties of merged serial sections
	newri2 = (Amri*Bmri/(Amri+Bmri))+Pmri
	newSurf = max(Amsurf+Bmsurf,Pmsurf)

	# Update parent properties
	refP.mrgL = newL
	refP.mrgri = newri
	refP.mrgri2 = newri2
	refP.mrgSurf = newSurf
	refP.mrgdiam = newdiam
	# FIXME: unassigned/unused
	# refP.mrgdiamSer = newdiamSer + refP.mrgdiamSer
	# refP.mrgdiamPar = newdiamPar + refP.mrgdiamPar

def reduce_marasco():
	""" Implementation of Marasco (2013) CellPurkAnalysis() & PurkReduction() """

	# Initialize Gillies model
	h.xopen("createcell.hoc")
	# Make sections accesible by both name and index + allow to add attributes
	dendLsecs = [ExtSection(sec) for sec in h.SThcell[0].dend0]
	dendRsecs = [ExtSection(sec) for sec in h.SThcell[0].dend1]
	alldendsecs = dendLsecs + dendRsecs
	
	# Get tree topology
	stn_tree = reducemodel.combinedtree()
	assign_strahler(stn_tree)

	# Cluster based on functional regions
	clusterize_strahler_order(stn_tree)

	# Merge within-cluster branches until islands remain
	for clusterlabel in ['soma', 'smooth', 'spiny']:
		for secA, secB, secP in find_mergeable(alldendsecs, clusterlabel):
			# Now merge these sections
			if secP is not None and secB is None:
				mergeChildWithParent()
			elif secP is not None:
				mergeYWithParent()




if __name__ == '__main__':
	stn_tree = reducemodel.combinedtree()
	assign_strahler(stn_tree)