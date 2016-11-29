"""
Reduce Gillies & Willshaw (2006) STN neuron model using Marasco 2012/2013
reduction method


@author Lucas Koelman
@date	28-11-2016
"""
# Python modules
from collections import OrderedDict
from operator import attrgetter

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

	clusterize_strahler_order(leftref)
	clusterize_strahler_order(rightref)


################################################################################
# Merging
################################################################################

def get_next_merge(secrefs, clusterlabel):
	""" Get next sections to merge.

	@param secrefs	list of mutable SectionRef containing all sections
	@param clusterlabel	label of the cluster to merge
	"""

	# Get sections in current cluster and sort by order
	clustersecs = [sec in secrefs if sec.clusterlabel == clusterlabel]
	for secref in clustersecs: # mark all as unvisited and unmerged
		secref.visited = False
		secref.merged = False
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
		for secA, secB, secP in get_next_merge(alldendsecs, clusterlabel):



if __name__ == '__main__':
	stn_tree = reducemodel.combinedtree()
	assign_strahler(stn_tree)