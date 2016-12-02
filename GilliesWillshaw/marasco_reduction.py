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
import warnings

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

class Cluster(object):
	""" A cluster representing merged sections """
	def __init__(self, label):
		self.label = label




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

def clusterize_strahler(noderef, allsecrefs, labelsuffix=''):
	""" Cluster a tree based on strahler numbers alone

	@param noderef	any section starting from but not equal to soma
	@effect			assign label 'trunk'/'smooth'/spiny' to sections
					based on their Strahler number
	"""

	if noderef is None: return

	# Cluster label based on strahler's number
	if noderef.strahlernumber >= 5:
		noderef.cluster_label = 'trunk' + labelsuffix
	elif noderef.strahlernumber > 3:
		noderef.cluster_label = 'smooth' + labelsuffix
	else:
		noderef.cluster_label = 'spiny' + labelsuffix

	# Parent cluster
	parref = getsecref(noderef.parent, allsecrefs)
	noderef.parent_label = parref.cluster_label
	if parref.cluster_label.startswith('soma'):
		noderef.parent_pos = 0.5
	else:
		noderef.parent_pos = 0.0

	# Cluster children (iteratively)
	childiter = iter(noderef.sec.children())
	leftref = getsecref(next(childiter, None), allsecrefs)
	rightref = getsecref(next(childiter, None), allsecrefs)
	clusterize_strahler_order(leftref, allsecrefs)
	clusterize_strahler_order(rightref, allsecrefs)

def clusterize_strahler_trunks(allsecrefs):
	""" Cluster a tree based on strahler numbers and depending
		on the trunk section it is attached to (trunk sections
		are large dendritic sections attached to soma)

	@param allsecrefs	list of SectionRef (mutable) with first element
						a ref to root/soma section

	ALGORITHM: this is the algorithm used in Marasco (2012)
	- each trunk section (neighbouring soma) gets its own cluster
	- for each trunk, two clusters are created: one for smooth and
	  one for spiny sections attached to that trunk
	"""
	# Cluster soma
	somaref = allsecrefs[0]
	somaref.cluster_label = 'soma'
	somaref.parent_label = 'soma'
	somaref.parent_pos = 0.0

	# Cluster each trunk and its subtree
	for i, trunksec in enumerate(somaref.sec.children()):
		trunkref = getsecref(trunksec, allsecrefs)
		clusterize_strahler(trunkref, allsecrefs, labelsuffix='_'+str(i))

def clusterize_marasco(allsecrefs):
	""" Assign each section to cluster and determine cluster topology
		according to algorithm Marasco (2012)

	ALGORITHM: algorithm from clusterise1PurkinjeCell()
	- soma gets its own cluster (the root cluster)
	- sections with strahler order >= 5 (trunk sections) get their own cluster
		- except if their parent is also a trunk section
	- for each trunk cluster: one new cluster is created for spiny sections in
	  its subtree and one for smooth sections
	"""
	# Split it soma/dendrite sections
	somaref = allsecrefs[0]
	alldendrefs = allsecrefs[1:-1]
	alldendrefs.sort(key=attrgetter('order'), reverse=False) # sord by nseg from soma

	# Create one cluster for soma
	somaref.cluster_label = 'soma'
	somaref.parent_label = 'soma'
	somaref.parent_pos = 0.0

	# Create one cluster for each trunk section
	n_trunk = 0
	for secref in alldendrefs if secref.strahlernumber >= 5 or secref.parent is somaref.sec
		parref = getsecref(secref.parent, allsecrefs)
		if hasattr(parref, 'cluster_label'):
			# Create new trunk cluster for this section
			secref.cluster_label = 'trunk' + str(n_trunk)
			secref.trunk_number = n_trunk
			n_trunk += 1

			# assign parent cluster
			secref.parent_label = parref.cluster_label
			if secref.parent is somaref.sec:
				secref.parent_pos = 0.5
			else:
				secref.parent_pos = 1.0
		else:
			warnings.warn('Encountered trunk section with unclustered parent')

	# cluster subtree of each trunk sections (assign cluster, set topology)
	for trunkref in alldendrefs if (hasattr(secref, 'cluster_label') and 
									secref.cluster_label.startswith('trunk')):
		i_trunk = trunkref.trunk_number

		# Get all sections in trunk subtree
		trunksubtree = h.SectionList()
		trunkref.sec.push()
		trunksubtree.subtree()
		h.pop_section()
		trunksubtree = list(trunksubtree)
		trunksubtree.remove(trunkref.sec) # don't include trunk itself
		trunksubrefs = [getsecref(sec, allsecrefs) for sec in trunksubtree]

		# All smooth secs in subtree get their own cluster
		has_smooth = any(secref in trunksubrefs if secref.strahlernumber > 3)
		for subref in trunksubrefs:
			secref.trunk_number = i_trunk # mark the trunk it is on
			if subref.strahlernumber > 3: # SMOOTH
				secref.cluster_label = 'smooth' + str(i_trunk)
				secref.parent_label = trunkref.cluster_label
				secref.parent_pos = 1.0
			elif has_smooth: # SPINY with smooth as parent
				secref.cluster_label = 'spiny' + str(i_trunk)
				secref.parent_label = 'smooth' + str(i_trunk)
				secref.parent_pos = 1.0
			else:			 # SPINY with trunk as parent
				secref.cluster_label = 'spiny' + str(i_trunk)
				secref.parent_label = 'trunk' + str(i_trunk)
				secref.parent_pos = 1.0

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
					secref.pathri0 += seg.ri()
		
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
		secref.doMerge = False

def find_mergeable(secrefs, clusterlabel):
	""" Find next mergeable sections

	@param secrefs			list of mutable SectionRef containing all sections
	@param clusterlabel		label of the cluster to merge
	@effect					- find next available section A and its sibling B
							- set them to visited (unavailable for search)
							- find their parent P if available and within cluster
							- if P found: set A and B availbe for merge
	@return 				tuple of Section Ref (secA, secB, secP)
	"""

	# Get sections in current cluster and sort by order
	clustersecs = [sec in secrefs if sec.cluster_label == clusterlabel]
	
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
			if secA: secA.doMerge = True
			if secB: secB.doMerge = True

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

def calc_eq_ppties_mrg(cluster, allsecrefs):
	"""
	Calculate properties of equivalent section for cluster that are based
	on the sections marked for merging
	"""
	# Initialize cluster properties
	cluster.maxL = 0 # L of longest sec in cluster
	cluster.totsurf = 0 # sum of surf calc from mrg dimensions for each merge available sec
	cluster.totdiam = 0 # sum of mrgdiam for each merge available sec
	cluster.eqri2Prod = 1
	cluster.eqri2Sum = 0
	cluster.surfSum = 0 # sum of mrgSum property for each merge available sec

	# Gather sections for merging
	mrg_secs = [secref in allsecrefs if secref.doMerge and secref.cluster_label==cluster.label]
	cluster.weightsL = [1.0]*len(mrg_secs)

	# Update cluster properties based on mergeable sections
	for i, secref in enumerate(mrg_secs):

		if cluster.maxL < secref.mrgL:
			cluster.maxL = secref.mrgL
		cluster.totsurf += secref.mrgL*PI*secref.mrgdiam
		cluster.totdiam += secref.mrgdiam
		cluster.eqri2Prod *= secref.mrgri2
		cluster.eqri2Sum += secref.mrgri2
		cluster.surfSum += secref.mrgsurf

		# Equivalent dimensions
		cluster.normalFact = totsurf
		cluster.weightsL[i] = secref.mrgL*PI*secref.mrgdiam
		cluster.eqL += secref.mrgL*cluster.weightsL[i]
		cluster.eqdiam += (secref.sec.Ra*cluster.maxL*4.)/(secref.mrgri*PI*100.) # rho squared

	cluster.eqdiam = math.sqrt(cluster.eqdiam)
	cluster.eqri2 = cluster.eqri2Sum/len(mrg_secs)
	cluster.eqSurf = cluster.eqL*PI*cluster.eqdiam
	cluster.orSurf1 = cluster.surfSum

	return cluster

def calc_eq_ppties_all(cluster, allsecrefs):
	"""
	Calculate properties of equivalent section for cluster that are based
	on all of its sections

	@param allsecrefs	list of SectionRef (mutable) with first element
						a ref to root/soma section
	"""

	somaref = allsecrefs[0]

	# Compute cluster properties based on all its sections
	clu_secs = [secref in secrefs if secref.cluster_label==cluster.label]
	cluster.eqRa = sum(secref.sec.Ra for secref in clu_secs)/len(clu_secs)
	cluster.orMaxpathri = max(secref.pathri1 for secref in clu_secs)
	cluster.orMinpathri = min(secref.pathri0 for secref in clu_secs)

	# Clusters that contain only one section (trunk/soma)
	if len(clu_secs) == 1:
		secref = clu_secs[0]
		if secref.has_parent() and secref.parent() is not somaref.sec
			parref = getsecref(secref.parent(), allsecrefs)
			orMinpathri = parref.pathri1
		else:
			orMinpathri = 0

	# Correct if min > max
	if orMinpathri > orMaxpathri:
		orMinpathri, orMaxpathri = orMaxpathri, orMinpathri # swap


################################################################################
# Equivalent/reduced cell
################################################################################

def create_equivalent_sections(eq_clusters, allsecrefs):
	""" Create the reduced/equivalent cell by creating 
		a section for each cluster 

	@param eq_clusters	list of Cluster objects containing data
						for each cluster
	@param allsecrefs	list of SectionRef (mutable) with first element
						a ref to root/soma section
	"""
	# Create equivalent section for each clusters
	eq_sections = [h.Section() for clu in eq_clusters]
	for i, clu_i in enumerate(eq_clusters):
		for j, clu_j in enumerate(eq_clusters):
			if clu_j is not clu_i and clu_j.parent_label == clu_i.label:
				eq_sections[j].connect(eq_sections[i], clu_j.parent_pos, 0)

	# Set L/diam/Ra/cm/g_leak for each equivalent section
	
	# Calculate equivalent path resistance for each cluster/equivalent section
	# FIXME: this was wrong, calculation happend on equivalent sections, after created
	for i, secref in enumerate(eq_secs):

		# Cluster corresponding to current section
		cluster = next(clu in clusterList if clu.label = secref.cluster_label)
		cluster.eqsecpathri1 = 0.0
		cluster.eqsecpathri0 = 0.0

		# Get path from root node to this sections
		rootsec = treeroot(secref)
		calc_path = h.RangeVarPlot('v')
		rootsec.push()
		calc_path.begin(.5)
		sec.push()
		calc_path.end(.5)
		root_path = h.SectionList()
		calc_path.list(root_path) # store path in list
		h.pop_section()
		h.pop_section()

		# Compute axial path resistances
		path_secs = list(root_path)
		path_len = len(root_path)
		for i, psec in enumerate(path_secs):
			for seg in psec:
				cluster.eqsecpathri1 += seg.ri()
				if i < path_len-1:
					cluster.eqsecpathri0 += seg.ri()

################################################################################
# Main routine
################################################################################

def reduce_marasco():
	""" Implementation of Marasco (2013) CellPurkAnalysis() & PurkReduction() """

	# Initialize Gillies model
	h.xopen("createcell.hoc")
	# Make sections accesible by both name and index + allow to add attributes
	somaref = ExtSection(h.SThcell[0].soma)
	dendLrefs = [ExtSection(sec) for sec in h.SThcell[0].dend0]
	dendRrefs = [ExtSection(sec) for sec in h.SThcell[0].dend1]
	alldendrefs = dendLrefs + dendRrefs
	allsecrefs = [somaref] + alldendrefs
	
	# Cluster subtree of each trunk section
	clusterize_strahler_trunks(allsecrefs)
	cluster_labels = list(set(secref.cluster_label for secref in allsecrefs)) # unique labels
	eq_clusters = [Cluster(label) for label in cluster_labels]

	# Merge within-cluster branches until islands remain
	for cluster in eq_clusters:

		# First merge children into parents (iteratively update parent properties)
		for secA, secB, secP in find_mergeable(alldendrefs, cluster.label):
			# Now merge these sections
			if secP is not None and secB is not None:
				mergeYWithParent()
			elif secP is not None:
				mergeChildWithParent()
				

		# Calculate cluster/equivalent section properties
		calc_eq_ppties_mrg(cluster, allsecrefs)
		calc_eq_ppties_all(cluster, allsecrefs)
		clusterList.append(cluster)

	# Initialize the equivalent model (one equivalent sec per cluster)
	create_equivalent_sections(eq_clusters, allsecrefs)

if __name__ == '__main__':
	stn_tree = reducemodel.combinedtree()
	assign_strahler(stn_tree)