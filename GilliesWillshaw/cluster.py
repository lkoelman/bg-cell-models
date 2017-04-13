"""
Functions for clustering sections in dendritic trees.
"""

import logging
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module

from reduction_tools import getsecref

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

class EqProps:
	""" Equivalent properties of merged sections """
	def __init__(self, **kwds):
		self.__dict__.update(kwds)

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
		# one child: inherit strahler number
		noderef.strahlernumber = leftref.strahlernumber
	elif leftref.strahlernumber != rightref.strahlernumber:
		# nonzero and unequal: max strahler number of children
		noderef.strahlernumber = max(leftref.strahlernumber, rightref.strahlernumber)
	else:
		# nonzero and equal: increment strahler number
		noderef.strahlernumber = leftref.strahlernumber + 1

def label_from_regions_gbar(secref):
	"""
	Assign cluster label based on functional regions identified from
	channel distribution (max conductance for each channel)
	"""
	if secref.tree_index == 0: # left tree
		if secref.table_index == 1:
			pass
		else:
			pass
	elif secref.tree_index == 1: # right tree
		pass
	else:
		raise Exception("Section has unknown tree index {}".format(secref.tree_index))

def clusterize_seg_electrotonic(noderef, allsecrefs, thresholds, clusterlist, labelsuffix=''):
	""" Cluster segments in dendritic tree based on length divided by 
		length constant (i.e. length in terms of its electrotonic length) 
		measured from soma section.

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

	# Cluster each segment
	noderef.cluster_labels = ['unassigned'] * noderef.sec.nseg
	for i in xrange(noderef.sec.nseg):
		pathL = noderef.pathL_elec[i] # electrotonic path length of segment i
		if pathL <= thresholds[0]:
			noderef.cluster_labels[i] = 'trunk' + labelsuffix
		elif len(thresholds) > 1 and pathL <= thresholds[1]:
			noderef.cluster_labels[i] = 'smooth' + labelsuffix
		else:
			noderef.cluster_labels[i] = 'spiny' + labelsuffix

	# If any new clusters created, create Cluster objects
	for label in noderef.cluster_labels:
		if not any(clu.label == label for clu in clusterlist):
			newclu = Cluster(label)
			clusterlist.append(newclu)

	# Cluster children (iteratively)
	for childsec in noderef.sec.children():
		childref = getsecref(childsec, allsecrefs)
		clusterize_seg_electrotonic(childref, allsecrefs, thresholds, clusterlist, labelsuffix)


def clusterize_sec_electrotonic(noderef, allsecrefs, thresholds, clusterlist, labelsuffix=''):
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
		noderef.cluster_label = 'trunk' + labelsuffix
	elif len(thresholds) > 1 and L_mid <= thresholds[1]:
		noderef.cluster_label = 'smooth' + labelsuffix
	else:
		noderef.cluster_label = 'spiny' + labelsuffix

	logger.debug("Section with index {} assigned to cluster {}".format(
					noderef.table_index, noderef.cluster_label))

	# If first section belonging to this cluster, add new Cluster object
	if not any(clu.label==noderef.cluster_label for clu in clusterlist):
		newclu = Cluster(noderef.cluster_label)
		clusterlist.append(newclu)

	# Cluster children (iteratively)
	for childsec in noderef.sec.children():
		childref = getsecref(childsec, allsecrefs)
		clusterize_sec_electrotonic(childref, allsecrefs, thresholds, clusterlist, labelsuffix)

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

	# Add new Cluster object if necessary
	if not any(clu.label==noderef.cluster_label for clu in clusterlist):
		newclu = Cluster(noderef.cluster_label)
		clusterlist.append(newclu)

	# Cluster children (iteratively)
	for childsec in noderef.sec.children():
		childref = getsecref(childsec, allsecrefs)
		clusterize_custom(childref, allsecrefs, clusterfun, clusterlist, labelsuffix)

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

	# Add new cluster
	if not any(clu.label==noderef.cluster_label for clu in clusterlist):
		newclu = Cluster(noderef.cluster_label)
		clusterlist.append(newclu)

	# Cluster children (iteratively)
	for childsec in noderef.sec.children():
		childref = getsecref(childsec, allsecrefs)
		clusterize_strahler(childref, allsecrefs, thresholds, clusterlist, labelsuffix)

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

def assign_topology(clusters, radial_prefixes):
	""" 
	Assign parent_label and parent_pos for each Cluster in clusters
	based on the given prefixes, which should be ordered from
	proximal to distal.
	"""
	for clu in clusters:
		# Get current cluster prefix
		prepat = re.compile(r'^[a-zA-Z0-9]+') # same as ^[^\W_]+ : matches anything before first underscore
		match = re.search(prepat, clu.label) 
		clu_prefix = match.group()

		# Defaults
		clu.parent_label = clu.label # if no parent found it is root cluster
		clu.parent_pos = 1.0

		# look for next prefix in anti-radial direction that is available
		prefix_index = radial_prefixes.index(clu_prefix)
		parent_index = prefix_index

		# Root cluster
		if prefix_index == 0:
			clu.parent_label = clu.label
			clu.order = 0
		
		# Non-root clusters
		while parent_index > 0:
			parent_prefix = radial_prefixes[parent_index-1]
			parent_clu = next((clu for clu in clusters if clu.label.startswith(parent_prefix)), None)
			if parent_clu is not None:
				logger.debug("Found parent <%s> for cluster <%s>" % (parent_clu.label, clu.label))
				clu.parent_label = parent_clu.label
				clu.order = parent_clu.order + 1
				break
			parent_index -= 1

		# Sanity check
		assert(prefix_index==0 or clu.parent_label!=clu.label)