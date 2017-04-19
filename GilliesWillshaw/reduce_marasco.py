"""
Reduce Gillies & Willshaw (2006) STN neuron model using the method
described in Marasco & Migliore (2012)


@author Lucas Koelman
@date	5-12-2016
"""

# Python modules
import math
PI = math.pi

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module

import sys
import os.path
scriptdir, scriptfile = os.path.split(__file__)
modulesbase = os.path.normpath(os.path.join(scriptdir, '..'))
sys.path.append(modulesbase)

# NEURON modules
import neuron
h = neuron.h

# Load NEURON function libraries
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

# Load NEURON mechanisms
# add this line to nrn/lib/python/neuron/__init__.py/load_mechanisms()
# from sys import platform as osplatform
# if osplatform == 'win32':
# 	lib_path = os.path.join(path, 'nrnmech.dll')
NRN_MECH_PATH = os.path.normpath(os.path.join(scriptdir, 'nrn_mechs'))
neuron.load_mechanisms(NRN_MECH_PATH)

# Own modules
import reduction_tools as redtools
from reduction_tools import ExtSecRef, getsecref, lambda_AC, prev_seg, seg_index # for convenience
import cluster as clutools
from cluster import Cluster, EqProps
import reduce_bush_sejnowski as redbush
import reduction_analysis as analysis

# Global variables (convert to class members in future)
gillies_mechs_chans = {'STh': ['gpas'], # passive/leak channel
				'Na': ['gna'], 'NaL': ['gna'], # Na channels
				'KDR': ['gk'], 'Kv31': ['gk'], 'sKCa':['gk'], # K channels
				'Ih': ['gk'], # nonspecific channels
				'CaT': ['gcaT'], 'HVA': ['gcaL', 'gcaN'], # Ca channels
				'Cacum': []} # No channels

mechs_chans = gillies_mechs_chans
gleak_name = 'gpas_STh'
glist = [gname+'_'+mech for mech,chans in mechs_chans.iteritems() for gname in chans]

def merge_parallel(childrefs, allsecrefs):
	"""
	Merge parallel sections at branch point using <br> equations in Marasco (2012)

	ALGORITHM
	- `L_br = sum(S_i*L_i)/sum(S_i)` where S_I is the area of branch i
	- `diam_br = sqrt(sum(diam_i^2))`
	- `r_a,br = prod(r_a,i)/sum(r_a,i)` where r_a is the axial resistance Ri
	"""
	# Initialize combined properties of branched sections (children)
	L_br = 0.
	diam_br = 0.
	Ra_br = 0.
	rin_br = 1.
	eqsurf_sum = 0.
	ri_sum = 0.
	gtot_br = dict((gname, 0.0) for gname in glist) # sum of gbar multiplied by area
	cmtot_br = 0. # sum of cm multiplied by area

	# Update combined properties using the dimensions/electrical properties
	# of each child according to <br> expressions in Marasco (2012) 
	for childref in childrefs:
		# get equivalent child properties
		L_child, diam_child, Ra_child, ri_child, cmtot_child, gtot_child = merge_sequential(childref, allsecrefs)
		eqsurf_child = PI*diam_child*L_child

		# combine according to <br> (parallel) expressions
		eqsurf_sum += eqsurf_child
		L_br += eqsurf_child*L_child # LENGTH: eq (1) - weight by area
		diam_br += diam_child**2 # RADIUS: eq (2) - 2-norm of radii
		Ra_br += Ra_child # SPECIFIC AXIAL RESISTANCE - average Ra
		rin_br *= ri_child # ABSOLUTE AXIAL RESISTANCE - parallel conductances
		ri_sum += ri_child # need sum in final calculation

		# Distributed properties
		cmtot_br += cmtot_child
		for gname in glist:
			gtot_br[gname] += gtot_child[gname]

		# mark child as absorbed
		childref.absorbed = True
		childref.visited = True

	# Finalize <br> calculation (MUST BE VALID IF ONLY ONE CHILDREF)
	L_br /= eqsurf_sum # eq. L_br
	diam_br = math.sqrt(diam_br) # eq. rho_br
	Ra_br = Ra_br/len(childrefs) # average Ra (NOTE: unused, cluster Ra calculated from equation Ra^{eq})
	cross_br = PI*diam_br**2/4. # cross-section area
	rax_br = Ra_br*L_br/cross_br/100. # absolute axial resistance of section with equivalent dimensions and Ra
	if len(childrefs) > 1:
		rin_br /= ri_sum # eq. r_a,br: product/sum

	return L_br, diam_br, Ra_br, rin_br, rax_br, cmtot_br, gtot_br

def merge_sequential(rootref, allsecrefs):
	""" 
	Merge sequential sections into one equivalent sections
	"""
	# Get references
	rootsec = rootref.sec
	allchildrefs = [getsecref(sec, allsecrefs) for sec in rootsec.children()]
	childrefs = [ref for ref in allchildrefs if ref.cluster_label==rootref.cluster_label]

	# Gather properties for root sec
	L_root = rootsec.L
	diam_root = rootsec.diam
	Ra_root = rootsec.Ra
	ri_root = sum(seg.ri() for seg in rootsec) # absolute axial resistance between 0-1 ends

	# Distributed properties
	cmtot_seq = sum(seg.cm*seg.area() for seg in rootsec) # sum of cm multiplied by area
	gtot_seq = dict((gname, 0.0) for gname in glist) # sum of gbar multiplied by area
	for gname in glist:
		gtot_seq[gname] += sum(getattr(seg, gname)*seg.area() for seg in rootsec)

	# Handle leaf sections
	if not any(childrefs):
		return L_root, diam_root, Ra_root, ri_root, cmtot_seq, gtot_seq

	# Combine properties of parallel branched sections (children)
	L_br, diam_br, Ra_br, rin_br, rax_br, cmtot_br, gtot_br = merge_parallel(childrefs, allsecrefs)

	# use <seq> expressions in Marasco (2012) to merge equivalent child into parent
	L_seq = L_root + L_br # L_seq equation
	Ra_seq = (Ra_root + Ra_br)/2.
	ri_seq = ri_root + rin_br # var 'newri2' in Marasco code used for ri calculation
	rax_seq = ri_root + rax_br # var 'newri' in Marasco code used for diam calculation
	diam_seq = math.sqrt(Ra_seq*L_seq*4./PI/rax_seq/100.) # rho_seq equation (conserves ri_seq)
	
	# Keep track of total conductance/capacitance
	cmtot_seq += cmtot_br
	for gname in glist:
		gtot_seq[gname] += gtot_br[gname]

	return L_seq, diam_seq, Ra_seq, ri_seq, cmtot_seq, gtot_seq

def collapse_subtree(rootref, allsecrefs):
	""" 
	Recursively merge within-cluster connected sections in subtree
	of the given node using <br> and <seq> expressions in Marasco (2012).
	"""
	# Collapse is equal to sequential merge of the root and equivalent parallel circuit of its children
	return merge_sequential(rootref, allsecrefs)

def collapse_seg_subtree(rootseg, allsecrefs):
	""" 
	Recursively merge within-cluster connected sections in subtree
	of the given segment using equations <br> and <seq> in Marasco (2012).

	@pre		in the tree, all connections between sections must be made
				according to the convention that the 0-end of the child
				connects to the 1-end of the parent

	ALGORITHM
	- for each child segment: recursively call `equivalent_properties = collapse_subtree(child)`
	- then combine the equivalent properties: absorb into current rootseg and return
	"""
	# Get segment info
	rootsec = rootseg.sec
	rootref = getsecref(rootsec, allsecrefs)
	neighbors = [seg for seg in rootsec]
	i_rootseg = seg_index(rootseg)
	# i_rootseg = next(i for i, seg in enumerate(neighbors) if round(rootseg.x,3)==round(seg.x,3))
	rootlabel = rootref.cluster_labels[i_rootseg]

	# Calculate properties of root segment
	L_root = rootsec.L/rootsec.nseg
	diam_root = rootseg.diam
	Ra_root = rootsec.Ra
	Ri_root = rootseg.ri() # axial resistance between start-end (0-1)

	# 1. Get Children to absorb ###############################################

	# Get children
	child_refs = [getsecref(sec, allsecrefs) for sec in rootsec.children()]

	# If end segment: add child segments if in same cluster
	child_segs = []
	if i_rootseg == rootsec.nseg-1: # or: rootseg.x >= neighbors[-1].x:
		assert rootseg.x >= neighbors[-1].x
		# Gather child segments that are in same cluster
		for secref in child_refs:
			seg = next(seg for seg in secref.sec) # get the first segment
			if secref.cluster_labels[0]==rootlabel:
				# Check if attempting to merge children not connected to 1-end
				if round(secref.sec.parentseg().x,3) < round(neighbors[-1].x, 3):
					raise Exception("Merging algorithm does not support topologies where "
									"a child section is not connected to 1-end of parent.")
				child_segs.append(seg)
				# mark segment
				secref.visited[0] = True
				secref.absorbed[0] = True

	else: # If not end segment: add adjacent segment if in same cluster
		if rootref.cluster_labels[i_rootseg+1] == rootlabel:
			child_segs.append(neighbors[i_rootseg+1])
			rootref.visited[i_rootseg+1] = True
			rootref.absorbed[i_rootseg+1] = True

	# 2. Use solution of recursive call to solve ##############################

	# Base Case: leaf segments (i.e. no child segments in same cluster)
	if not any(child_segs):
		eq_props = EqProps(L_eq=L_root, diam_eq=diam_root, Ra_eq=Ra_root, Ri_eq=Ri_root)
		return eq_props

	# Parallel merge of child properties (use <br> equations Marasco (2012))
	# NOTE: if only one child, <br> equations must yield properties of that child
	L_br = 0.
	diam_br = 0.
	Ra_br = 0.
	Ri_br = 1.
	area_br = 0.
	cross_area_br = 0.
	Ri_br_sum = 0. # sum of axial resistances of parallel child branches
	for child in child_segs:
		cp = collapse_seg_subtree(child, allsecrefs)
		ch_area = PI*cp.diam_eq*cp.L_eq

		# Update equivalent properties of all child branches in parallel
		area_br += ch_area
		L_br += ch_area*cp.L_eq		# Length: Eq. L_br
		diam_br += cp.diam_eq**2	# Diameter: Eq. rho_br
		Ra_br += cp.Ra_eq			# Axial resistivity: average Ra
		Ri_br *= cp.Ri_eq			# Axial resistance: Eq. r_a,br
		Ri_br_sum += cp.Ri_eq		# Sum of branch axial resistances

	# Finalize <br> calculation
	L_br /= area_br
	diam_br = math.sqrt(diam_br)
	Ra_br = Ra_br/len(child_segs) # NOTE: unused, cluster Ra calculated from equation Ra^{eq}
	cross_area_br = PI*diam_br**2/4.
	if len(child_segs) > 1:
		Ri_br /= Ri_br_sum # Must be valid parallel circuit of Ri if only one child
	Ri_br_eqgeom = Ra_br*L_br/cross_area_br/100. # Axial resistance of section with geometry equal to merged properties


	# Sequential merge of root & merged child properties (use <seq> equations Marasco (2012))
	L_seq = L_root + L_br			# Eq. L_seq
	Ra_seq = (Ra_root + Ra_br)/2.	# for rho_seq calculation
	Ri_seq = Ri_root + Ri_br		# Eq. r_a,seq
	Ri_seq_eqgeom = Ri_root + Ri_br_eqgeom						# for diam_seq calculation
	diam_seq = math.sqrt(Ra_seq*L_seq*4./PI/Ri_seq_eqgeom/100.)	# Eq. rho_seq
	eq_props = EqProps(L_eq=L_seq, diam_eq=diam_seq, Ra_eq=Ra_seq, Ri_eq=Ri_seq)
	return eq_props


def map_pathri_gbar(cluster, allsecrefs):
	"""
	Map axial path resistance values to gbar values in the cluster

	ALGORITHM:
	- for each section:
		- for each segment in section, save gbar and axial path resistance
			- axial path resistance obtained by interpolating pathri0 & pathri1
	"""
	clu_secs = [secref for secref in allsecrefs if secref.cluster_label==cluster.label]

	# Keep a dict that maps gname to a collection of data points (pathri, gbar)
	cluster.pathri_gbar = dict((gname, []) for gname in glist)
	for secref in clu_secs:
		for seg in secref.sec:
			for gname in glist:
				if not hasattr(seg, gname):
					continue # section doesn't have ion channel: skip
				# Add data point
				seg_pathri = secref.pathri0 + seg.x*(secref.pathri1-secref.pathri0)
				seg_gbar = getattr(seg, gname)
				cluster.pathri_gbar[gname].append((seg_pathri, seg_gbar))

def calc_gbar(cluster, gname, pathri):
	"""
	Calculate gbar for a point on the equivalent section of given cluster
	given the axial path resistance to that point.
	"""
	gbar_pts = cluster.pathri_gbar[gname] # list of (pathri, gbar) data points
	gbar_pts.sort(key=lambda pt: pt[0]) # sort by pahtri ascending
	eq_path_x = (pathri - cluster.pathri0) / (cluster.pathri1 - cluster.pathri0)
	if eq_path_x <= 0.:
		return gbar_pts[0]
	elif eq_path_x >= 1.:
		return gbar_pts[-1]
	else:
		# average of points that lie within pathri +/- 11% of segment axial resistance
		deltari = 0.11*(cluster.pathri1-cluster.pathri0)
		surr_pts = [pt for pt in gbar_pts if (pt[0] >= pathri-deltari) and (pt[0] <= pathri+deltari)]
		if not any(surr_pts):
			# take average of two closest points
			gbar_pts.sort(key=lambda pt: abs(pt[0]-pathri))
			surr_pts = [pt for i, pt in enumerate(gbar_pts) if i < 2]
		return sum(pt[1] for pt in surr_pts)/len(surr_pts) # take average

def has_cluster_parentseg(seg, cluster, clu_secs):
	""" Utility function to check if segment has parent segment in same cluster """
	parseg = prev_seg(seg)
	if parseg is None:
		return False # no parent segment
	parref = getsecref(parseg.sec, clu_secs)
	if parref is None:
		return False # parent section has no segments in same cluster
	for j, seg in enumerate(parref.sec):
		if (seg.x==parseg.x) and (parref.cluster_labels[j]==cluster.label):
			return True
	return False

def merge_seg_cluster(cluster, allsecrefs, average_Ri):
	"""
	Merge cluster of segments (use for segment-based clustering)

	@param average_Ri	see param 'average_Ri' in merge_sec_cluster()

	@post	cluster will have following attributes, calculated from member segments:

			or_area		original surface area of all member segments
			or_cmtot	original summed capacitance (nonspecific) of all member segments
			or_gtot		original maximum conductance (nonspecific) of all member segments,
						for all inserted conductance mechanisms
			
			eqL			equivalent length
			eqdiam		equivalent diameter
			eqri		equivalent axial resistance Ri (absolute, in Mega Ohms)
			eqRa		equivalent axial cytoplasmic resistivity
			eq_area		equivalent area based on equivalent dimensions and passive properties
			eq_area_sum	sum of equilvalent surfaces of 'islands' of segments

	"""
	# Gather member segments
	clu_secs = [secref for secref in allsecrefs if (cluster.label in secref.cluster_labels)]
	clu_segs = []
	for ref in allsecrefs:
		# Flag all sections as unvisited
		if not hasattr(ref, 'absorbed'):
			ref.absorbed = [False] * ref.sec.nseg
		if not hasattr(ref, 'visited'):
			ref.visited = [False] * ref.sec.nseg

		# Gather segments that are member of current cluster
		for i, seg in enumerate(ref.sec):
			if ref.cluster_labels[i] == cluster.label:
				clu_segs.append(seg)

	# Calculate axial path resistance to all segments in cluster
	clu_seg_pathri = []
	for secref in clu_secs:
		# Assign axial path resistances
		redtools.sec_path_ri(secref, store_seg_ri=True) # stores pathri on SectionRefs
		# Store pathri to start of segments in cluster
		segs_pathri = [pathri for i, pathri in enumerate(secref.pathri_seg) if (
						secref.cluster_labels[i]==cluster.label)]
		# Also store pathri to end of most distal segment
		if secref.cluster_labels[-1]==cluster.label:
			segs_pathri.append(secref.pathri1)
		clu_seg_pathri.extend(segs_pathri)

	# Get min & max path resistance in cluster (full model)
	cluster.orMaxpathri = max(clu_seg_pathri)
	cluster.orMinpathri = min(clu_seg_pathri)
	
	# Initialize equivalent properties
	cluster.eqL = 0.
	cluster.eqdiam = 0.
	cluster.eqri = 0.
	cluster.eq_area_sum = 0. # sum of surface calculated from equivalent dimensions

	# Initialize original properties
	cluster.or_area = sum(seg.area() for seg in clu_segs)
	cluster.or_cmtot = sum(seg.cm*seg.area() for seg in clu_segs)
	cluster.or_gtot = dict((gname, 0.0) for gname in glist)
	for gname in glist:
		cluster.or_gtot[gname] += sum(getattr(seg, gname)*seg.area() for seg in clu_segs)

	# Make generator that finds root segments
	root_gen = (seg for ref in clu_secs for i, seg in enumerate(ref.sec) if (
					not ref.visited[i] and not has_cluster_parentseg(seg, cluster, clu_secs)))

	# Start merging algorithm
	num_roots = 0
	while True:
		# Find next root segment in cluster (i.e. with no parent segment in same cluster)
		rootseg = next(root_gen, None)
		if rootseg is None:
			break # No more root segment left
		rootref = getsecref(rootseg.sec, allsecrefs)

		# Collapse subtree
		logger.debug("Collapsing subtree of cluster root segment %s", repr(rootseg))
		eq_props = collapse_seg_subtree(rootseg, allsecrefs)

		# Combine properties of collapse sections using <eq> expressions
		eq_area = eq_props.L_eq * PI * eq_props.diam_eq
		cluster.eq_area_sum += eq_area
		cluster.eqL += eq_props.L_eq * eq_area	# eq. L^eq
		cluster.eqdiam += eq_props.diam_eq**2	# eq. rho^eq
		cluster.eqri += eq_props.Ri_eq			# eq. ra^eq

		# Mark as visited
		i_seg = seg_index(rootseg)
		rootref.visited[i_seg] = True
		num_roots += 1

	# Finalize <or> calculation
	cluster.or_cm = cluster.or_cmtot / cluster.or_area

	# Finalize <eq> calculation
	cluster.eqL /= cluster.eq_area_sum			# eq. L^eq
	cluster.eqdiam = math.sqrt(cluster.eqdiam)	# eq. rho^eq
	cluster.eqri /= num_roots					# eq. ra^eq
	if average_Ri:
		cluster.eqRa = PI*(cluster.eqdiam/2.)**2*cluster.eqri*100./cluster.eqL # eq. Ra^eq
	else:
		cluster.eqRa = sum(seg.sec.Ra for seg in clu_segs)/len(clu_segs) # alternative Ra^eq: average in cluster
	cluster.eq_area = cluster.eqL*PI*cluster.eqdiam # area_eq based on equivalent geometry

	# Debugging info
	logger.debug("Merged cluster '%s': equivalent properties are:\
		\n\teqL\teqdiam\teqRa\teqri\teq_area\
		\n\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f",
		cluster.label, cluster.eqL, cluster.eqdiam, cluster.eqRa, cluster.eqri, cluster.eq_area)


def merge_sec_cluster(cluster, allsecrefs, average_Ri):
	"""
	Merge sections in cluster

	@param average_Ri	If True, Ra is calculated so that Ri (absolute axial
						resistance) of the equivalent section for each cluster is 
						the average Ri of all disconnected subtrees merged into that
						section. This does NOT conserve input resistance of the tree.

							If False, Ra is the average Ra in the cluster and the diameter 
						is calculated so that the absolute axial resistance Ri is 
						equivalent to the parallel circuit of all the unconnected 
						subtrees. This preserves input resistance

	ALGORITHM
	- find the next root of a within-cluster connected subtree
		- i.e. a section without a parent in the same cluster
		- this can be an isolated section (no mergeable children within cluster)
	- collapse subtree of that root section
	- update equivalent cluster properties using the <eq> expressions in Marasco (2012)

	"""
	# Gather sections in this cluster and mark/flag them
	clu_secs = [secref for secref in allsecrefs if secref.cluster_label==cluster.label]
	for sec in clu_secs:
		sec.absorbed = False
		sec.visited = False

	# Assign axial path resistance (sum of Ri (seg.ri()) along path)
	for sec in clu_secs:
		redtools.sec_path_ri(sec) # assigns pathri0/pathri1

	# Get min & max path resistance in cluster (full model)
	cluster.orMaxpathri = max(secref.pathri1 for secref in clu_secs)
	cluster.orMinpathri = min(secref.pathri0 for secref in clu_secs)
	
	# Initialize equivalent properties
	cluster.eqL = 0.
	cluster.eqdiam = 0.
	cluster.eqri = 0.
	cluster.eq_area_sum = 0. # sum of surface calculated from equivalent dimensions
	
	# Initialize original properties
	cluster.or_area = sum(sum(seg.area() for seg in secref.sec) for secref in clu_secs)
	cluster.or_cmtot = 0.
	cluster.or_gtot = dict((gname, 0.0) for gname in glist)

	# utility function to check if sec has parent within cluster
	def has_clusterparent(secref):
		return secref.has_parent() and (getsecref(secref.parent, clu_secs) is not None)

	# Find connected subtrees within cluster and merge/collapse them
	rootfinder = (sec for sec in clu_secs if (not sec.visited and not has_clusterparent(sec))) # compiles generator function
	for secref in rootfinder:
		rootref = clutools.clusterroot(secref, clu_secs) # make sure it is a cluster root

		# Collapse subtree
		logger.debug("Collapsing subtree of cluster root %s", repr(rootref))
		L_eq, diam_eq, Ra_eq, ri_eq, cmtot_eq, gtot_eq = collapse_subtree(rootref, allsecrefs)

		# Combine properties of collapse sections using <eq> expressions
		surf_eq = L_eq*PI*diam_eq
		cluster.eq_area_sum += surf_eq
		cluster.eqL += L_eq * surf_eq
		cluster.eqdiam += diam_eq**2
		cluster.eqri += ri_eq

		# Save distributed properties
		cluster.or_cmtot += cmtot_eq
		for gname in glist:
			cluster.or_gtot[gname] += gtot_eq[gname]

		# Mark as visited
		rootref.visited = True

	# Check each section either absorbed or rootsec
	# assert not any(not sec.absorbed and has_clusterparent(sec) for sec in clu_secs), (
	assert all(sec.absorbed or not has_clusterparent(sec) for sec in clu_secs), (
			'Each section should be either absorbed or be a root within the cluster')

	# Finalize <or> calculation
	cluster.or_cm = cluster.or_cmtot / cluster.or_area

	# Finalize <eq> calculation
	cluster.eqL /= cluster.eq_area_sum # LENGTH: equation L_eq
	cluster.eqdiam = math.sqrt(cluster.eqdiam) # RADIUS: equation rho_eq
	cluster.eqri /= sum(not sec.absorbed for sec in clu_secs) # ABSOLUTE AXIAL RESISTANCE: equation r_a,eq
	if average_Ri:
		cluster.eqRa = PI*(cluster.eqdiam/2.)**2*cluster.eqri*100./cluster.eqL # conserve eqri as absolute axial resistance
	else:
		cluster.eqRa = sum(secref.sec.Ra for secref in clu_secs)/len(clu_secs) # average Ra in cluster
	cluster.eq_area = cluster.eqL*PI*cluster.eqdiam # EQUIVALENT SURFACE

	# Debugging info
	logger.debug("Merged cluster '%s': equivalent properties are:\
		\n\teqL\teqdiam\teqRa\teqri\teq_area\
		\n\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f",
		cluster.label, cluster.eqL, cluster.eqdiam, cluster.eqRa, cluster.eqri, cluster.eq_area)

def min_nseg_marasco(sec):
	""" Minimum number of segments based on electrotonic length """
	return int((sec.L/(0.1*lambda_AC(sec,100.))+0.9)/2)*2 + 1  

def equivalent_sections(clusters, allsecrefs, gradients):
	""" Create the reduced/equivalent cell by creating 
		a section for each cluster 

	@param clusters		list of Cluster objects containing data
						for each cluster

	@param allsecrefs	list of SectionRef (mutable) with first element
						a ref to root/soma section

	@param gradients	if True, gbar in each segment is determined from
						the average gbar at the same path resistance in the
						original model (i.e. nonuniform gbar). If False,
						a uniform gbar is used in each equivalent section.

	@return				list of SectionRef containing equivalent Section 
						for each cluster (in same order) as well as min
						and max path resistance for each cluster/section
						as properties pathri0/pathri1 on SectionRef objects
	"""
	# Create equivalent section for each clusters
	# eq_secs = [h.Section() for clu in clusters]
	for clu in clusters:
		h("create %s" % clu.label) # ensures references are not destroyed and names are clear
	eq_secs = [getattr(h, clu.label) for clu in clusters]
	eq_secrefs = [ExtSecRef(sec=sec) for sec in eq_secs]

	# Connect sections
	for i, clu_i in enumerate(clusters):
		for j, clu_j in enumerate(clusters):
			if clu_j is not clu_i and clu_j.parent_label == clu_i.label:
				eq_secs[j].connect(eq_secs[i], clu_j.parent_pos, 0)

	# Set dimensions, passive properties, active properties
	for i, secref in enumerate(eq_secrefs):
		sec = secref.sec
		sec.push() # Make section the CAS
		cluster = clusters[i]

		# Store some cluster properties on SectionRef
		secref.cluster_label = cluster.label
		secref.order = cluster.order

		# Set geometry 
		sec.L = cluster.eqL
		sec.diam = cluster.eqdiam
		sec_area = sum(seg.area() for seg in sec) # should be same as cluster eq_area
		surf_fact = cluster.or_area/cluster.eq_area # scale factor: ratio areas original/equivalent

		# Passive electrical properties (except Rm/gleak)
		sec.cm = cluster.or_cmtot / sec_area
		sec.Ra = cluster.eqRa

		# Set number of segments based on rule of thumb electrotonic length
		sec.nseg = redtools.min_nseg_hines(sec)

		# calculate min/max path resistance in equivalent section (cluster)
		pathri0, pathri1 = redtools.sec_path_ri(eq_secrefs[i])
		cluster.pathri0 = pathri0
		cluster.pathri1 = pathri1
		sec_ri = sum(seg.ri() for seg in sec)
		assert (pathri0 < pathri1), ('Axial path resistance at end of section '
									 'should be higher than at start of section')
		assert (-1e-6 < (pathri1 - pathri0) - sec_ri < 1e-6), ('absolute axial '
						'resistance not consistent with axial path resistance')

		# Insert all mechanisms and set conductances
		for mech in mechs_chans.keys():
			sec.insert(mech)
		for seg in sec:
			for gname in glist:
				if gradients:
					# Look for average gbar value at points with same path resistance
					seg_pathri = pathri0 + seg.x*(pathri1-pathri0)
					gval = calc_gbar(cluster, gname, seg_pathri)
				else:
					gval = cluster.or_gtot[gname] / sec_area # yields same sum(gbar*area) as in full model
				seg.__setattr__(gname, gval)
		
		# Re-scale gbar distribution to yield same total gbar (sum(gbar*area))
		if gradients:
			for gname in glist:
				gtot_eq = sum(getattr(seg, gname)*seg.area() for seg in sec)
				gtot_or = cluster.or_gtot[gname]
				if gtot_eq <= 0. : gtot_eq = 1.
				for seg in sec:
					# NOTE: old method, almost same but less accurate
					# seg.__setattr__(gname, getattr(seg, gname)*surf_fact)
					# NOTE: this conserves gtot since sum(g_i*area_i * gtot_or/gtot_eq) = gtot_or/gtot_eq*sum(gi*area_i) = gtot_or/gtot_eq*gtot_eq
					seg.__setattr__(gname, getattr(seg, gname)*gtot_or/gtot_eq)
				# Check calculation
				gtot_eq_scaled = sum(getattr(seg, gname)*seg.area() for seg in sec)
				logger.debug("== Conductance %s ===\nOriginal: gtot = %.3f\nEquivalent: gtot = %.3f",
								gname, gtot_or, gtot_eq_scaled)

		# Debugging info:
		logger.debug("Created equivalent section '%s' with \n\tL\tdiam\tcm\tRa\tri\tpathri0\tpathri1\tnseg\
		\n\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d", cluster.label, sec.L, sec.diam, sec.cm, sec.Ra, 
		sec_ri, pathri0, pathri1, sec.nseg)

		# Unset CAS
		h.pop_section()

	return eq_secs, eq_secrefs # return both or secs will be deleted

def label_order(label):
	""" Return order (distance from soma) based on label """
	if label.startswith('soma'):
		return 0
	elif label.startswith('trunk'):
		return 1
	elif label.startswith('smooth'):
		return 2
	elif label.startswith('spiny'):
		return 3
	else:
		return 4

def reduce_gillies_pathLambda(segment_based=True, delete_old_cells=True):
	"""
	Reduce Gillies & Willshaw STN neuron model.

	To set active conductances, interpolates using electrotonic path length
	values (L/lambda).
	"""
	############################################################################
	# 0. Load full model to be reduced (Gillies & Willshaw STN)
	for sec in h.allsec():
		if not sec.name().startswith('SThcell') and delete_old_cells:
			h.delete_section() # delete existing cells
	if not hasattr(h, 'SThcells'):
		h.xopen("createcell.hoc")

	# Make sections accesible by name and index
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendLrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0] # 0 is left tree
	dendRrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1] # 1 is right tree
	allsecrefs = [somaref] + dendLrefs + dendRrefs

	# Assign indices used in Gillies code to read section properties from file
	somaref.tree_index = -1
	somaref.table_index = 0
	for j, dendlist in enumerate((dendLrefs, dendRrefs)):
		for i, secref in enumerate(dendlist):
			secref.tree_index = j # left tree is 0, right is 1
			secref.table_index = i+1 # same as in /sth-data/treeX-nom.dat

	############################################################################
	# 0. Pre-clustering: calculate properties

	# Assign lambda and electrotonic path length to each section and segment
	logger.info("Computing electrotonic path lengths...")
	f_lambda = 100. # frequency for electrotonic length constant lambda
	redtools.assign_electrotonic_length(somaref, allsecrefs, f_lambda, 
										gleak_name, allseg=True)

	############################################################################
	# 1. Cluster based on identified functional regions

	# Cluster soma
	somaclu = Cluster('soma')
	somaref.cluster_label = 'soma'
	somaref.cluster_labels = ['soma'] * somaref.sec.nseg
	somaclu.parent_label = 'soma'
	somaclu.parent_pos = 0.0
	somaclu.order = 0
	clusters = [somaclu]

	# Cluster dendritic trees
	clu_fun = clutools.label_from_custom_regions
	clu_args = {'marker_mech': ('hh', 'gnabar')} # flag segments based on cluster label
	somaref.sec.insert('hh')
	for seg in somaref.sec:
		seg.gnabar_hh = 5

	logger.info("Clustering left tree (dend0)...")
	clutools.clusterize_custom(dendLrefs[0], allsecrefs, clusters, '_0', clu_fun, clu_args)
	logger.info("Clustering right tree (dend1)...")
	clutools.clusterize_custom(dendRrefs[0], allsecrefs, clusters, '_1', clu_fun, clu_args)

	# Determine cluster relations/topology
	clutools.assign_topology(clusters, ['soma', 'trunk', 'smooth', 'spiny'])

	############################################################################
	# 2. Create equivalent sections

	# Merge sections within each cluster: i.e. calculate properties of equivalent section
	logger.info("Merging within-cluster sections...")
	average_Ri = True
	for cluster in clusters:
		if segment_based:
			merge_seg_cluster(cluster, allsecrefs, average_Ri)
		else:
			merge_sec_cluster(cluster, allsecrefs, average_Ri)

	# Create new sections
	logger.info("Creating equivalent sections...")
	eq_secs = redbush.equivalent_sections(clusters, allsecrefs, f_lambda, 
				use_segments=False, gbar_scaling='area', 
				interp_path=(1, (1,3,8)), interp_method='left_neighbor')

	############################################################################
	# 3. Finalize & Analyze
	
	# Delete original model sections & set ion styles
	for sec in h.allsec(): # makes each section the CAS
		if sec.name().startswith('SThcell') and delete_old_cells: # original model sections
			h.delete_section()
		else: # equivalent model sections
			h.ion_style("na_ion",1,2,1,0,1)
			h.ion_style("k_ion",1,2,1,0,1)
			h.ion_style("ca_ion",3,2,1,1,1)

	# Print tree structure
	logger.info("Equivalent tree topology:")
	if logger.getEffectiveLevel() >= logging.DEBUG:
		h.topology()

	return clusters, eq_secs

def reduce_gillies_pathRi(customclustering, average_Ri):
	""" Reduce Gillies & Willshaw STN neuron model

	To set active conductances, interpolates using axial path resistance
	values (sum(Ri)).

	@param customclustering		see param 'customclustering' in function
								cluster_sections()

	@param average_Ri			see param 'average_Ri' in function
								merge_sec_cluster
	"""

	# Initialize Gillies model
	for sec in h.allsec():
		if not sec.name().startswith('SThcell'): # original model sections
			h.delete_section()
	if not hasattr(h, 'SThcells'):
		h.xopen("createcell.hoc")

	# Make sections accesible by both name and index + allow to add attributes
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendLrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0]
	dendRrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1]
	alldendrefs = dendLrefs + dendRrefs
	allsecrefs = [somaref] + alldendrefs
	or_refs = (somaref, dendLrefs, dendRrefs)

	############################################################################
	# 0. Pre-clustering: calculate properties

	# Assign Strahler numbers
	logger.info("Assingling Strahler's numbers...")
	clutools.assign_strahler_order(dendLrefs[0], dendLrefs, 0)
	clutools.assign_strahler_order(dendRrefs[0], dendRrefs, 0)
	somaref.order = 0 # distance from soma
	somaref.strahlernumber = dendLrefs[0].strahlernumber # same as root of left tree

	############################################################################
	# 1. Cluster based on identified functional regions

	# Cluster sections
	logger.info("Clustering sections...")
	dendLroot, dendRroot = dendLrefs[0], dendRrefs[0]

	# Cluster soma
	somaref.cluster_label = 'soma'
	somaclu = Cluster('soma')
	somaclu.parent_label = 'soma'
	somaclu.parent_pos = 0.0
	somaclu.order = 0
	clusters = [somaclu]

	# Cluster dendritic trees
	if customclustering:
		# Based on diameters: see Gilles & Willshaw (2006) fig. 1
		clu_fun = lambda (ref): 'spiny' if ref.sec.diam <= 1.0 else 'trunk'
		clu_args = {}
	else:
		clu_fun = clutools.label_from_strahler
		clu_args = {'thresholds':(1,2)}

	clutools.clusterize_custom(dendLroot, allsecrefs, clusters, '_0', clu_fun, clu_args)
	clutools.clusterize_custom(dendRroot, allsecrefs, clusters, '_1', clu_fun, clu_args)

	# Determine cluster relations/topology
	clutools.assign_topology(clusters, ['soma', 'trunk', 'smooth', 'spiny'])

	# Debug info
	for cluster in clusters:
		logger.debug("Cluster '{0}'' has parent cluster '{1}'".format(cluster.label, cluster.parent_label))

	############################################################################
	# 2. Create equivalent sections (compute ppties from cluster sections)

	# Calculate cluster properties
	logger.info("Merging within-cluster sections...")
	for cluster in clusters:
		# Merge sections within each cluster: 
		merge_sec_cluster(cluster, allsecrefs, average_Ri)

		# Map axial path resistance values to gbar values
		map_pathri_gbar(cluster, allsecrefs)

	# Create equivalent section for each cluster
	logger.info("Creating equivalent sections...")
	eq_secs, eq_secrefs = equivalent_sections(clusters, allsecrefs, gradients=True)

	# Sort equivalent sections
	eq_sorted = sorted(eq_secrefs, key=lambda ref: label_order(ref.cluster_label))
	eq_somaref = next(ref for ref in eq_secrefs if ref.cluster_label.startswith('soma'))
	eq_dendLrefs = [ref for ref in eq_secrefs if ref.cluster_label.endswith('0')]
	eq_dendRrefs = [ref for ref in eq_secrefs if ref.cluster_label.endswith('1')]
	eq_somasec = eq_somaref.sec
	eq_dendLsecs = [ref.sec for ref in eq_dendLrefs]
	eq_dendRsecs = [ref.sec for ref in eq_dendRrefs]

	############################################################################
	# 3. Finalize & Analyze

	# Compare full/reduced model
	eq_secs = (eq_somasec, eq_dendLsecs, eq_dendRsecs)
	eq_refs = (eq_somaref, eq_dendLrefs, eq_dendRrefs)
	analysis.compare_models(or_refs, eq_refs, [])

	# Delete original model sections
	for sec in h.allsec(): # makes each section the CAS
		if sec.name().startswith('SThcell'): # original model sections
			h.delete_section()
		else: # equivalent model sections
			h.ion_style("na_ion",1,2,1,0,1)
			h.ion_style("k_ion",1,2,1,0,1)
			h.ion_style("ca_ion",3,2,1,1,1)

	logger.info("Equivalent tree topology:")
	if logger.getEffectiveLevel() >= logging.DEBUG:
		h.topology() # prints topology
	
	# return data structures
	return clusters, eq_secs, eq_refs

if __name__ == '__main__':
	clusters, eq_secs = reduce_gillies_pathLambda(segment_based=True, delete_old_cells=False)
	from neuron import gui