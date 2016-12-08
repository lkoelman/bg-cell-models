"""
Reduce Gillies & Willshaw (2006) STN neuron model using the method
described in Bush & Sejnowski (1993)


@author Lucas Koelman
@date	5-12-2016
"""

# Python modules
import math
PI = math.pi

# NEURON modules
import neuron
h = neuron.h

# Own modules
import reducemodel
import marasco_reduction as marasco
from marasco_reduction import ExtSecRef, Cluster, getsecref

# Global variables (convert to class members in future)
mechs_chans = {'STh': ['gpas'], # passive/leak channel
				'Na': ['gna'], 'NaL': ['gna'], # Na channels
				'KDR': ['gk'], 'Kv31': ['gk'], 'sKCa':['gk'], # K channels
				'Ih': ['gk'], # nonspecific channels
				'CaT': ['gcaT'], 'HVA': ['gcaL', 'gcaN'], # Ca channels
				'Cacum': []} # No channels
glist = [gname+'_'+mech for mech,chans in mechs_chans.iteritems() for gname in chans]

def collapse_subtree(rootref, allsecrefs):
	""" 
	Recursively merge within-cluster connected sections in subtree
	of the given node using <br> and <seq> expressions in Marasco (2012).
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
	cmtot_sum = sum(seg.cm*seg.area() for seg in rootsec) # sum of cm multiplied by area
	gtot_sum = dict((gname, 0.0) for gname in glist) # sum of gbar multiplied by area
	for gname in glist:
		gtot_sum[gname] += sum(getattr(seg, gname)*seg.area() for seg in rootsec)

	# Handle leaf sections
	if not any(childrefs):
		return L_root, diam_root, Ra_root, ri_root, cmtot_sum, gtot_sum

	# Initialize combined properties of branched sections (children)
	L_br = 0.
	diam_br = 0.
	Ra_br = 0.
	ri_br = 1.
	eqsurf_sum = 0.
	ri_sum = 0.
	gtot_sum = dict((gname, 0.0) for gname in glist) # sum of gbar multiplied by area
	cmtot_sum = 0. # sum of cm multiplied by area

	# Update combined properties using the dimensions/electrical properties
	# of each child according to <br> expressions in Marasco (2012) 
	for childref in childrefs:
		# get equivalent child properties
		L_child, diam_child, Ra_child, ri_child, cmtot_child, gtot_child = collapse_subtree(childref, allsecrefs)
		eqsurf_child = PI*diam_child*L_child

		# combine according to <br> (parallel) expressions
		eqsurf_sum += eqsurf_child
		L_br += eqsurf_child*L_child # LENGTH: eq (1) - weight by area
		diam_br += diam_child**2 # RADIUS: eq (2) - 2-norm of radii
		Ra_br += Ra_child # SPECIFIC AXIAL RESISTANCE - average Ra
		ri_br *= ri_child # ABSOLUTE AXIAL RESISTANCE - parallel conductances
		ri_sum += ri_child # need sum in final calculation

		# Distributed properties
		cmtot_sum += cmtot_child
		for gname in glist:
			gtot_sum[gname] += gtot_child[gname]

		# mark child as absorbed
		childref.absorbed = True
		childref.visited = True

	# Finalize <br> calculation
	L_br /= eqsurf_sum
	diam_br = math.sqrt(diam_br)
	Ra_br = Ra_br/len(childrefs) # average Ra
	ri_br /= ri_sum

	# use <seq> expressions in Marasco (2012) to merge equivalent child into parent
	L_seq = L_root + L_br # L_seq equation
	Ra_seq = (Ra_root + Ra_br)/2.
	ri_seq = ri_root + ri_br # r_a,seq equation
	diam_seq = math.sqrt(Ra_seq*L_seq*4./PI/ri_seq/100.) # rho_seq equation

	return L_seq, diam_seq, Ra_seq, ri_seq, cmtot_sum, gtot_sum

def merge_cluster(cluster, allsecrefs):
	"""
	Merge sections in cluster

	ALGORITHM
	- find the next root of a within-cluster connected subtree
		- i.e. a section without a parent in the same cluster
		- this can be an isolated sections (no mergeable children within cluster)
	- collapse subtree of that root section
	- update equivalent cluster properties using the <eq> expressions in Marasco (2012)

	"""
	# Gather sections and mark/flag them
	clu_secs = [secref for secref in allsecrefs if secref.cluster_label==cluster.label]
	for sec in clu_secs:
		sec.absorbed = False
		sec.visited = False

	# Calculate min/max path resistance in cluster (full model)
	for sec in clu_secs:
		marasco.calc_path_ri(sec) # assigns pathri0/pathri1
	cluster.orMaxpathri = max(secref.pathri1 for secref in clu_secs)
	cluster.orMinpathri = min(secref.pathri0 for secref in clu_secs)

	# Initialize equivalent properties
	cluster.eqL = 0.
	cluster.eqdiam = 0.
	cluster.eqri = 0.
	cluster.orSurfSum = sum(sum(seg.area() for seg in secref.sec) for secref in clu_secs)
	cluster.eqSurfSum = 0. # sum of surface calculated from equivalent dimensions
	cluster.cmtot_sum = 0.
	cluster.gtot_sum = dict((gname, 0.0) for gname in glist)

	# utility function to check if sec has parent within cluster
	def has_clusterparent(secref):
		return secref.has_parent() and (getsecref(secref.parent, clu_secs) is not None)

	# Find connected subtrees within cluster and merge/collapse them
	rootfinder = (sec for sec in clu_secs if (not sec.visited and not has_clusterparent(sec))) # compiles generator function
	for secref in rootfinder:
		rootref = marasco.clusterroot(secref, clu_secs) # make sure it is a cluster root

		# Collapse subtree
		L_eq, diam_eq, Ra_eq, ri_eq, cmtot_eq, gtot_eq = collapse_subtree(rootref, allsecrefs)

		# Combine properties of collapse sections using <eq> expressions
		surf_eq = L_eq*PI*diam_eq
		cluster.eqSurfSum += surf_eq
		cluster.eqL += L_eq * surf_eq
		cluster.eqdiam += diam_eq**2
		cluster.eqri += ri_eq

		# Save distributed properties
		cluster.cmtot_sum += cmtot_eq
		for gname in glist:
			cluster.gtot_sum[gname] += gtot_eq[gname]

		# Mark as visited
		rootref.visited = True

	# Check each section either absorbed or rootsec
	assert not any(not sec.absorbed and has_clusterparent(sec) for sec in clu_secs), (
			'Each section should be either absorbed or be a root within the cluster')

	# Finalize <eq> calculation
	cluster.eqL /= cluster.eqSurfSum # LENGTH: equation L_eq
	cluster.eqdiam = math.sqrt(cluster.eqdiam) # RADIUS: equation rho_eq
	cluster.eqri /= sum(not sec.absorbed for sec in clu_secs) # ABSOLUTE AXIAL RESISTANCE: equation r_a,eq
	cluster.eqRa = PI*(cluster.eqdiam**2)*cluster.eqri*100./cluster.eqL # SPECIFIC AXIAL RESISTANCE: equation R_a,eq
	cluster.eqSurf = cluster.eqL*PI*cluster.eqdiam # EQUIVALENT SURFACE

def lambda_f(f, diam, Ra, cm):
	""" Compute electrotonic length (taken from stdlib.hoc) """
	return 1e5*math.sqrt(diam/(4*math.pi*f*Ra*cm))

def min_nseg_hines(sec):
	""" Minimum number of segments based on electrotonic length """
	return int(sec.L/(0.1*lambda_f(100., sec.diam, sec.Ra, sec.cm))) + 1

def min_nseg_marasco(sec):
	""" Minimum number of segments based on electrotonic length """
	return int((sec.L/(0.1*lambda_f(100., sec.diam, sec.Ra, sec.cm))+0.9)/2)*2 + 1  

def equivalent_sections(clusters, allsecrefs):
	""" Create the reduced/equivalent cell by creating 
		a section for each cluster 

	@param clusters		list of Cluster objects containing data
						for each cluster
	@param allsecrefs	list of SectionRef (mutable) with first element
						a ref to root/soma section
	@return				list of SectionRef containing equivalent Section 
						for each cluster (in same order) as well as min
						and max path resistance for each cluster/section
						as properties pathri0/pathri1 on SectionRef objects
	"""
	# Create equivalent section for each clusters
	eq_secs = [h.Section() for clu in eq_clusters]
	eq_secrefs = [ExtSecRef(sec=sec) for sec in eq_secs]

	# Connect sections
	for i, clu_i in enumerate(eq_clusters):
		for j, clu_j in enumerate(eq_clusters):
			if clu_j is not clu_i and clu_j.parent_label == clu_i.label:
				eq_secs[j].connect(eq_secs[i], clu_j.parent_pos, 0)

	# Set dimensions, passive properties, active properties
	for i, sec in enumerate(eq_secs):
		sec.push() # Make section the CAS

		# Set geometry 
		sec.L = clusters[i].eqL
		sec.diam = clusters[i].eqdiam
		sec_area = sum(seg.area() for seg in sec) # should be same as cluser eqSurf
		surf_fact = clusters[i].orSurfSum/clusters[i].eqSurf # scale factor: ratio areas original/equivalent

		# Passive electrical properties (except Rm/gleak)
		sec.cm = clusters[i].cmtot_sum / sec_area
		sec.Ra = clusters[i].eqRa

		# Set number of segments based on rule of thumb electrotonic length
		sec.nseg = min_nseg_hines(sec)

		# Insert all mechanisms and set conductances (TODO: incorporate gradients)
		for mech in mechs_chans.keys():
			sec.insert(mech)
		for gname in glist:
			for seg in sec:
				gval = clusters[i].gtot_sum[gname] / sec_area # same as divided by eqSurf
				sec.__setattr__(gname, gval)
		
		# calculate min/max path resistance in cluster (equivalent model)
		marasco.calc_path_ri(eq_secrefs[i])

		# Unset CAS
		h.pop_section()

	return eq_secs, eq_secrefs # return both or secs will be deleted

# def reduce_gillies():
if __name__ == '__main__':
	""" Reduce Gillies & Willshaw STN neuron model """

	# Initialize Gillies model
	h.xopen("createcell.hoc")

	# Make sections accesible by both name and index + allow to add attributes
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendLrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0]
	dendRrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1]
	alldendrefs = dendLrefs + dendRrefs
	allsecrefs = [somaref] + alldendrefs

	# Assign Strahler numbers
	marasco.assign_strahler_order(dendLrefs[0], dendLrefs, 0)
	marasco.assign_strahler_order(dendRrefs[0], dendRrefs, 0)
	somaref.order = 0 # distance from soma
	somaref.strahlernumber = dendLrefs[0].strahlernumber # same as root of left tree

	# Cluster subtree of each trunk section
	marasco.clusterize_strahler_trunks(allsecrefs, thresholds=(1,2))
	cluster_labels = list(set(secref.cluster_label for secref in allsecrefs)) # unique labels
	eq_clusters = [Cluster(label) for label in cluster_labels]

	# Determine cluster topology
	cluster_relations = set()
	cluster_relations.add(('soma', 'soma')) # soma is own parent
	marasco.cluster_topology(somaref, allsecrefs, cluster_relations)
	for cluster in eq_clusters:
		cluster.parent_pos = 1.
		cluster.parent_label = next(rel[0] for rel in cluster_relations if rel[1]==cluster.label)

	# Merge sections within each cluster: 
	# i.e. calculate properties of equivalent section for each cluster
	for cluster in eq_clusters:
		merge_cluster(cluster, allsecrefs) # stores equivalent properties in cluster

	# Create equivalent section for each cluster
	eq_secs, eq_secrefs = equivalent_sections(eq_clusters, allsecrefs)

	# Delete original model sections
	for sec in h.allsec(): # makes each section the CAS
		if sec.name().startswith('SThcell'):
			h.delete_section()


# if __name__ == '__main__':
# 	reduce_gillies()