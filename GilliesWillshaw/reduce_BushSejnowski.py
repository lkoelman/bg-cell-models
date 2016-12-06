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

def merge_subtree(rootref, allsecrefs):
	""" 
	Recursively merge within-cluster connected sections in subtree
	of the given node using <br> and <seq> expressions in Marasco (2012).
	"""
	# Handle leaf sections
	rootsec = rootref.sec
	allchildrefs = [getsecref(sec, allsecrefs) for sec in rootsec.children()]
	childrefs = [ref in allchildrefs if ref.cluster_label==rootref.cluster_label]
	if not any(childrefs):
		return rootsec.L, rootsec.diam, rootsec.Ra, sum(seg.ri() for seg in rootsec)

	# use <br> expressions for L/rho/r_a in Marasco (2012) 
	# to combine properties of sibling sections at branch point
	L_br = 0
	diam_br = 0
	Ra_br = 0
	ri_br = 0
	eqsurf_sum = 0
	ri_sum = 0
	for childref in childrefs:
		# get equivalent child properties
		L_child, diam_child, Ra_child, ri_child = merge_subtree(childref, allsecrefs)
		eqsurf_child = PI*diam_child*L_child

		# combine according to <br> (parallel) expressions
		eqsurf_sum += eqsurf_child
		L_br += eqsurf_child*L_child # LENGTH: eq (1) - weight by area
		diam_br += diam_child**2 # RADIUS: eq (2) - 2-norm of radii
		Ra_br += Ra_child # SPECIFIC AXIAL RESISTANCE - average Ra
		ri_br *= ri_child # ABSOLUTE AXIAL RESISTANCE - parallel conductances

		# mark child as absorbed
		childref.absorbed = True

	# Finalize <br> calculation
	L_br /= eqsurf_sum
	diam_br = math.sqrt(diam_br)
	Ra_br = Ra_br/len(children) # average Ra
	ri_br /= ri_sum

	# use <seq> expressions in Marasco (2012) to merge equivalent child into parent
	L_seq = rootsec.L + L_br # L_seq equation
	Ra_seq = (rootsec.Ra + Ra_br)/2.
	ri_seq = sum(seg.ri() for seg in rootsec) + ri_br # r_a,seq equation
	diam_seq = math.sqrt(Ra_seq*L_seq*4./PI/ri_seq/100.) # rho_seq equation

def merge_islands(cluster, allsecrefs):
	""" 
	Merge within-cluster unconnected sections after connected subtrees 
	within cluster have been merged (absorbed into root of subtree).
	Merging is performed using <eq> expressions in Marasco (2012).
	"""

	# Gather sections for merging
	clu_secs = [secref for secref in secrefs if secref.cluster_label==cluster.label]
	mrg_secs = [secref for secref in clu_secs if not secref.absorbed]

	# Cluster properties computed from all its sections
	cluster.orSurfSum = sum(secref.sec.L*PI*secref.sec.diam for secref in clu_secs)

	# Combine properties of of unabsorbed sections using <eq> expressions
	cluster.eqL = 0.
	cluster.eqdiam = 0.
	cluster.eqri = 0.
	cluster.eqSurf = 0. # surface calculated equivalent dimensions
	cluster.eqSurfSum = 0. # sum of surface calculated from equivalent dimensions
	for i, secref in enumerate(mrg_secs):
		# Get equivalent properties for each island
		L_island = secref.mrgL
		diam_island = secref.mrgdiam
		ri_island = secref.mrgri
		eqSurf_island = secref.mrgL*PI*secref.mrgdiam

		# Update equivalent properties according to <eq> expressions
		cluster.eqSurfSum += eqSurf_island
		cluster.eqL += L_island*eqSurf_island
		cluster.eqdiam += diam_island**2
		cluster.eqri += ri_island

	# Finalize <eq> calculation
	cluster.eqL /= cluster.eqSurfSum # LENGTH: equation L_eq
	cluster.eqdiam = math.sqrt(cluster.eqdiam) # RADIUS: equation rho_eq
	cluster.eqri /= len(mrg_secs) # ABSOLUTE AXIAL RESISTANCE: equation r_a,eq
	cluster.eqRa = PI*(cluster.eqdiam**2)*cluster.eqri/cluster.eqL # SPECIFIC AXIAL RESISTANCE: equation R_a,eq
	cluster.eqSurf = cluster.eqL*PI*cluster.eqdiam # EQUIVALENT SURFACE
	
	return cluster

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

	# TODO: merge connected & unconnected

	# TODO: create equivalent sections

# if __name__ == '__main__':
# 	reduce_gillies()