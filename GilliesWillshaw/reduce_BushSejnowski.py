"""
Reduce Gillies & Willshaw (2006) STN neuron model using the method
described in Bush & Sejnowski (1993)


@author Lucas Koelman
@date	5-12-2016
"""

# NEURON modules
import neuron
h = neuron.h

# Own modules
import reducemodel
import marasco_reduction as marasco
from marasco_reduction import ExtSecRef, Cluster

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

# if __name__ == '__main__':
# 	reduce_gillies()