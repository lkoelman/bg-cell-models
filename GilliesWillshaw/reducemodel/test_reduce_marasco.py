"""
Test suite for Marasco reduction module

@author Lucas Koelman
@date	14-12-2016
"""

# Python modules
import math

# NEURON modules
import neuron
h = neuron.h

# Own modules
import reduction_tools
import reduce_marasco as marasco
from reduction_tools import ExtSecRef, getsecref
from cluster import Cluster

def test_Rin_Ycluster():
	"""
	Test input resistance of simple tree before & after reduction
	"""
	# Child section A
	asec = h.Section()
	asec.L = 50.
	asec.diam = 5.
	asec.Ra = 35.
	asec.insert('hh')
	asec.gl_hh = 3e-4
	asec.nseg = 5

	# Child section B
	bsec = h.Section()
	bsec.L = 25.
	bsec.diam = 8.
	bsec.Ra = 35.
	bsec.insert('hh')
	bsec.gl_hh = 3e-4
	bsec.nseg = 5

	# Parent section P
	psec = h.Section()
	psec.L = 40.
	psec.diam = 10.
	psec.Ra = 35.
	psec.insert('hh')
	psec.gl_hh = 3e-4
	psec.nseg = 5

	# Connect them in a tree
	somasec = h.Section()
	somasec.insert('hh')
	asec.connect(psec, 1, 0)
	bsec.connect(psec, 1, 0)
	psec.connect(somasec, 1, 0)

	# Compute input resistances
	lambA = reduction_tools.sec_lambda(asec, asec.gl_hh, 0.)
	RinA = reduction_tools.inputresistance_sealed(asec, asec.gl_hh, 0.)
	lambB = reduction_tools.sec_lambda(bsec, bsec.gl_hh, 0.)
	RinB = reduction_tools.inputresistance_sealed(bsec, bsec.gl_hh, 0.)
	R_end = 1./(1./RinA	+ 1./RinB)
	RinP = reduction_tools.inputresistance_leaky(psec, psec.gl_hh, 0., R_end)

	# Do marasco Reduction
	marasco.mechs_chans = {'hh': ['gnabar', 'gkbar', 'gl']}

	# Set up pre-clustered example
	allsecrefs = [ExtSecRef(sec=sec) for sec in [somasec, psec, asec, bsec]]
	for ref in allsecrefs:
		ref.cluster_label = 'trunk'
		ref.parent_label = 'soma'
		ref.parent_pos = 1.0
	allsecrefs[0].cluster_label = 'soma'
	cluster = Cluster('trunk')

	# Merge cluster
	marasco.merge_sec_cluster(cluster, allsecrefs, True)

	# Create equivalent sections
	eq_secs, eq_secrefs = marasco.equivalent_sections([cluster], allsecrefs, gradients=False)
	csec = eq_secs[0]

	# Compute inputresistance
	RinC = reduction_tools.inputresistance_sealed(csec, csec.gl_hh, 0.)


if __name__ == '__main__':
	test_Rin_Ycluster()