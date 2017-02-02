"""
Analysis utilities for Marasco reduction

@author Lucas Koelman
@date	20-12-2016
"""

# Python modules
import math
import matplotlib.pyplot as plt

# Own modules
import reduction_tools
from reduction_tools import ExtSecRef, Cluster, getsecref

def plotconductance(secref, gname, titleattr):
	sec = secref.sec
	xg = [(seg.x, getattr(seg, gname)) for seg in sec]
	xvals, gvals = zip(*xg)

	plt.figure()
	plt.plot(xvals, gvals, 'o')
	plt.ylim(0, 1e-2)
	plt.xlabel('x')
	plt.ylabel(gname)
	plt.suptitle('Ion channel density in {}'.format(getattr(secref, titleattr)))

def plot_chan_dist(rootref, allsecrefs, gname, titleattr, plotted=None):
	""" Recursively plot conductances in each branch by descending the tree.
		Makes only one plot per cluster per order number (nb. secs from soma).
	@param rootref		SectionRef to root section
	@param titleattr	name of attribute on SectionRef to use as figure title
	"""
	if rootref is None: return
	if plotted is None: plotted={}

	# Plot root node
	clu_orders = plotted.setdefault(rootref.cluster_label, set())
	if rootref.order not in clu_orders:
		plotconductance(rootref, gname, titleattr)
		clu_orders.add(rootref.order)

	# plot children
	for childsec in rootref.sec.children():
		childref = getsecref(childsec, allsecrefs)
		plot_chan_dist(childref, allsecrefs, gname, titleattr, plotted)

def compare_models(or_secrefs, eq_secrefs, plot_glist):
	""" Compare model properties """
	somaref, dendLrefs, dendRrefs = or_secrefs[0], or_secrefs[1], or_secrefs[2]
	eq_somaref, eq_dendLrefs, eq_dendRrefs = eq_secrefs[0], eq_secrefs[1], eq_secrefs[2]

	# Compare input resistance of large/left tree 
	rootsecs = [dendLrefs[0].sec, eq_dendLrefs[0].sec, dendRrefs[0].sec, eq_dendRrefs[0].sec]
	Rin_DC = [reduction_tools.inputresistance_tree(sec, 0., 'gpas_STh') for sec in rootsecs]
	Rin_AC = [reduction_tools.inputresistance_tree(sec, 100., 'gpas_STh') for sec in rootsecs]
	
	print("\n=== INPUT RESISTANCE ===\
	\nLarge tree (Original model):\
	\nRin_AC={:.3f} \tRin_DC={:.3f}\
	\nLeft tree (Equivalent model):\
	\nRin_AC={:.3f} \tRin_DC={:.3f}\
	\n\
	\nSmall tree (Original model):\
	\nRin_AC={:.3f} \tRin_DC={:.3f}\
	\nRight tree (Equivalent model):\
	\nRin_AC={:.3f} \tRin_DC={:.3f}".format(
	Rin_AC[0], Rin_DC[0], Rin_AC[1], Rin_DC[1],
	Rin_AC[2], Rin_DC[2], Rin_AC[3], Rin_DC[3]))

	# Plot ion channel distribution
	for gname in plot_glist:
		print("\n=== FULL ion channel distribution ===")
		plot_chan_dist(somaref, dendLrefs+dendRrefs, gname, 'cluster_label')
		plt.show(block=True)

		print("\n=== REDUCED {} channel distribution ===".format(gname))
		plot_chan_dist(eq_somaref, eq_dendLrefs+eq_dendRrefs, gname, 'cluster_label')
		plt.show(block=True)