"""
Analysis utilities for Marasco reduction

@author Lucas Koelman
@date	20-12-2016
"""

# Python modules
import math
import matplotlib.pyplot as plt

from neuron import h

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

	EXAMPLE
	fig, axes = plt.subplots(len(ionchans), 1, sharex=True) # one plot for each channel
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

def plot_path_ppty(propname, secfilter=None, labelfunc=None):
	""" Plot segment property along a given dendritic path in the full
		Gillies & Willshaw STN model

	@param	secfilter	function that maps a SectionRef to a bool indicating
						whether it should be plotted or not

	@param	labelfunc	function that maps a SectionRef to a label for the
						plot that is generated for this section
	"""
	# Initialize Gillies model
	if not hasattr(h, 'SThcells'):
		h.xopen("createcell.hoc")

	if secfilter is None:
		path_indices = (1,2,4,6,8) # longest path in tree
		secfilter = lambda secref: secref.table_index in path_indices
	
	if labelfunc is None:
		labelfunc = lambda secref: "sec {}".format(secref.table_index)

	# Make sections accesible by both name and index + allow to add attributes
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendLrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0]
	dendRrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1]
	allsecrefs = [somaref] + dendLrefs + dendRrefs

	# Assign indices
	somaref.table_index = 0
	for j, dendlist in enumerate((dendLrefs, dendRrefs)):
		for i, secref in enumerate(dendlist):
			secref.table_tree = j
			secref.table_index = i+1

	# Plot conductance along path
	plot_tree_ppty(somaref, [somaref]+dendLrefs, propname, secfilter, labelfunc)

def plot_tree_ppty(secref, allsecrefs, propname, secfilter, labelfunc, y_range=None, fig=None, ax=None):
	""" Descend the given dendritic tree start from the given root section
		and plot the given propertie.

	@param	secref		SectionRef to root section

	@param	allsecrefs	list(SectionRef) with references to all sections
						in tree

	@param	propname	segment property to plot

	@param	secfilter	filter function applied to each SectionRef
	"""
	if secref is None:
		return
	first_call = fig is None
	if first_call:
		fig = plt.figure()
	if y_range is None:
		y_range = [float('inf'), float('-inf')]

	# Plot current node
	if secfilter(secref):
		if not ax:
			nax = len(fig.axes)
			for i, ax in enumerate(fig.axes):
				ax.change_geometry(nax+1, 1, i+1) # change grid and position in grid
			ax = fig.add_subplot(nax+1, 1, nax+1)

		# get data to plot
		sec = secref.sec
		xg = [(seg.x, getattr(seg, propname)) for seg in sec]
		xvals, gvals = zip(*xg)
		if min(gvals) < y_range[0]:
			y_range[0] = min(gvals)
		if max(gvals) > y_range[1]:
			y_range[1] = max(gvals)

		# plot it
		ax.plot(xvals, gvals, 'o')
		ax.plot(xvals, gvals, '-')
		# ax.set_xlabel('x')
		ax.set_ylabel(labelfunc(secref))

		if y_range is not None:
			ax.set_ylim(y_range)

	# plot children
	for childsec in secref.sec.children():
		childref = getsecref(childsec, allsecrefs)
		plot_tree_ppty(childref, allsecrefs, propname, secfilter, labelfunc, y_range, fig)

	if first_call:
		plt.suptitle(propname)
		y_span = y_range[1]-y_range[0]
		for ax in fig.axes:
			ax.set_ylim((y_range[0]-0.1*y_span, y_range[1]+0.1*y_span))
		plt.show(block=False)
	return fig


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