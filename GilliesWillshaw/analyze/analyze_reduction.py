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
from reducemodel import redutils
from reducemodel.redutils import ExtSecRef, getsecref
import gillies_model
from reducemodel import reduce_cell


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
	"""
	Compare model properties
	"""
	
	somaref, dendLrefs, dendRrefs = or_secrefs[0], or_secrefs[1], or_secrefs[2]
	eq_somaref, eq_dendLrefs, eq_dendRrefs = eq_secrefs[0], eq_secrefs[1], eq_secrefs[2]

	# Compare input resistance of large/left tree 
	rootsecs = [dendLrefs[0].sec, eq_dendLrefs[0].sec, dendRrefs[0].sec, eq_dendRrefs[0].sec]
	Rin_DC = [redutils.inputresistance_tree(sec, 0., 'gpas_STh') for sec in rootsecs]
	Rin_AC = [redutils.inputresistance_tree(sec, 100., 'gpas_STh') for sec in rootsecs]
	
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


def inspect_passive_electrotonic_structure(reduced=True):
	"""
	Inspect passive electrotonic strucutre in NEURON GUI.

	After running this function, plot attenuation properties via Tools > Electrotonic Analysis
	"""
	if reduced:
		soma_refs, dend_refs = reduce_cell.fold_gillies_marasco(False)
		allsecrefs = soma_refs + dend_refs

	else:
		somaref, dendLrefs, dendRrefs = gillies_model.get_stn_refs()
		allsecrefs = [somaref] + dendLrefs + dendRrefs

	# Disable all active conductances (can also use 'uninsert' on all sections)
	gbar_active = [g for g in gillies_model.gillies_glist if (g != gillies_model.gleak_name)]

	sec_modified = 0
	seg_modified = 0

	# for sec in h.allsec():
	for secref in allsecrefs:
		sec = secref.sec
		secref.orig_range_props = redutils.get_range_props(secref, gbar_active)

		for seg in sec:
			for gbar in gbar_active:
				setattr(seg, gbar, 0.0)
			seg_modified += 1
		
		sec_modified += 1

	print("Set gbar to zero in {} sections and {} segments".format(sec_modified, seg_modified))

	from neuron import gui # opens GUI in another thread

if __name__ == '__main__':
	inspect_passive_electrotonic_structure()
