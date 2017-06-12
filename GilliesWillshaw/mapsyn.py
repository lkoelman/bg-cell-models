"""
Algorithms for mapping synaptic input locations from detailed compartmental neuron model
to reduced morphology model.
"""

import re
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module

import neuron
from neuron import h
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

import gillies_model as gillies
import reduction_tools as redtools
from reduction_tools import ExtSecRef, getsecref, seg_index

class SynInfo(object):
	"""
	Data for constructing an equivalent synaptic input.
	(This is just a struct/bunch-like class)
	"""
	def __init__(self, **kwds):
		self.__dict__.update(kwds)

def get_syn_info(rootsec, allsecrefs, syn_mod_pars=None, 
					Z_freq=25., initcell_fun=None, secref_attrs=None):
	"""
	For each synapse on the neuron, calculate and save information for placing an equivalent
	synaptic input on a morphologically simplified neuron.

	@param rootsec		any section of the cell

	@param syn_mod_pars	dict of <synaptic mechanism name> : list(<attribute names>)
						containing attributes that need to be stored

	@param initcell_fun	function to bring the cell to the desired state to measure transfer
						impedances, e.g. simulating under a particular input
	"""
	if syn_mod_pars is None:
		syn_mod_pars = {
			'ExpSyn': ['tau', 'e'],
			'Exp2Syn': ['tau1', 'tau2', 'e'],
			'AlphaSynapse': ['onset', 'tau', 'gmax', 'e'],
		}

	if secref_attrs is None:
		secref_attrs = []

	# Calculate section path properties for entire tree
	for secref in allsecrefs:
		redtools.sec_path_props(secref, 100., gillies.gleak_name)

	# Measure transfer impedance and filter parameters
	imp = h.Impedance()
	imp.loc(0.5, sec=rootsec)

	# Make sure cell is in a physiological state
	logger.debug("Running simulation function...")
	initcell_fun() # TODO: save times before, during, after burst and measure at these times

	# Compute impedances at current state of cell
	linearize_gating = 0 # 0 = calculation with current values of gating vars, 1 = linearize gating vars around V
	imp.compute(Z_freq, linearize_gating) # compute transfer impedance between loc and all segments

	# Find all Synapses on cell (all Sections in whole tree)
	dummy_syn = h.Exp2Syn(rootsec(0.5))
	dummy_nc = h.NetCon(None, dummy_syn)
	cell_nc = [nc for nc in list(dummy_nc.postcelllist()) if not nc.same(dummy_nc)] # all NetCon targetting the same cell
	cell_syns = set([nc.syn() for nc in cell_nc]) # unique synapses targeting the same cell
	logger.debug("Found {} NetCon with {} unique synapses".format(len(cell_nc), len(cell_syns)))

	# Save synapse properties
	logger.debug("Getting synapse properties...")
	syn_data = []
	for syn in cell_syns:
		# Get synaptic mechanism name
		match_mechname = re.search(r'^[a-zA-Z]+', syn.hname())
		synmech = match_mechname.group()
		if synmech not in syn_mod_pars:
			raise Exception("Synaptic mechanism '{}' not in given mechanism list".format(synmech))

		syn_seg = syn.get_segment()
		syn_sec = syn_seg.sec
		syn_secref = getsecref(syn_sec, allsecrefs)
		syn_loc = syn.get_loc() # changes CAS
		h.pop_section()

		syn_info = SynInfo()
		syn_info.mod_name = synmech
		syn_info.sec_name = syn_sec.name()
		syn_info.sec_hname = syn_sec.hname()
		syn_info.sec_loc = syn_loc # can also use nc.postcell() and nc.postloc()

		# Save synapse parameters
		mech_params = syn_mod_pars[synmech]
		for par in mech_params:
			setattr(syn_info, par, getattr(syn, par))

		# Save requested properties of synapse section
		for attr in secref_attrs:
			setattr(syn_info, attr, getattr(syn_secref, attr))

		# Get axial path resistance to synapse
		syn_info.path_ri = syn_secref.pathri_seg[seg_index(syn_seg)] # summed seg.ri() up to synapse segment
		syn_info.max_path_ri = max(syn_secref.pathri_seg) # max path resistance in Section
		syn_info.min_path_ri = min(syn_secref.pathri_seg) # min path resistance in Section

		# Get transfer impedances
		syn_info.Z_transfer = imp.transfer(syn_loc, sec=syn_sec) # query transfer impedanc,e i.e.  |v(loc)/i(x)| or |v(x)/i(loc)|
		syn_info.Z_input = imp.input(syn_loc, sec=syn_sec) # query input impedance, i.e. v(x)/i(x)
		syn_info.V_ratio = imp.ratio(syn_loc, sec=syn_sec) # query voltage transfer ratio, i.e. |v(loc)/v(x)|

		syn_data.append(syn_info)
	return syn_data

def subtree_meets_crit(noderef, allsecrefs, crit_func):
	"""
	Check if the given function applies to (returns True) any node
	in subtree of given node
	"""
	if noderef is None:
		return False
	elif crit_func(noderef):
		return True
	else:
		childsecs = noderef.sec.children()
		childrefs = [getsecref(sec, allsecrefs) for sec in childsecs]
		for childref in childrefs:
			if subtree_meets_crit(childref, allsecrefs, crit_func)
				return True
		return False

def map_synapse(noderef, allsecrefs, syn_info, imp):
	"""
	Map synapse to a section in subtree of noderef
	"""
	cur_sec = cur_node.sec
	Zc = syn_info.Z_transfer
	Zc_0 = imp.transfer(0.0, sec=cur_sec)
	Zc_1 = imp.transfer(1.0, sec=cur_sec)

	if (Zc_0 <= Zc <= Zc_1) or (Zc_1 <= Zc <= Zc_0):
		# Calculate Zc at midpoint of each internal segment
		segs_loc_Zc = [(seg.x, imp.transfer(seg.x, sec=cur_sec)) for seg in cur_sec]
		Zc_diffs = [abs(Zc-pts[1]) for pts in segs_loc_Zc]

		# Map synapse with closest Zc at midpoint
		seg_index = Zc_diffs.index(min(Zc_diffs))
		x_map = segs_loc_Zc[seg_index][0]
		return noderef, x_map	
	else:
		childsecs = noderef.sec.children()
		childrefs = [getsecref(sec, allsecrefs) for sec in childsecs]

		# If we are in correct tree but Zc smaller than endpoints, return endpoint
		if not any(child_refs):
			assert Zc < Zc_0 and Zc < Zc_1
			return noderef, 1.0

		# Else, recursively search child nodes
		for childref in childrefs:
			if subtree_meets_crit(childref, allsecrefs, crit_func)
				return map_synapse(childref, allsecrefs, syn_info, imp)
		raise Exception("The synapse did not map onto any segment in this subtree.")



def map_synapses(rootref, allsecrefs, orig_syn_info, initcell_fun, Z_freq):
	"""
	Map synapses to equivalent synaptic inputs on given morphologically
	reduced cell.

	@param rootsec			root section of reduced cell

	@param orig_syn_info	SynInfo for each synapse on original cell
	"""

	# Class for measuring transfer impedances
	imp = h.Impedance()
	imp.loc(0.5, sec=rootsec)
	initcell_fun()

	# Compute impedances at current state of cell
	linearize_gating = 0 # 0 = calculation with current values of gating vars, 1 = linearize gating vars around V
	imp.compute(Z_freq, linearize_gating) # compute transfer impedance between loc and all segments

	# Loop over all synapses
	for syn_info in orig_syn_info:

		# Find the segment with same tree index and closest Ztransfer match,
		map_ref, map_x = map_synapse(rootref, allsecrefs, syn_info, imp)

		# Make the synapse
		syn_mod = syn_info.mod_name
		synmech_ctor = getattr(h, syn_mod) # constructor function for synaptic mechanism
		mapped_syn = synmech_ctor(map_ref.sec(map_x))

	# TODO: (equation 10-11 in Jaffe & Carnevale 1999: Vsoma = k_{syn->soma}*Zin_{syn}*gsyn(V-Esyn) , with k=Zc/Zin
	#       Use this relation to position the synapse.
	#		The free parameters are where to place it (determining k & Zin) and gsyn.
	#		E.g. position @ loc with same k (which should show greates variability)
	#		then use Zc (=k*Zin) or measure Zin to scale gsyn to satisfy constraint of same Vsoma.
	
	# NOTE: first plot these properties in Impedance tool, and compare them between full and reduced cells

	# First find the section that the original section mapped to,
	# then move to parent or child direction to find location with same v transfer ratio,
	# TODO: when sections are collapsed: keep a list of original section names on the secref
	pass


def test_map_synapses(export_locals=False):
	"""
	Test for function get_syn_info()

	@param export_locals	bring function local variables into global namespace
							for interactive testing
	"""
	# Make STN cell
	soma, dends, stims = gillies.stn_cell_gillies()
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendL_refs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0] # 0 is left tree
	dendR_refs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1] # 1 is right tree
	allsecrefs = [somaref] + dendL_refs + dendR_refs

	# Connect some synapses
	import numpy as np
	rng = np.random.RandomState(15203008)
	test_syns = []
	test_nc = []

	# Connect a few synapses per dendritic section
	n_syn_sec = 3
	for dend_ref in (dendL_refs + dendR_refs):
		dend_sec = dend_ref.sec
		# make synapses
		for i in xrange(n_syn_sec):
			syn_loc = rng.rand()
			syn = h.ExpSyn(dend_sec(syn_loc))
			syn.tau = rng.rand() * 20.
			syn.e = rng.rand() * 10.
			test_syns.append(syn)

			# make NetCon
			nc = h.NetCon(None, syn)
			nc.threshold = 0.
			nc.delay = 5.
			nc.weight[0] = rng.rand() * 10.
			test_nc.append(nc)

	# Make function to run simulation and bring cell to desired state
	def stn_setstate():
		h.celsius = 35
		h.set_aCSF(4)
		h.v_init = -60
		h.tstop = 460
		h.dt = 0.025
		h.init()
		h.run()

	# Get synapse info
	secref_attrs = ['table_index', 'tree_index']
	syn_info = get_syn_info(soma, allsecrefs, 
					impedances=True, initcell_fun=stn_setstate)

	# TODO: Create reduced cells
	import reduce_marasco as marasco
	eq_secs, newsecrefs = marasco.reduce_gillies_incremental(n_passes=7, zips_per_pass=100)

	# TODO: Map synapses to reduced cell

	if export_locals:
		globals().update(locals())

if __name__ == '__main__':
	test_map_synapses(export_locals=True)