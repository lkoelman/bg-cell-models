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

	def __repr__(self):
		return "{} at {}({})".format(self.mod_name, self.sec_hname, self.sec_loc)

def get_syn_info(rootsec, allsecrefs, syn_mod_pars=None, 
					Z_freq=25., initcell_fun=None, save_ref_attrs=None):
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
			'ExpSyn': ['tau', 'g', 'e'],
			'Exp2Syn': ['tau1', 'tau2', 'g', 'e'],
			'AlphaSynapse': ['onset', 'tau', 'gmax', 'e'],
		}

	if save_ref_attrs is None:
		save_ref_attrs = []

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
	if len(cell_syns) == 0:
		logger.warn("No synapses found on tree of Section {}".format(rootsec))
	logger.debug("Found {} NetCon with {} unique synapses".format(len(cell_nc), len(cell_syns)))

	# Save synapse properties
	logger.debug("Getting synapse properties...")
	syn_data = []
	for syn in cell_syns:
		# Get synaptic mechanism name
		match_mechname = re.search(r'^[a-zA-Z0-9]+', syn.hname())
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
		syn_info.saved_ref_attrs = save_ref_attrs
		for attr in save_ref_attrs:
			setattr(syn_info, attr, getattr(syn_secref, attr))

		# Get axial path resistance to synapse
		syn_info.path_ri = syn_secref.pathri_seg[seg_index(syn_seg)] # summed seg.ri() up to synapse segment
		syn_info.max_path_ri = max(syn_secref.pathri_seg) # max path resistance in Section
		syn_info.min_path_ri = min(syn_secref.pathri_seg) # min path resistance in Section

		# Get transfer impedances
		syn_info.Zc = imp.transfer(syn_loc, sec=syn_sec) # query transfer impedanc,e i.e.  |v(loc)/i(x)| or |v(x)/i(loc)|
		syn_info.Zin = imp.input(syn_loc, sec=syn_sec) # query input impedance, i.e. v(x)/i(x)
		syn_info.k_syn_soma = imp.ratio(syn_loc, sec=syn_sec) # query voltage transfer ratio, i.e. |v(loc)/v(x)|

		syn_data.append(syn_info)
	return syn_data

def subtree_has_node(crit_func, noderef, allsecrefs):
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
			if subtree_has_node(crit_func, childref, allsecrefs):
				return True
		return False

def map_synapse(noderef, allsecrefs, syn_info, imp, passed_synsec=False):
	"""
	Map synapse to a section in subtree of noderef
	"""
	cur_sec = noderef.sec
	Zc = syn_info.Zc
	Zc_0 = imp.transfer(0.0, sec=cur_sec)
	Zc_1 = imp.transfer(1.0, sec=cur_sec)

	# Assume monotonically decreasing Zc away from root section
	if (Zc_0 <= Zc <= Zc_1) or (Zc_1 <= Zc <= Zc_0) or (Zc >= Zc_0 and Zc >= Zc_1):

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
		if not any(childrefs):
			assert Zc < Zc_0 and Zc < Zc_1
			return noderef, 1.0

		# Else, recursively search child nodes
		# Function for checking that section maps to original synapse section
		mapsto_synsec = lambda ref: (ref.tree_index==syn_info.tree_index) and (
									ref.gid==syn_info.gid or (
										hasattr(ref, 'zipped_sec_gids') and 
										(syn_info.gid in ref.zipped_sec_gids)
										)
									)
		passed_synsec = passed_synsec or mapsto_synsec(noderef)
		# child_mapsto_synsec = [subtree_has_node(mapsto_synsec, ref, allsecrefs) for ref in childrefs]
		# passed_synsec = not any(child_mapsto_synsec)
		# if passed_synsec:
		# 	assert mapsto_synsec(noderef) # assume that synapse was positioned on this Section

		for childref in childrefs:
			if passed_synsec or subtree_has_node(mapsto_synsec, childref, allsecrefs):
				return map_synapse(childref, allsecrefs, syn_info, imp, passed_synsec)
		raise Exception("The synapse did not map onto any segment in this subtree.")



def map_synapses(rootref, allsecrefs, orig_syn_info, initcell_fun, Z_freq, syn_mod_pars=None):
	"""
	Map synapses to equivalent synaptic inputs on given morphologically
	reduced cell.

	@param rootsec			root section of reduced cell

	@param orig_syn_info	SynInfo for each synapse on original cell

	@effect					Create one synapse for each original synapse in 
							orig_syn_info. A reference to this synapse is saved 
							as as attribute 'mapped_syn' on the SynInfo object.
	"""
	# Synaptic mechanisms
	if syn_mod_pars is None:
		syn_mod_pars = {
			'ExpSyn': ['tau', 'g', 'e'],
			'Exp2Syn': ['tau1', 'tau2', 'g', 'e'],
			'AlphaSynapse': ['onset', 'tau', 'gmax', 'e'],
		}

	# Compute transfer impedances
	imp = h.Impedance()
	imp.loc(0.5, sec=rootref.sec)
	initcell_fun()
	linearize_gating = 0 # 0 = calculation with current values of gating vars, 1 = linearize gating vars around V
	imp.compute(Z_freq, linearize_gating) # compute transfer impedance between loc and all segments

	# Loop over all synapses
	logger.debug("Mapping synapses to reduced cell...")
	for syn_info in orig_syn_info:

		# Find the segment with same tree index and closest Ztransfer match,
		map_ref, map_x = map_synapse(rootref, allsecrefs, syn_info, imp)
		map_sec = map_ref.sec
		logger.debug("Synapse was in {}({}) -> mapped to {}\n".format(
						syn_info.sec_name, syn_info.sec_loc, map_sec(map_x)))

		# Make the synapse
		syn_mod = syn_info.mod_name
		synmech_ctor = getattr(h, syn_mod) # constructor function for synaptic mechanism
		mapped_syn = synmech_ctor(map_sec(map_x))
		syn_info.mapped_syn = mapped_syn

		# Copy synapse properties
		mech_params = syn_mod_pars[syn_mod]
		for par in mech_params:
			val = getattr(syn_info, par)
			setattr(mapped_syn, par, val)

		gsyn_attr = next((p for p in mech_params if p.startswith('g')))

		# Ensure synaptic conductance is scaled correctly to produce same effect at soma
		# TODO: if cannt match Zc, measure Zc (=k*Zin) and adjust g
		# TODO: change get_syn_info to save weight instead of g (=/=param!) on syn_info (see syn mod file for meaning)
		# TODO: save gsyn_mapped on the syn_info, then later set NetCon.weight to this value
		map_Zc = imp.transfer(map_x, sec=map_sec) # query transfer impedanc,e i.e.  |v(loc)/i(x)| or |v(x)/i(loc)|
		map_Zin = imp.input(map_x, sec=map_sec) # query input impedance, i.e. v(x)/i(x)
		map_k = imp.ratio(map_x, sec=map_sec) # query voltage transfer ratio, i.e. |v(loc)/v(x)|
		
		gsyn_orig = getattr(syn_info, gsyn_attr)
		logger.debug("Original synapse:\nZc={}\tZin={}\tk={}\nk*Zin*gsyn={}\nZc*gsyn={}\n".format(
						syn_info.Zc, syn_info.Zin, syn_info.k_syn_soma, 
						syn_info.k_syn_soma*syn_info.Zin*gsyn_orig,
						syn_info.Zc*gsyn_orig))

		# NOTE: if synapse positioned using k, would need to measure Zin and rescale gbar,
		#		but if positioned using Zc no scaling is required
		gsyn_mapped = getattr(mapped_syn, gsyn_attr)
		logger.debug("Mapped synapse:\nZc={}\tZin={}\tk={}\nk*Zin*gsyn={}\nZc*gsyn={}\n".format(
						map_Zc, map_Zin, map_k, 
						map_k*map_Zin*gsyn_mapped,
						map_Zc*gsyn_mapped))
