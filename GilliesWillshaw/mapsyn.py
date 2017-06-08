"""
Algorithms for mapping synaptic input locations from detailed compartmental neuron model
to reduced morphology model.
"""

import neuron
from neuron import h
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

import reduction_tools as redtools
from reduction_tools import ExtSecRef, getsecref, seg_index

class SynInfo(object):
	"""
	Data for constructing an equivalent synaptic input.
	(This is just a struct/bunch-like class)
	"""
	def __init__(self, **kwds):
		self.__dict__.update(kwds)

def get_syn_info(rootsec, syn_mod_pars=None):
	"""
	For each synapse on the neuron, calculate and save information for placing an equivalent
	synaptic input on a morphologically simplified neuron.

	@param rootsec		any section of the cell

	@param syn_mod_pars	dict of <synaptic mechanism name> : list(<attribute names>)
						containing attributes that need to be stored
	"""
	if syn_mod_pars is None:
		syn_mod_pars = {
			'ExpSyn': ['tau', 'e'],
			'Exp2Syn': ['tau1', 'tau2', 'e'],
			'AlphaSynapse': ['onset', 'tau', 'gmax', 'e'],
		}

	# Calculate section path properties for entire tree
	for secref in allsecrefs:
		redtools.sec_path_props(secref, 100., gillies.gleak_name)

	# Find all Synapses on cell (on any connected sections)
	dummy_syn = h.Exp2Syn(rootsec(0.5))
	dummy_nc = h.NetCon(None, dummy_syn)
	cell_nc = [nc for nc in list(dummy_nc.postcelllist()) if not nc.same(dummy_nc)] # all NetCon targetting the same cell
	cell_syns = set([nc.syn() for nc in cell_nc]) # unique synapses targeting the same cell

	# Save synapse properties
	# - attributes in syn_mod_pars
	# - section name/identifier
	for syn in cell_syns:
		# Get synaptic mechanism name
		match_mechname = re.search(r'^[a-zA-Z]+', syn.hname())
		synmech = match_mechname.group()
		if synmech not in syn_mod_pars:
			raise Exception("Synaptic mechanism '{}' not in given mechanism list".format(synmech))
		syn_seg = syn.get_segment()
		syn_sec = syn_seg.sec
		syn_secref = getsecref(syn_sec, allsecrefs)

		syn_info = SynInfo()
		syn_info.mech_name = synmech
		syn_info.sec_name = syn_sec.name()
		syn_info.sec_hname = syn_sec.hname()
		syn_info.sec_loc = syn.get_loc() # can also use nc.postcell() and nc.postloc()

		# Save synapse parameters
		mech_params = syn_mod_pars[synmech]
		for par in mech_params:
			setattr(syn_info, par, getattr(syn, par))

		# Calculate axial path resistance to synapse
		syn_info.path_ri = syn_secref.pathri_seg[seg_index(syn_seg)] # summed seg.ri() up to synapse segment
		syn_info.max_path_ri = max(syn_secref.pathri_seg) # max path resistance in Section
		syn_info.min_path_ri = min(syn_secref.pathri_seg) # min path resistance in Section

		# Measure transfer impedance and filter parameters

def map_synapses(rootsec, orig_syn_info):
	"""
	Map synapses to equivalent synaptic inputs on given morphologically
	reduced cell.

	@param rootsec			root section of reduced cell

	@param orig_syn_info	SynInfo for each synapse on original cell
	"""

	# Loop over all synapses

	# First find the section that the original section mapped to
	# TODO: when sections are collapsed: keep a list of original section names on the secref
	pass


def test_get_syn_info(export_locals=False):
	"""
	Test for function get_syn_info()

	@param export_locals	bring function local variables into global namespace
							for interactive testing
	"""
	# Make STN cell
	import gillies_model as gillies
	soma, dends, stims = gillies.stn_cell_gillies()
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendL_refs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0] # 0 is left tree
	dendR_refs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1] # 1 is right tree
	allsecrefs = [somaref] + dendLrefs + dendRrefs

	# Connect some synapses
	import numpy as np
	rgn = np.random.RandomState(15203008)
	test_syns = []
	test_nc = []

	# Connect a few synapses per dendritic section
	n_syn_sec = 3
	for dend_ref in [dendL_refs + dendR_refs]:
		dend_sec = dend_ref.sec
		# make synapses
		for i in xrange(n_syn_sec):
			syn_loc = rgn.rand()
			syn = h.ExpSyn(dend_sec(syn_loc))
			syn.tau = rgn.rand() * 20.
			syn.e = rng.rand() * 10.
			test_syns.append(syn)

			# make NetCon
			nc = hoc.NetCon(None, syn)
			nc.threshold = 0.
			nc.delay = 5.
			nc.weight[0] = rgn.rand() * 10.
			test_nc.append(nc)

	# Get synapse info
	syn_mod_pars = {
		'ExpSyn': ['tau', 'e'],
		'Exp2Syn': ['tau1', 'tau2', 'e'],
		'AlphaSynapse': ['onset', 'tau', 'gmax', 'e'],
	}
	# syn_info = get_syn_info()

	if export_locals:
		globals().update(locals())

if __name__ == '__main__':
	test_get_syn_info(export_locals=True)