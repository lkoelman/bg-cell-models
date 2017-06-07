"""
Algorithms for mapping synaptic input locations from detailed compartmental neuron model
to reduced morphology model.
"""

class SynData(object):
	"""
	Data for constructing an equivalent synaptic input.
	(This is just a struct/bunch-like class)
	"""
	def __init__(self, **kwds):
		self.__dict__.update(kwds)

def get_syn_info(rootsec, syn_attrs):
	"""
	For each synapse on the neuron, calculate and save information for placing an equivalent
	synaptic input on a morphologically simplified neuron.

	@param rootsec		any section of the cell

	@param syn_attrs	dict of <synaptic mechanism name> : list(<attribute names>)
						containing attributes that need to be stored
	"""

	# Find all Synapses on cell (on any connected sections)
	dummy_syn = h.Exp2Syn(rootsec(0.5))
	dummy_nc = h.NetCon(None, dummy_syn)
	cell_nc = [nc for nc in list(dummy_nc.postcelllist()) if not nc.same(dummy_nc)]
	cell_syns = [nc.syn() for nc in cell_nc]

	# Save synapse properties
	# - attributes in syn_attrs
	# - section name/identifier 

	# Calculate axial path resistance to synapse

	# Calculate max and min axial path resistance of Section

	# Measure transfer impedance and filter parameters

def map_synapses(rootsec, orig_syn_info):
	"""
	Map synapses to equivalent synaptic inputs on given morphologically
	reduced cell.

	@param rootsec			root section of reduced cell

	@param orig_syn_info	SynData for each synapse on original cell
	"""

	# Loop over all synapses

	# First find the section that the original section mapped to
	# TODO: when sections are collapsed: keep a list of original section names on the secref


