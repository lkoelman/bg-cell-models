"""
Experimental protocol with minimal number / configuration of synapses
to elicit interesting behaviours, e.g. a burst.

@author Lucas Koelman
"""

import neuron
h = neuron.h

# Physiological parameters
import cellpopdata as cpd
from cellpopdata import PhysioState, Populations as Pop, NTReceptors as NTR, ParameterSource as Cit

# Stimulation protocols
from proto_common import *

################################################################################
# Interface functions
################################################################################

# see StnModelEvaluator

################################################################################
# Building block functions
################################################################################

def make_GLU_inputs(self, n_ctx_syn, connector=None):
	"""
	Make excitatory inputs distributed randomly over the dendrite
	"""
	if connector is None:
		cc = cpd.CellConnector(self.physio_state, self.rng)
	else:
		cc = connector

	model = self.target_model
		
	# Add CTX inputs using Tsodyks-Markram synapses
	# Distribute synapses over dendritic trees
	is_ctx_target = lambda seg: seg.diam <= 1.0         
	dendrites = self.model_data[model]['sec_refs']['dendrites']
	dend_secrefs = sum(dendrites, [])
	ctx_target_segs = pick_random_segments(dend_secrefs, n_ctx_syn, is_ctx_target, rng=self.rng)

	# Make synapses
	new_inputs = {}
	for target_seg in ctx_target_segs:

		# Make poisson spike generator
		stim_rate = 100.0 # hz
		stim_T = stim_rate**-1*1e3
		stimsource = h.NetStim() # Create a NetStim
		stimsource.interval = stim_T # Interval between spikes
		stimsource.number = 5 # max number of spikes
		stimsource.noise = 0.0 # Fractional noise in timing

		# Custom synapse parameters
		syn_mech = 'GLUsyn'
		syn_params = {
			'U1': 0.7,
			'tau_rec': 200., # 1./20. / 2. * 1e3, # 95% recovery of RRP under 20Hz stim (Gradinaru 2009)
			'tau_facil': 1., # no facilitation
		}

		# Make synapse and NetCon
		syn, nc, wvecs = cc.make_synapse((Pop.CTX, Pop.STN), (stimsource, target_seg), 
							syn_mech, (NTR.AMPA, NTR.NMDA), use_sources=(Cit.Custom, Cit.Default),
							custom_synpar=syn_params)

		print("Made {} synapse with following parameters:".format(syn_mech))
		for pname in cc.getSynMechParamNames(syn_mech):
			print("{} : {}".format(pname, str(getattr(syn, pname))))

		# Compensate for effect max value Hill factor and U1 on gmax_GABAA and gmax_GABAB
		syn.gmax_AMPA = syn.gmax_AMPA / syn.U1
		syn.gmax_NMDA = syn.gmax_NMDA / syn.U1

		# Control netstim
		tstart = 850
		tstop = tstart + 10*stim_T
		stimsource.start = tstart
		turn_off = h.NetCon(None, stimsource)
		turn_off.weight[0] = -1
		def queue_events():
			turn_off.event(tstop)
		fih = h.FInitializeHandler(queue_events)

		# Save inputs
		extend_dictitem(new_inputs, 'PyInitHandlers', queue_events)
		extend_dictitem(new_inputs, 'HocInitHandlers', fih)
		extend_dictitem(new_inputs, 'syn_NetCons', nc)
		extend_dictitem(new_inputs, 'com_NetCons', turn_off)
		extend_dictitem(new_inputs, 'synapses', syn)
		extend_dictitem(new_inputs, 'NetStims', stimsource)
		extend_dictitem(new_inputs, 'stimweightvec', wvecs)

	self.add_inputs('ctx', model, **new_inputs)



def make_GABA_inputs(self, n_gpe_syn, connector=None):
	"""
	Make GABAergic synapses on the STN neuron.

	@param self         StnModelEvaluator object
	"""
	if connector is None:
		cc = cpd.CellConnector(self.physio_state, self.rng)
	else:
		cc = connector

	model = self.target_model
		
	# Add GPe inputs using Tsodyks-Markram synapses
	# NOTE: one synapse represents a multi-synaptic contact from one GPe axon
	new_inputs = {}

	# Pick random segments in dendrites for placing synapses
	is_gpe_target = lambda seg: seg.diam > 1.0 # select proximal dendrites
	dendrites = self.model_data[model]['sec_refs']['dendrites']
	dend_secrefs = sum(dendrites, [])
	gpe_target_segs = pick_random_segments(dend_secrefs, n_gpe_syn, is_gpe_target, rng=self.rng)

	# Make synapses
	for target_seg in gpe_target_segs:

		# Make poisson spike generator
		stim_rate = 100.0 # hz
		stim_T = stim_rate**-1*1e3
		stimsource = h.NetStim() # Create a NetStim
		stimsource.interval = stim_T # Interval between spikes
		stimsource.number = 8 # max number of spikes
		stimsource.noise = 0.0 # Fractional noise in timing

		# Custom synapse parameters
		syn_mech = 'GABAsyn'
		syn_params = {
			'use_stdp_A': 1,
			'use_stdp_B': 1,
		}

		# Make synapse and NetCon
		syn, nc, wvecs = cc.make_synapse((Pop.GPE, Pop.STN), (stimsource, target_seg), 
							syn_mech, (NTR.GABAA, NTR.GABAB), 
							use_sources=(Cit.Custom, Cit.Chu2015, Cit.Fan2012, Cit.Atherton2013),
							custom_synpar=syn_params)

		print("Made {} synapse with following parameters:".format(syn_mech))
		for pname in cc.getSynMechParamNames(syn_mech):
			print("{} : {}".format(pname, str(getattr(syn, pname))))

		# Compensate for effect max value Hill factor and U1 on gmax_GABAA and gmax_GABAB
		syn.gmax_GABAA = syn.gmax_GABAA / syn.U1
		syn.gmax_GABAB = syn.gmax_GABAB / 0.21

		# Control netstim
		tstart = 700
		tstop = tstart + 5*stim_T
		stimsource.start = tstart
		turn_off = h.NetCon(None, stimsource)
		turn_off.weight[0] = -1
		def queue_events():
			turn_off.event(tstop)
		
		extend_dictitem(new_inputs, 'PyInitHandlers', queue_events)
		extend_dictitem(new_inputs, 'HocInitHandlers', h.FInitializeHandler(queue_events))
		extend_dictitem(new_inputs, 'syn_NetCons', nc)
		extend_dictitem(new_inputs, 'com_NetCons', turn_off)
		extend_dictitem(new_inputs, 'synapses', syn)
		extend_dictitem(new_inputs, 'NetStims', stimsource)
		extend_dictitem(new_inputs, 'stimweightvec', wvecs)

	# Save inputs
	self.add_inputs('gpe', model, **new_inputs)

