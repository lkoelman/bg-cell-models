"""
Experimental protocol with minimal number / configuration of synapses
to elicit interesting behaviours, e.g. a burst.

@author Lucas Koelman
"""

import neuron
h = neuron.h

# Physiological parameters
import cellpopdata as cpd
from cellpopdata import (
	Populations as Pop, 
	NTReceptors as NTR, 
	ParameterSource as Cit
)

# Stimulation protocols
from proto_common import (
	StimProtocol, EvaluationStep, 
	register_step, pick_random_segments, extend_dictitem,
)

################################################################################
# Interface functions
################################################################################

@register_step(EvaluationStep.INIT_SIMULATION, StimProtocol.SINGLE_SYN_GLU)
@register_step(EvaluationStep.INIT_SIMULATION, StimProtocol.SINGLE_SYN_GABA)
def init_sim(self, protocol):
	"""
	Initialize simulator and cell for stimulation protocol
	"""

	# Change simulation duration
	self._init_sim(dur=2000)

	# Make cell passive: disable all active channels
	gbar_active = self.model_data[self.target_model]['active_gbar_names']

	# Disable spiking (SETPARAM: disable spiking if necessary)
	disable_channels = 'spiking'
	for ref in self.all_sec_refs(self.target_model):
		for seg in ref.sec:

			# Disable ion channels
			if disable_channels == 'spiking':
				setattr(seg, 'gna_Na', 0.0)

			elif disable_channels == 'active':
				for gbar in gbar_active:
					setattr(seg, gbar, 0.0)

@register_step(EvaluationStep.INIT_SIMULATION, StimProtocol.MIN_SYN_BURST)
def init_sim_BURST(self, protocol):
	"""
	Initialize simulator and cell for stimulation protocol
	"""

	# Change simulation duration
	self._init_sim(dur=2000)

	# lower sKCa conductance to promote bursting
	for sec in h.allsec():
		for seg in sec:
			seg.gk_sKCa = 0.6 * seg.gk_sKCa


@register_step(EvaluationStep.MAKE_INPUTS, StimProtocol.SINGLE_SYN_GLU)
def make_inputs_GLU(self, connector=None):
	"""
	Make single synaptic input
	"""
	make_GLU_inputs(self, 1, connector=connector)


@register_step(EvaluationStep.MAKE_INPUTS, StimProtocol.SINGLE_SYN_GABA)
def make_inputs_GABA(self, connector=None):
	"""
	Make single synaptic input
	"""
	make_GABA_inputs(self, 1, connector=connector)


@register_step(EvaluationStep.MAKE_INPUTS, StimProtocol.MIN_SYN_BURST)
def make_inputs_BURST(self, connector=None):
	"""
	Make minimal number of synapses that elicit burst
	"""
	# Minimal number of GABA + GLU synapses to trigger burst
	make_GABA_inputs(self, 1)
	make_GLU_inputs(self, 4)


@register_step(EvaluationStep.RECORD_TRACES, StimProtocol.SINGLE_SYN_GLU)
def rec_traces_GLU(self, protocol, traceSpecs):
	"""
	Record all traces for this protocol.
	"""
	# Record synaptic variables
	self.rec_GLU_traces(protocol, traceSpecs)

	# Record membrane voltages
	self.rec_Vm(protocol, traceSpecs)


@register_step(EvaluationStep.RECORD_TRACES, StimProtocol.SINGLE_SYN_GABA)
def rec_traces_GABA(self, protocol, traceSpecs):
	"""
	Record all traces for this protocol.
	"""
	# Record synaptic variables
	self.rec_GABA_traces(protocol, traceSpecs)

	# Record membrane voltages
	self.rec_Vm(protocol, traceSpecs)


@register_step(EvaluationStep.RECORD_TRACES, StimProtocol.MIN_SYN_BURST)
def rec_traces_BURST(self, protocol, traceSpecs):
	"""
	Record all traces for this protocol.
	"""
	# Record both GABA and GLU synapses
	self.rec_GABA_traces(protocol, traceSpecs)
	self.rec_GLU_traces(protocol, traceSpecs)

	# Record membrane voltages
	self.rec_Vm(protocol, traceSpecs)


@register_step(EvaluationStep.PLOT_TRACES, StimProtocol.SINGLE_SYN_GLU)
def plot_traces_GLU(self, model, protocol):
	"""
	Plot relevant traces for this protocol.
	"""
	# Plot membrane voltages (one figure)
	self._plot_all_Vm(model, protocol, fig_per='cell')

	# Plot synaptic variables
	self._plot_GLU_traces(model, protocol)


@register_step(EvaluationStep.PLOT_TRACES, StimProtocol.SINGLE_SYN_GABA)
def plot_traces_GABA(self, model, protocol):
	"""
	Plot relevant traces for this protocol.
	"""
	# Plot membrane voltages (one figure)
	self._plot_all_Vm(model, protocol, fig_per='cell')

	# Plot synaptic variables
	self._plot_GABA_traces(model, protocol)


@register_step(EvaluationStep.PLOT_TRACES, StimProtocol.MIN_SYN_BURST)
def plot_traces_BURST(self, model, protocol):
	"""
	Plot relevant traces for this protocol.
	"""
	# Plot membrane voltages (one figure)
	self._plot_all_Vm(model, protocol, fig_per='cell')

	# Plot synaptic variables
	self._plot_GLU_traces(model, protocol)
	self._plot_GABA_traces(model, protocol)


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
	dend_secrefs = self.model_data[model]['dend_refs']
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
			# 'gmax_AMPA': 0.0, # SETPARAM: enable only the GLU conductance you want to test
		}

		# Make synapse and NetCon
		syn, nc, wvecs = cc.make_synapse((Pop.CTX, Pop.STN), (stimsource, target_seg), 
							syn_mech, (NTR.AMPA, NTR.NMDA), use_sources=(Cit.Custom, Cit.Default),
							custom_synpar=syn_params)

		print("Made {} synapse with following parameters:".format(syn_mech))
		for pname in cc.getSynMechParamNames(syn_mech):
			print("{} : {}".format(pname, str(getattr(syn, pname))))

		# Control netstim
		tstart = 850
		tstop = tstart + 10*stim_T # SETPARAM: number of spikes in burst
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
	dend_secrefs = self.model_data[model]['dend_refs']
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
			# 'gmax_GABAB': 0.0, # SETPARAM: enable only the GABA conductance you want to test
		}

		# Make synapse and NetCon
		syn, nc, wvecs = cc.make_synapse((Pop.GPE, Pop.STN), (stimsource, target_seg), 
							syn_mech, (NTR.GABAA, NTR.GABAB), 
							use_sources=(Cit.Custom, Cit.Chu2015, Cit.Fan2012, Cit.Atherton2013),
							custom_synpar=syn_params)

		print("Made {} synapse with following parameters:".format(syn_mech))
		for pname in cc.getSynMechParamNames(syn_mech):
			print("{} : {}".format(pname, str(getattr(syn, pname))))

		# Control netstim
		tstart = 700
		tstop = tstart + 5*stim_T # SETPARAM: number of spikes in burst
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

