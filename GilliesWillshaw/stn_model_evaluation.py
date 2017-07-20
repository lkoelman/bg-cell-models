"""
Evaluation of STN cell models under different physiological and stimulus conditions.
"""

# Standard library modules
import logging
logging.basicConfig(format='%(levelname)s:%(message)s @%(filename)s:%(lineno)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module

import collections
from enum import Enum

# Third party modules
import numpy as np

# NEURON modules
import neuron
from neuron import h
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

# Add our own modules to Python path
import sys, os.path
scriptdir, scriptfile = os.path.split(__file__)
modulesbase = os.path.normpath(os.path.join(scriptdir, '..'))
sys.path.append(modulesbase)

# Our own modules
import gillies_model as gillies
import reduce_marasco as marasco
import mapsyn
from common import analysis
from cellpopdata import PhysioState

class StimProtocol(Enum):
	"""
	Synaptic stimulus sets or electrode stimulation protocols
	to administer to STN cell
	"""
	SPONTANEOUS = 0 # spontaneous firing (no inputs)
	CLAMP_PLATEAU = 1 # plateau potential (Gillies 2006, Fig. 10C-D)
	CLAMP_REBOUND = 2 # rebound burst (Gillies 2006, Fig. 3-4)
	SYN_BACKGROUND_HIGH = 3 # synaptic bombardment, high background activity
	SYN_BACKGROUND_LOW = 4 # synaptic bombardment, low background activity
	SYN_PARK_PATTERNED = 5 # pathological input, strong patterned cortical input with strong GPi input in antiphase

class StnModel(Enum):
	"""
	STN cell models
	"""
	Gillies2005 = 0
	Miocinovic2006 = 1
	Gillies_GIF = 2
	Gillies_BranchZip = 3


def pick_random_segments(secrefs, n_segs, elig_func, rng=None):
	"""
	Pick random segments with spatially uniform distribution.
	"""
	# Get random number generator
	if rng is None:
		rng = np.random

	# Gather segments that are eligible.
	elig_segs = [seg for ref in secrefs for seg in ref.sec if elig_func(seg)]
	logger.debug("Found {} eligible target segments".format(len(elig_segs)))

	# Sample segments
	# 	Note that nseg/L is not necessarily uniform so that randomly picking
	# 	segments will not lead to a uniform spatial distribution of synapses.
	target_segs = [] # target segments, including their x-location
	Ltotal = sum((seg.sec.L/seg.sec.nseg for seg in elig_segs)) # summed length of all found segments
	for i in xrange(n_segs):
		sample = rng.random_sample() # in [0,1)
		# Pick segment at random fraction of combined length of Sections
		Ltraversed = 0.0
		for seg in elig_segs:
			Lseg = seg.sec.L
			if Ltraversed <= (sample*Ltotal) < Ltraversed+Lseg:
				# Find x on Section by interpolation
				percent_seg = (sample*Ltotal - Ltraversed)/Lseg
				xwidth = 1.0/seg.sec.nseg
				x0_seg = seg.x - 0.5*xwidth
				x_on_sec = x0_seg + percent_seg*xwidth
				target_segs.append(seg.sec(x_on_sec))
			Ltraversed += Lseg

	return target_segs


class StnModelEvaluator(object):
	"""
	Evaluate STN models

	Inspired by:
	- optimization.py/Simulation
	- optimization.py/StnCellController
	- bgmodel/models/kang/model.py/BGSim

	Improvements:
	- make evaluator subclasses for the different protocols
	"""

	def __init__(self, target_model, physio_state=PhysioState.NORMAL):
		"""
		Initialize new evaluator in given physiological state.
		"""
		self._physio_state = physio_state

		self.model_data = dict(((model, {}) for model in list(StnModel)))
		self.target_model = target_model

		somaref, dendrefs = self.build_cell(self.target_model)

		self.model_data[self.target_model]['sec_refs'] = {
			'soma': somaref,
			'dendrites': dendrefs # one list(SectionRef) per dendrite
		}

		self.model_data[self.target_model]['rec_data'] = dict(((proto, {}) for proto in StimProtocol))

		self.sim_dur = 1000.
		self.sim_dt = 0.025
		self.rng = np.random.RandomState(25031989)

	@property
	def physio_state(self):
		"""
		Get cell physiological state.
		"""
		return self._physio_state

	@physio_state.setter
	def set_physio_state(self, state):
		"""
		Set cell physiological state.
		"""
		# Change model parameters
		if self.target_model == StnModel.Gillies2005:

			if state == PhysioState.NORMAL:
				pass # this case corresponds to default model parameters

			elif (state == PhysioState.PARKINSONIAN or state == PhysioState.PARK_DBS):

				# TODO: decide parameter modifications for DA-depleted state from literature
				somaref = self.model_data[self.target_model]['sec_refs']['soma']
				dendrefs =  self.model_data[self.target_model]['sec_refs']['dendrites']

				# 1. Reduce sKCA channel conductance by 90%, from sources:
				#	- Gillies & Willshaw 2005 (see refs)
				for secref in [somaref] + dendrefs:
					for seg in secrref.sec:
						seg.gk_sKCa = 0.1 * seg.gk_sKCa

				# 2. Modifications to GPE GABA IPSPs
				#	- Changes:
				#		- Increased strength of GABA IPSPs
				#		- longer decay kinetics, 
				#		- increase in number of functional synapses (1 afferent axon has more activated synaptic contacts)
				# 	- References:
				#		- Fan (2012), "Proliferation of External Globus Pallidus-Subthalamic Nucleus Synapses following Degeneration of Midbrain Dopamine Neurons"
				#	- see changes in cellpopdata.py

				# 3. Modifications to GPE AMPA EPSCs (see hLTP)
				#	- See changes in cellpopdata.py

				# 4. Changes to regularity/variability of spontaneous firing (summarize literature)

				if state == PhysioState.PARK_DBS:
					# 5. Neurochemical effects of DBS?
					raise NotImplementedError()
			else:
				raise NotImplementedError()
		else:
			raise Exception("Model '{}' not supported".format(
					self.target_model))

		# Set the state flag
		self._physio_state = state

	def build_cell(self, model):
		"""
		Build cell model using current physiological state
		"""

		if self.model_data[model].get('built', False):
			logger.warning("Attempting to build model {} which has already been built.".format(
							model))

		if model == StnModel.Gillies2005:
			# Make Gillies STN cell
			soma, dends, stims = gillies.stn_cell_gillies()
			somaref, dendL_refs, dendR_refs = gillies.get_stn_refs()
			dendrefs = [dendL_refs, dendR_refs]

		elif model == StnModel.Gillies_BranchZip:
			# Incremental reduction with 'Branch Zipping' algorithm
			logger.warn("Gillies STN model will be modified if created")
			eq_secs, newsecrefs = marasco.reduce_gillies_incremental(n_passes=7, zips_per_pass=100)
			somaref = next(ref for ref in newsecrefs if 'soma' in ref.sec.name())
			dendrefs = [[ref] for ref in newsecrefs if not 'soma' in ref.sec.name()]

		else:
			raise Exception("Model '{}' not supported".format(
					model))

		# Indicate that given model has been built
		self.model_data[model]['built'] = True

		return somaref, dendrefs


	def make_inputs(self, stim_protocol):
		"""
		Make the inputs for given stimulation protocol.
		"""
		if self.target_model != StnModel.Gillies2005:
			raise Exception("Found target model '{}'. Only model '{}' is supported as a target model").format(
				self.target_model, StnModel.Gillies2005)

		# Allocate data for input
		self.model_data[self.target_model]['inputs'] = {}

		# Get sections
		somasec = h.SThcell[0].soma

		if stim_protocol == StimProtocol.SPONTANEOUS:
			# Spontaneous firing has no inputs
			pass

		elif stim_protocol == StimProtocol.CLAMP_PLATEAU:
			
			# Set up stimulation (5 mA/cm2 for 80 ms)
			I_hyper = -0.17 # hyperpolarize to -70 mV (see fig. 10C)
			I_depol = I_hyper + 0.2 # see fig. 10D: 0.2 nA (=stim.amp) over hyperpolarizing current
			dur_depol = 50 # see fig. 10D, top right
			del_depol = 1000
			burst_time = [del_depol-50, del_depol+200] # empirical

			stim1, stim2 = h.stim1, h.stim2

			stim1.delay = 0
			stim1.dur = del_depol
			stim1.amp = I_hyper

			stim2.delay = del_depol
			stim2.dur = dur_depol
			stim2.amp = I_depol

			stim3.delay = del_depol + dur_depol
			stim3.dur = dur - (del_depol + dur_depol)
			stim3.amp = I_hyper

			# Save inputs
			self.model_data[self.target_model]['inputs']['IClamps'] = [stim1, stim2]

		elif stim_protocol == StimProtocol.CLAMP_REBOUND:
			
			stim1, stim2, stim3 = h.stim1, h.stim2, h.stim3

			stim1.delay = 0
			stim1.dur = 500
			stim1.amp = 0.0

			# stim2.delay = 200
			# stim2.dur = 500
			# stim2.amp = -0.11 # -0.25 in full model (hyperpolarize to -75 mV steady state)
			stim2.amp = 0

			# Use voltage clamp (space clamp) instead of current clamp
			clamp = h.SEClamp(soma(0.5))
			clamp.dur1 = 0
			clamp.dur2 = 0
			clamp.dur3 = 500
			clamp.amp3 = -75

			stim3.delay = 1000
			stim3.dur = 1000
			stim3.amp = 0.0

			# Save inputs
			self.model_data[self.target_model]['inputs']['IClamps'] = [stim1, stim2, stim3]
			self.model_data[self.target_model]['inputs']['SEClamps'] = [clamp]

		elif stim_protocol == StimProtocol.SYN_BACKGROUND_HIGH:
			# TODO: implement inputs for SYN_BACKGROUND_HIGH
			raise NotImplementedError()

		elif stim_protocol == StimProtocol.SYN_BACKGROUND_LOW:
			# TODO: implement inputs for SYN_BACKGROUND_LOW
			raise NotImplementedError()

		elif stim_protocol == StimProtocol.SYN_PARK_PATTERNED:
			
			# Method 1:
			# 	- add netstim and play input signal into weight (see Dura-Berndal arm example)
			# 	- advantage: low weight in off-period will provide background noise
			# 	- if no input desired in off-period: periodically turn it on/off using events or other NetStim
			
			# Method 2:
			# 	- use nsloc.mod (=netstim with variable rate)

			####################################################################
			# CTX inputs
			####################################################################

			# TODO: calibrate cortical inputs
			#	- test that EPSP/IPSP have desired size
			#	- in DA-depleted state: should trigger bursts

			n_ctx_syn = 10
			ctx_syns = []
			ctx_ncs = []
			ctx_stims = []

			# Make weight signal
			ctx_timevec = np.arange(0, self.sim_dur, 0.05) # update every 0.05 ms
			ctx_pattern_freq = 20.0 # frequency [Hz]
			pattern = np.sin(2*np.pi*ctx_pattern_freq*1e-3*ctx_timevec) # amplitude 1.0
			noise_amp = 0.05 # TODO: fractional noise amplitude
			pattern[pattern<=noise_amp] = noise_amp
			
			# DEBUG: visualize waveform
			# from matplotlib import pyplot as plt
			# plt.plot(ctx_timevec, pattern)
			# plt.show(block=True)
			
			# gmax is in [uS] in POINT_PROCESS synapses
			# 	1 [uS] * 1 [mV] = 1 [nA]
			# 	we want ~ -300 [pA] = -0.3 [nA]
			# 	gmax [uS] * (-68 [mV] - 0 [mV]) = gmax * -68 [nA]
			# 	gmax * -68 = -0.3 <=> gmax = 0.3/68 = 0.044 [uS]
			gmax = 0.390/68. # 390 [pA] @ RMP=-68 [mV]
			ctx_weightvec = pattern * gmax

			stimweightvec = h.Vector(ctx_weightvec)
			stimtimevec = h.Vector(ctx_timevec)

			# Distribute synapses over dendritic trees
			is_ctx_target = lambda seg: seg.diam <= 1.0			
			dendrites = self.model_data[self.target_model]['sec_refs']['dendrites']
			dend_secrefs = sum(dendrites, [])
			ctx_target_segs = pick_random_segments(dend_secrefs, n_ctx_syn, is_ctx_target, rng=self.rng)

			# Make synapses
			for target_seg in ctx_target_segs:

				# Make a synapse
				asyn = h.Exp2Syn(target_seg) # TODO: use depressing synapses (tmgsyn)
				asyn.tau1 = 1.0
				asyn.tau2 = 1.75 + self.rng.rand()*2.25 # see refs: ~ 1.75-4 ms
				asyn.e = 0.0
				ctx_syns.append(asyn)

				# Make poisson spike generator
				# TODO: (CTX) calibrate poisson noise & rate parameters, use references for this (reported in vivo firing rates, traces etc)
				stim_rate = 50.0
				stimsource = h.NetStim() # Create a NetStim
				stimsource.interval = stim_rate**-1*1e3 # Interval between spikes
				stimsource.number = 1e9 # max number of spikes
				stimsource.noise = 0.33 # Fractional noise in timing
				# stimsource.noiseFromRandom(stimrand) # Set it to use this random number generator
				ctx_stims.append(stimsource) # Save this NetStim

				# Make NetCon
				stimconn = h.NetCon(stimsource, asyn)
				stimconn.delay = 1.0
				stimweightvec.play(stimconn._ref_weight[0], stimtimevec)
				ctx_ncs.append(stimconn)

			# Save inputs
			self.model_data[self.target_model]['inputs']['ctx'] = {
				'stimtimevec': stimtimevec,
				'stimweightvec': stimweightvec,
				'synapses': ctx_syns,
				'NetCons': ctx_ncs,
				'NetStims': ctx_stims,
			}

			####################################################################
			# GPe inputs
			####################################################################

			# Add GPe inputs using Tsodyks-Markram synapses
			n_gpe_syn = 10
			gpe_syns = []
			gpe_ncs = []
			gpe_stims = []

			# Make weight signal
			gpe_timevec = np.arange(0, self.sim_dur, 0.05) # update every 0.05 ms
			gpe_pattern_freq = 20.0 # frequency [Hz]
			gpe_pattern_phase = np.pi # anti-phase from CTX
			gpe_pattern = np.sin(2*np.pi*gpe_pattern_freq*1e-3*gpe_timevec + gpe_pattern_phase) # amplitude 1.0
			gpe_pattern[gpe_pattern<=0] = 0.05 # fractional noise amplitude

			# See calculation above: gmax = x [nA]/RMP [mV] = y [uS]
			# TODO: correct both gmax for attenuation from dendrites to soma. Do this separately for GPe and CTX inputs since they have to travel different path lengths.
			gmax = 0.450/68. # 450 [pA] @ RMP=-68 [mV]
			gpe_weightvec = gpe_pattern * gmax

			stimweightvec = h.Vector(gpe_weightvec)
			stimtimevec = h.Vector(gpe_timevec)

			# Pick random segments in dendrites for placing synapses
			is_gpe_target = lambda seg: seg.diam > 1.0 # select proximal dendrites
			dendrites = self.model_data[self.target_model]['sec_refs']['dendrites']
			dend_secrefs = sum(dendrites, [])
			gpe_target_segs = pick_random_segments(dend_secrefs, n_gpe_syn, is_gpe_target, rng=self.rng)

			# Make synapses
			for target_seg in gpe_target_segs:

				# Make a synapse
				asyn = h.Exp2Syn(target_seg)
				asyn.tau1 = 2.1 + self.rng.rand()*1. # see refs: ~ 2.1-3.1 ms
				asyn.tau2 = 2.75 + self.rng.rand()*3.25 # see refs: ~ 2.75-6 ms
				asyn.e = 0.0
				gpe_syns.append(asyn)

				# Make poisson spike generator
				# TODO: (GPe) calibrate poisson noise & rate parameters, see CTX notes
				stim_rate = 50.0
				stimsource = h.NetStim() # Create a NetStim
				stimsource.interval = stim_rate**-1*1e3 # Interval between spikes
				stimsource.number = 1e9 # max number of spikes
				stimsource.noise = 0.33 # Fractional noise in timing
				# stimsource.noiseFromRandom(stimrand) # Set it to use this random number generator
				gpe_stims.append(stimsource) # Save this NetStim

				# Make NetCon
				stimconn = h.NetCon(stimsource, asyn)
				stimconn.delay = 1.0
				stimweightvec.play(stimconn._ref_weight[0], stimtimevec)
				gpe_ncs.append(stimconn)

			# Save inputs
			self.model_data[self.target_model]['inputs']['gpe'] = {
				'stimtimevec': stimtimevec,
				'stimweightvec': gpe_weightvec,
				'synapses': gpe_syns,
				'NetCons': gpe_ncs,
				'NetStims': gpe_stims,
			}



	def map_inputs(self, cand_model):
		"""
		Map inputs from target model to candidate model.
		"""
		raise NotImplementedError()


	def rec_traces(self, protocol, recordStep=0.025):
		"""
		Set up recording Vectors to record from relevant pointers
		"""
		# Specify sections to record from
		if self.target_model == StnModel.Gillies2005:

			somasec = h.SThcell[0].soma
			dendsec = h.SThcell[0].dend1[7]

			# Assign label to each recorded section
			rec_segs = {
				'soma': somasec(0.5), # middle of soma
				'dist_dend': dendsec(0.8), # approximate location along dendrite in fig. 5C
			}
			
			self.model_data[self.target_model]['rec_segs'] = rec_segs

		else:
			raise NotImplementedError("""Recording from other models 
					besides {} not yet implemented""".format(StnModel.Gillies2005))

		# Start trace specification
		traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)
		traceSpecs['t_global'] = {'var':'t'}
		self.rec_dt = recordStep

		# PROTOCOL-SPECIFIC TRACES
		if protocol == StimProtocol.SPONTANEOUS: # spontaneous firing (no inputs)
			
			# Trace specs for membrane voltages
			for seclabel, seg in rec_segs.iteritems():
				traceSpecs['V_'+seclabel] = {'sec':seclabel, 'loc':seg.x, 'var':'v'}

			# Trace specs for recording ionic currents, channel states
			analysis.rec_currents_activations(traceSpecs, 'soma', 0.5)

		elif protocol == StimProtocol.CLAMP_PLATEAU: # plateau potential (Gillies 2006, Fig. 10C-D):
			
			# Trace specs for membrane voltages
			for seclabel, seg in rec_segs.iteritems():
				traceSpecs['V_'+seclabel] = {'sec':seclabel, 'loc':seg.x, 'var':'v'}

			# Record Ca and Ca-activated currents in dendrite
			dendloc = rec_segs['dist_dend'].x

			# K currents (dendrite)
			traceSpecs['I_KCa_d'] = {'sec':'dist_dend','loc':dendloc,'mech':'sKCa','var':'isKCa'}
			
			# Ca currents (dendrite)
			traceSpecs['I_CaL_d'] = {'sec':'dist_dend','loc':dendloc,'mech':'HVA','var':'iLCa'}
			traceSpecs['I_CaN_d'] = {'sec':'dist_dend','loc':dendloc,'mech':'HVA','var':'iNCa'}
			traceSpecs['I_CaT_d'] = {'sec':'dist_dend','loc':dendloc,'mech':'CaT','var':'iCaT'}

		elif protocol == StimProtocol.CLAMP_REBOUND: # rebound burst (Gillies 2006, Fig. 3-4)

			# Trace specs for membrane voltages
			for seclabel, seg in rec_segs.iteritems():
				traceSpecs['V_'+seclabel] = {'sec':seclabel, 'loc':seg.x, 'var':'v'}

			# Trace specs for recording ionic currents, channel states
			analysis.rec_currents_activations(traceSpecs, 'soma', 0.5)
			
			# Ca and K currents in distal dendrites
			dendloc = rec_segs['dist_dend'].x
			analysis.rec_currents_activations(traceSpecs, 'dist_dend', dendloc, ion_species=['ca','k'])

		elif protocol == StimProtocol.SYN_BACKGROUND_HIGH: # synaptic bombardment, high background activity
			# TODO: decide what to record for SYN_BACKGROUND_HIGH
			raise NotImplementedError()

		elif protocol == StimProtocol.SYN_BACKGROUND_LOW: # synaptic bombardment, low background activity
			# TODO: decide what to record for SYN_BACKGROUND_LOW
			raise NotImplementedError()

		elif protocol == StimProtocol.SYN_PARK_PATTERNED: # pathological input, strong patterned cortical input with strong GPi input in antiphase
			
			# See diagram in marasco_reduction.pptx
			# dist_dend0_ids = [8,9,7,10,12,13,18,19,17,20,22,23]
			# prox_dend0_ids = [2,4,5,3,14,15]
			# dist_dend1_ids = [6,7,5,8,10,11]
			# prox_dend1_ids = [1,2,3]

			# Pick some proximal and distal dendritic sections for recording
			dist_secs = [(0,9), (0,10), (0,17), (0,23), (1,6), (1,8)]
			prox_secs = [(0,2), (0,3), (1,1)]

			# Add to recorded sections
			for tree_id, sec_id in dist_secs:
				tree_name = 'dend' + str(tree_id)
				sec = getattr(h.SThcell[0], tree_name)[sec_id-1]
				rec_segs['dist_' + repr(sec)] = sec(0.9)

			for tree_id, sec_id in prox_secs:
				tree_name = 'dend' + str(tree_id)
				sec = getattr(h.SThcell[0], tree_name)[sec_id-1]
				rec_segs['prox_' + repr(sec)] = sec(0.9)

			# Pick some segments that are targeted by synapse
			gpe_ncs = self.model_data[self.target_model]['inputs']['gpe']['NetCons']
			ctx_ncs = self.model_data[self.target_model]['inputs']['ctx']['NetCons']
			gpe_picks = [gpe_ncs[i] for i in self.rng.choice(len(gpe_ncs), 3, replace=False)]
			ctx_picks = [ctx_ncs[i] for i in self.rng.choice(len(ctx_ncs), 3, replace=False)]
			for nc in gpe_picks + ctx_picks:
				seg = nc.syn().get_segment()
				rec_segs['postsyn_' + repr(seg.sec)] = seg


			# Specify which traces you want in these sections
			for seclabel, seg in rec_segs.iteritems():
				# Membrane voltages
				traceSpecs['V_'+seclabel] = {'sec':seclabel, 'loc':seg.x, 'var':'v'}

				# # K currents (dendrite)
				# traceSpecs['I_KCa_'+seclabel] = {'sec':'dist_dend','loc':seg.x,'mech':'sKCa','var':'isKCa'}
				
				# # Ca currents (dendrite)
				# traceSpecs['I_CaL_'+seclabel] = {'sec':'dist_dend','loc':seg.x,'mech':'HVA','var':'iLCa'}
				# traceSpecs['I_CaN_'+seclabel] = {'sec':'dist_dend','loc':seg.x,'mech':'HVA','var':'iNCa'}
				# traceSpecs['I_CaT_'+seclabel] = {'sec':'dist_dend','loc':seg.x,'mech':'CaT','var':'iCaT'}


		# Prepare dictionary (label -> Section)
		rec_secs = dict((seclabel, seg.sec) for seclabel, seg in rec_segs.iteritems())

		# Use trace specs to make Hoc Vectors
		recData = analysis.recordTraces(rec_secs, traceSpecs, recordStep)

		# Save trace specs and recording Vectors
		self.model_data[self.target_model]['rec_data'][protocol] = {
			'trace_specs': traceSpecs,
			'trace_data': recData,
			'rec_dt': recordStep,
		}
		


	def plot_traces(self, protocol, model=None):
		"""
		Plot relevant recorded traces for given protocol
		"""

		# Get recorded data
		if model is None:
			model = self.target_model
		recData = self.model_data[model]['rec_data'][protocol]['trace_data']
		recordStep = self.model_data[model]['rec_data'][protocol]['rec_dt']

		# Plot membrane voltages
		def plot_all_Vm():
			recV = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.startswith('V_')]) # preserves order
			figs_vm = analysis.plotTraces(recV, recordStep, yRange=(-80,40), 
											traceSharex=True, oneFigPer='trace')
			return figs_vm

		# Extra plots depending on simulated protocol
		if protocol == StimProtocol.SPONTANEOUS:
			plot_all_Vm()

			# Plot ionic currents, (in)activation variables
			figs, cursors = analysis.plot_currents_activations(self.recData, recordStep)

		elif protocol == StimProtocol.CLAMP_PLATEAU:
			plot_all_Vm()

			# Plot ionic currents, (in)activation variables
			figs, cursors = analysis.plot_currents_activations(recData, recordStep)

			# Dendrite currents during burst
			recDend = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.endswith('_d')])
			analysis.cumulPlotTraces(recDend, recordStep, timeRange=burst_time)

		elif protocol == StimProtocol.CLAMP_REBOUND:
			plot_all_Vm()

			# Plot ionic currents, (in)activation variables
			figs_soma, cursors_soma = analysis.plot_currents_activations(recData, recordStep, sec_tag='soma')
			figs_dend, cursors_dend = analysis.plot_currents_activations(recData, recordStep, sec_tag='dist_dend')
			figs = figs_soma + figs_dend
			cursors = cursors_soma + cursors_dend

		elif protocol == StimProtocol.SYN_BACKGROUND_HIGH:
			plot_all_Vm() # only plot membrane voltages

		elif protocol == StimProtocol.SYN_BACKGROUND_LOW:
			plot_all_Vm() # only plot membrane voltages

		elif protocol == StimProtocol.SYN_PARK_PATTERNED:
			V_prox = analysis.match_traces(recData, lambda t: t.startswith('V_prox'))
			V_dist = analysis.match_traces(recData, lambda t: t.startswith('V_dist'))
			V_postsyn = analysis.match_traces(recData, lambda t: t.startswith('V_postsyn'))

			analysis.plotTraces(V_prox, recordStep, yRange=(-80,40), traceSharex=True)
			analysis.plotTraces(V_dist, recordStep, yRange=(-80,40), traceSharex=True)
			analysis.plotTraces(V_postsyn, recordStep, yRange=(-80,40), traceSharex=True)

	def run_sim(self, dur=None):
		"""
		Run NEURON simulator for `dur` or `self.sim_dur` milliseconds
		with precise measurement of runtime
		"""
		if dur is None:
			dur = self.sim_dur

		h.tstop = dur
		h.init() # calls finitialize() and fcurrent()
		logger.debug("Simulating...")
		t0 = h.startsw()
		h.run()
		t1 = h.startsw()
		h.stopsw() # or t1=h.startsw(); runtime = t1-t0
		logger.debug("Simulated for {:.6f} seconds".format(t1-t0))


	def run_protocol(self, protocol, model=None):
		"""
		Simulate cell in physiological state, under given stimulation protocol.

		@param model	Model to simulate protocol with. If no model is given,
						the target model is used.

		@pre		cell must be built using build_cell

		@pre		physiological state must be set

		@effect		makes inputs (afferents, stimulation) based on physiological 
					state and stimulation protocol (see make_inputs)

		@effect		runs a simulation
		"""
		
		if protocol == StimProtocol.SPONTANEOUS: # spontaneous firing (no inputs)
			
			# Set simulation parameters
			self.sim_dur = 1000
			h.dt = 0.025
			self.sim_dt = h.dt

			h.celsius = 35 # different temp from paper (fig 3B: 25degC, fig. 3C: 35degC)
			h.v_init = -60 # paper simulations use default v_init
			gillies.set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

			# Set up recording
			self.rec_traces(protocol, recordStep=0.05)

			# Simulate
			self.run_sim()

		elif protocol == StimProtocol.CLAMP_PLATEAU: # plateau potential (Gillies 2006, Fig. 10C-D):
			
			# Set simulation parameters
			self.sim_dur = 2000
			h.dt = 0.025
			self.sim_dt = h.dt

			h.celsius = 30 # different temp from paper
			h.v_init = -60 # paper simulations sue default v_init
			gillies.set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

			# Make inputs
			self.make_inputs(protocol)

			# Set up recording
			self.rec_traces(protocol, recordStep=0.025)

			# Simulate
			self.run_sim()

		elif protocol == StimProtocol.CLAMP_REBOUND: # rebound burst (Gillies 2006, Fig. 3-4)
			
			# Set simulation parameters
			self.sim_dur = 2000
			h.dt = 0.025
			self.sim_dt = h.dt

			h.celsius = 35 # different temp from paper
			h.v_init = -60 # paper simulations sue default v_init
			gillies.set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

			# Make inputs
			self.make_inputs(protocol)

			# Set up recording
			self.rec_traces(protocol, recordStep=0.05)

			# Simulate
			self.run_sim()

		elif protocol == StimProtocol.SYN_BACKGROUND_HIGH: # synaptic bombardment, high background activity
			raise NotImplementedError()

		elif protocol == StimProtocol.SYN_BACKGROUND_LOW: # synaptic bombardment, low background activity
			raise NotImplementedError()

		elif protocol == StimProtocol.SYN_PARK_PATTERNED: # pathological input, strong patterned cortical input with strong GPi input in antiphase
			
			# Set simulation parameters
			self.sim_dur = 2000
			h.dt = 0.025
			self.sim_dt = h.dt

			h.celsius = 35 # different temp from paper
			h.v_init = -60 # paper simulations sue default v_init
			gillies.set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

			# Make inputs
			self.make_inputs(protocol)

			# Set up recording
			self.rec_traces(protocol, recordStep=0.05)

			# Simulate
			self.run_sim()



if __name__ == '__main__':
	# Run Gillies 2005 model
	evaluator = StnModelEvaluator(StnModel.Gillies2005, PhysioState.NORMAL)

	# Choose protocol
	# proto = StimProtocol.SPONTANEOUS
	# proto = StimProtocol.CLAMP_PLATEAU
	# proto = StimProtocol.CLAMP_REBOUND
	proto = StimProtocol.SYN_PARK_PATTERNED
	# proto = StimProtocol.SYN_BACKGROUND_LOW
	# proto = StimProtocol.SYN_BACKGROUND_HIGH
	evaluator.run_protocol(proto)
	evaluator.plot_traces(proto)



