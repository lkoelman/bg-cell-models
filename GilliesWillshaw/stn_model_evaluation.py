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


class PhysioState(Enum):
	"""
	Physiological state of the cell
	"""
	NORMAL = 0
	PARKINSONIAN = 1
	PARK_DBS = 2

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


def distribute_synapses(secrefs, elig_func):
	"""
	Pick locations on given Sections so that each location has
	a uniform change of being selected.
	"""

	# Gather segments that are eligible.
	elig_segs = [seg for ref in secrefs for seg in ref.sec if elig_func(seg)]
	logger.debug("Found {} eligible target segments for CTX afferents".format(len(elig_segs)))

	# Sample segments
	# Note that nseg/L might not be uniform so that randomly picking
	# segments will not lead to a uniform spatial distribution of synapses.
	target_segs = [] # target segments, including their x-location
	Ltotal = sum((seg.sec.L/seg.sec.nseg for seg in elig_segs)) # summed length of all found segments
	for i in xrange(n_ctx_syn):
		sample = self.rng.random_sample() # in [0,1)
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

	Inspired by
	- optimization.py/Simulation
	- optimization.py/StnCellController
	- bgmodel/models/kang/model.py/BGSim
	"""

	def __init__(self, pstate, target_model):
		self._physio_state = pstate
		self.stim_protocol = sproto

		self.model_data = dict((model, {} for model in list(StnModel)))
		self.target_model = target_model

		somaref, dendrefs = self.build_cell(self.target_model)

		self.model_data[self.target_model]['sec_refs'] = {
			'soma': somaref,
			'dendrites': dendrefs # one list(SectionRef) per dendrite
		}

		self.model_data[self.target_model]['rec_data'] = dict((proto, {} for proto in StimProtocol))

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
		if self.target_model == StnModel.Gillies2005:
			# TODO: change to DA depleted state
			pass
		else:
			raise Exception("Model '{}' not supported".format(
					self.target_model))

		self._physio_state = state

	def build_cell(self, model):
		"""
		Build cell model using current physiological state
		"""
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
			pass

		elif stim_protocol == StimProtocol.SYN_BACKGROUND_LOW:
			# TODO: implement inputs for SYN_BACKGROUND_LOW
			pass

		elif stim_protocol == StimProtocol.SYN_PARK_PATTERNED:

			# Add cortical input
			# Method 1:
			# - add netstim and play input signal into weight (see Dura-Berndal arm example)
			# - advantage: low weight in off-period will provide background noise
			# - if no input desired in off-period: periodically turn it on/off using events or other NetStim
			# Method 2:
			# - use nsloc.mod (=netstim with variable rate)

			n_ctx_syn = 10
			ctx_syns = []
			ctx_ncs = []
			ctx_stims = []

			# Make weight signal
			stimtimevec = np.arange(0, self.sim_dur, 0.05) # update every 0.05 ms
			ctx_pattern_freq = 20.0 # frequency [Hz]
			pattern = np.sin(2*pi*ctx_pattern_freq*1e-3*stimtimevec) # amplitude 1.0
			pattern[pattern<=0] = 0.05 # TODO: fractional noise amplitude
			
			# gmax is in uS in POINT_PROCESS synapses
			# 1 [uS] * 1 [mV] = 1 [nA]
			# we want ~ 300 [pA] = 0.3 [nA]
			# gmax [uS] * (-68 [mV] - 0 [mV]) = gmax * -68 [nA]
			# gmax * -68 = 0.3 <=> gmax = 0.3/68 = 0.044 [uS]
			gmax = 0.390/68. # 390 [pA] @ RMP=-68 [mV]
			stimweightvec = pattern * gmax

			# Distribute synapses over dendritic trees
			is_ctx_target = lambda seg: seg.diam <= 1.0			
			dendrites = self.model_data[self.target_model]['sec_refs']['dendrites']
			dend_secrefs = sum(dendrites, [])
			ctx_target_segs = distribute_synapses(dend_secrefs, is_ctx_target)

			# Make synapses
			for target_seg in ctx_target_segs:

				# Make a synapse
				asyn = h.Exp2Syn(target_seg)
				asyn.tau1 = 1.0
				asyn.tau2 = 1.75 + rng.rand()*2.25 # see refs: ~ 1.75-4 ms
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
				'stimtimevec', stimtimevec,
				'stimweightvec', stimweightvec,
				'synapses', ctx_syns,
				'NetCons', ctx_ncs,
				'NetStims', ctx_stims,
			}


			# Add GPe inputs using Tsodyks-Markram synapses
			n_gpe_syn = 10
			gpe_syns = []
			gpe_ncs = []
			gpe_stims = []

			# Make weight signal
			stimtimevec = np.arange(0, self.sim_dur, 0.05) # update every 0.05 ms
			gpe_pattern_freq = 20.0 # frequency [Hz]
			gpe_pattern_phase = np.pi # anti-phase from CTX
			gpe_pattern = np.sin(2*pi*gpe_pattern_freq*1e-3*stimtimevec + gpe_pattern_phase) # amplitude 1.0
			gpe_pattern[gpe_pattern<=0] = 0.05 # fractional noise amplitude

			# See calculation above: gmax = x [nA]/RMP [mV] = y [uS]
			# TODO: correct both gmax for attenuation from dendrites to soma. Do this separately for GPe and CTX inputs since they have to travel different path lengths.
			gmax = 0.450/68. # 450 [pA] @ RMP=-68 [mV]
			stimweightvec = gpe_pattern * gmax

			# Distribute synapses over dendritic trees
			is_gpe_target = lambda seg: seg.diam > 1.0	
			dendrites = self.model_data[self.target_model]['sec_refs']['dendrites']
			dend_secrefs = sum(dendrites, [])
			gpe_target_segs = distribute_synapses(dend_secrefs, is_gpe_target)

			# Make synapses
			for target_seg in gpe_target_segs:

				# Make a synapse
				asyn = h.Exp2Syn(target_seg)
				asyn.tau1 = 2.1 + rng.rand()*1. # see refs: ~ 2.1-3.1 ms
				asyn.tau2 = 2.75 + rng.rand()*3.25 # see refs: ~ 2.75-6 ms
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
				'stimtimevec', stimtimevec,
				'stimweightvec', stimweightvec,
				'synapses', gpe_syns,
				'NetCons', gpe_ncs,
				'NetStims', gpe_stims,
			}



	def map_inputs(self, cand_model):
		"""
		Map inputs from target model to candidate model.
		"""
		pass


	def rec_traces(self, protocol, recordStep=0.025):
		"""
		Set up recording Vectors to record from relevant pointers
		"""
		# Define named sections to record from
		if self.target_model == StnModel.Gillies2005:

			somasec = h.SThcell[0].soma
			dendsec = h.SThcell[0].dend1[7]

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
			for segname, seg in rec_segs.iteritems():
				traceSpecs['V_'+segname] = {'sec':segname, 'loc':seg.x, 'var':'v'}

			# Trace specs for recording ionic currents, channel states
			rec_currents_activations(traceSpecs, 'soma', 0.5)

		elif protocol == StimProtocol.CLAMP_PLATEAU: # plateau potential (Gillies 2006, Fig. 10C-D):
			
			# Trace specs for membrane voltages
			for segname, seg in rec_segs.iteritems():
				traceSpecs['V_'+segname] = {'sec':segname, 'loc':seg.x, 'var':'v'}

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
			for segname, seg in rec_segs.iteritems():
				traceSpecs['V_'+segname] = {'sec':segname, 'loc':seg.x, 'var':'v'}

			# Trace specs for recording ionic currents, channel states
			rec_currents_activations(traceSpecs, 'soma', 0.5)
			
			# Ca and K currents in distal dendrites
			dendloc = rec_segs['dist_dend'].x
			rec_currents_activations(traceSpecs, 'dist_dend', dendloc, ion_species=['ca','k'])

		elif protocol == StimProtocol.SYN_BACKGROUND_HIGH. # synaptic bombardment, high background activity
			pass

		elif protocol == StimProtocol.SYN_BACKGROUND_LOW: # synaptic bombardment, low background activity
			pass

		elif protocol == StimProtocol.SYN_PARK_PATTERNED: # pathological input, strong patterned cortical input with strong GPi input in antiphase
			
			# Name some proximal and distal dendritic sections for recording
			# See diagram in marasco_reduction.pptx
			dist_dend0_ids = [8,9,7,10,12,13,18,19,17,20,22,23]
			prox_dend0_ids = [2,4,5,3,14,15]

			dist_dend1_ids = [6,7,5,8,10,11]
			prox_dend1_ids = [1,2,3]

			dist_secs = [(0,9), (0,10), (0,17), (0,23), (1,6), (1,8)]
			prox_secs = [(0,2), (0,3), (1,1)]

			for tree_id, sec_id in dist_secs:
				tree_name = 'dend' + str(tree_id)
				sec = getattr(h.SThcell[0], tree_name)[sec_id-1]
				rec_segs['dist_' + repr(sec)] = sec(0.9)

			for tree_id, sec_id in prox_secs:
				tree_name = 'dend' + str(tree_id)
				sec = getattr(h.SThcell[0], tree_name)[sec_id-1]
				rec_segs['prox_' + repr(sec)] = sec(0.9)

			# Traces to record in all sections
			for segname, seg in rec_segs.iteritems():
				# Membrane voltages
				traceSpecs['V_'+segname] = {'sec':segname, 'loc':seg.x, 'var':'v'}

				# K currents (dendrite)
				traceSpecs['I_KCa_'+segname] = {'sec':'dist_dend','loc':seg.x,'mech':'sKCa','var':'isKCa'}
				
				# Ca currents (dendrite)
				traceSpecs['I_CaL_'+segname] = {'sec':'dist_dend','loc':seg.x,'mech':'HVA','var':'iLCa'}
				traceSpecs['I_CaN_'+segname] = {'sec':'dist_dend','loc':seg.x,'mech':'HVA','var':'iNCa'}
				traceSpecs['I_CaT_'+segname] = {'sec':'dist_dend','loc':seg.x,'mech':'CaT','var':'iCaT'}


		# Prepare dictionary with name -> Section
		rec_secs = dict((secname, seg.sec) for secname, seg in rec_segs.iteritems)

		# Use trace specs to make Hoc Vectors
		recData = analysis.recordTraces(rec_secs, traceSpecs, recordStep)

		# Save trace specs and recording Vectors
		self.model_data[self.target_model]['rec_data'][protocol] = {
			'trace_specs': traceSpecs,
			'trace_data': recData,
		}
		


	def plot_traces(self, protocol):
		"""
		Plot relevant recorded traces for given protocol
		"""

		if protocol == StimProtocol.SPONTANEOUS:
			recData = self.model_data[self.target_model]['rec_data'][protocol]['trace_data']
			
			# Plot membrane voltages
			recV = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.startswith('V_')]) # preserves order
			figs_vm = analysis.plotTraces(recV, recordStep, yRange=(-80,40), traceSharex=True)
			vm_fig = figs_vm[0]
			vm_ax = figs_vm[0].axes[0]

			# Plot ionic currents, (in)activation variables
			figs, cursors = plot_currents_activations(self.recData, recordStep)

		elif protocol == StimProtocol.CLAMP_PLATEAU:
			recData = self.model_data[self.target_model]['rec_data'][protocol]['trace_data']
			
			# Plot membrane voltages
			recV = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.startswith('V_')]) # preserves order
			figs_vm = analysis.plotTraces(recV, recordStep, yRange=(-80,40), traceSharex=True)
			vm_fig = figs_vm[0]
			vm_ax = figs_vm[0].axes[0]

			# Plot ionic currents, (in)activation variables
			figs, cursors = plot_currents_activations(recData, recordStep)

			# Dendrite currents during burst
			recDend = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.endswith('_d')])
			analysis.cumulPlotTraces(recDend, recordStep, timeRange=burst_time)

		elif protocol == StimProtocol.CLAMP_REBOUND:
			recData = self.model_data[self.target_model]['rec_data'][protocol]['trace_data']

			# Plot membrane voltages
			recV = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.startswith('V_')]) # preserves order
			figs_vm = analysis.plotTraces(recV, recordStep, yRange=(-80,40), traceSharex=True)
			vm_fig = figs_vm[0]
			vm_ax = figs_vm[0].axes[0]

			# Plot ionic currents, (in)activation variables
			figs_soma, cursors_soma = plot_currents_activations(recData, recordStep, sec_tag='soma')
			figs_dend, cursors_dend = plot_currents_activations(recData, recordStep, sec_tag='dist_dend')
			figs = figs_soma + figs_dend
			cursors = cursors_soma + cursors_dend

		elif protocol == StimProtocol.SYN_BACKGROUND_HIGH:
			pass

		elif protocol == StimProtocol.SYN_BACKGROUND_LOW:
			pass

		elif protocol == StimProtocol.SYN_PARK_PATTERNED:
			pass

	def run_sim(self, dur=None):
		"""
		Run NEURON simulator for `dur` or `self.sim_dur` milliseconds
		with precise measurement of runtime
		"""
		if dur is None:
			dur = self.sim_dur

		h.tstop = dur
		h.init() # calls finitialize() and fcurrent()
		t0 = h.startsw()
		h.run()
		t1 = h.startsw()
		h.stopsw() # or t1=h.startsw(); runtime = t1-t0
		logger.debug("Simulation ran for {:.6f} seconds".format(t1-t0))


	def run_protocol(self, protocol, model):
		"""
		Simulate cell in physiological state, under given stimulation protocol
		"""
		
		if protocol == StimProtocol.SPONTANEOUS: # spontaneous firing (no inputs)
			
			# Set simulation parameters
			self.sim_dur = 2000
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

		elif protocol == StimProtocol.SYN_BACKGROUND_HIGH. # synaptic bombardment, high background activity
			pass

		elif protocol == StimProtocol.SYN_BACKGROUND_LOW: # synaptic bombardment, low background activity
			pass

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
	evaluator = StnModelEvaluator(PhysioState.NORMAL, StnModel.Gillies2005)
	evaluator.build_cell(StnModel.Gillies2005)
	evaluator.run_protocol(StimProtocol.SPONTANEOUS)
	evaluator.plot_traces(StimProtocol.SPONTANEOUS)



