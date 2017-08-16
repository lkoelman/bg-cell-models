"""
Evaluation of STN cell models under different physiological and stimulus conditions.
"""

# Standard library modules
import collections
import re
from enum import Enum
import logging
logging.basicConfig(format='%(levelname)s:%(message)s @%(filename)s:%(lineno)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module

# Third party modules
import numpy as np
from scipy import signal

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

# Gillies-Willshaw STN model
import gillies_model as gillies

# Cell reduction
import reduce_marasco as marasco
import mapsyn
# Adjust verbosity of loggers
marasco.logger.setLevel(logging.WARNING)
mapsyn.logger.setLevel(logging.WARNING)

# Plotting & recording
from common import analysis

# Physiological parameters
import cellpopdata as cpd
from cellpopdata import PhysioState, Populations, NTReceptors as NTR, ParameterSource as Cit
Pop = Populations

# Experimental protocols
import proto_common
proto_common.logger = logger
from proto_common import *
import proto_simple_syn as proto_simple
import proto_background


class StnModel(Enum):
	"""
	STN cell models
	"""
	Gillies2005 = 0
	Miocinovic2006 = 1
	Gillies_GIF = 2
	Gillies_BranchZip = 3


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

		# Initialize containers for model data
		for model in list(StnModel):
			self.model_data[model]['rec_data'] = dict(((proto, {}) for proto in StimProtocol))
			self.model_data[model]['rec_segs'] = dict(((proto, {}) for proto in StimProtocol))
			self.model_data[model]['inputs'] = {} # key for each pre-synaptic population

		self.sim_dur = 1000.
		self.sim_dt = 0.025
		self.base_seed = 25031989
		self.rng = np.random.RandomState(self.base_seed)

		# SIGNATURE: make_inputs(evaluator, connector)
		self.MAKE_INPUT_FUNCS = {
			StimProtocol.SYN_BACKGROUND_HIGH : proto_background.make_inputs,
		}

		# SIGNATURE: rec_traces(evaluator, protocol, traceSpecs)
		self.REC_TRACE_FUNCS = {
			StimProtocol.SYN_BACKGROUND_HIGH : proto_background.rec_traces,
		}

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
		# ! All changes should be enacted in build_cell() and make_inputs()
		# Set the state flag
		self._physio_state = state

	def build_cell(self, model, state=None):
		"""
		Build cell model using current physiological state

		@param state		PhysioState enum member
							(if none given, use self.physio_state)
		"""
		if state is None:
			state = self.physio_state

		if self.model_data[model].get('built', False):
			logger.warning("Attempting to build model {} which has already been built.".format(
							model))

		if model == StnModel.Gillies2005:
			somaref, dendrefs = self._build_gillies(state)

		elif model == StnModel.Gillies_BranchZip:
			somaref, dendrefs = self._reduce_map_gillies()

		else:
			raise Exception("Model '{}' not supported".format(
					model))

		# Save Sections
		self.model_data[model]['sec_refs'] = {
			'soma': somaref,
			'dendrites': dendrefs # one list(SectionRef) per dendrite
		}

		# Indicate that given model has been built
		self.model_data[model]['built'] = True
		self.model_data[model]['gid'] = 1 # we only have one cell

		return somaref, dendrefs

	def _build_gillies(self, state):
		"""
		Build the Gillies2005 model.
		"""

		# Make Gillies STN cell
		soma, dends, stims = gillies.stn_cell_gillies()
		somaref, dendL_refs, dendR_refs = gillies.get_stn_refs()
		dendrefs = [dendL_refs, dendR_refs]

		# State-dependent changes to model specification
		if state == PhysioState.NORMAL:
			pass # this case corresponds to default model parameters

		elif (state == PhysioState.PARKINSONIAN or state == rec.PARK_DBS):

			# TODO: decide parameter modifications for DA-depleted state from literature

			# 1. Reduce sKCA channel conductance by 90%, from sources:
			#	- Gillies & Willshaw 2005 (see refs)
			for secref in [somaref] + dendrefs:
				for seg in secrref.sec:
					seg.gk_sKCa = 0.1 * seg.gk_sKCa

			# 2. Modifications to GPE GABA IPSPs
			#	- DONE: see changes in cellpopdata.py
			#	- Changes:
			#		- Increased strength of GABA IPSPs
			#		- longer decay kinetics, 
			#		- increase in number of functional synapses (1 afferent axon has more activated synaptic contacts)
			# 	- References:
			#		- Fan (2012), "Proliferation of External Globus Pallidus-Subthalamic Nucleus Synapses following Degeneration of Midbrain Dopamine Neurons"

			# 3. Modifications to GPE AMPA EPSCs (see hLTP)
			#	- DONE: see changes in cellpopdata.py
			#	- NMDA is involved in this hLTP mechanism

			# 4. Changes to regularity/variability of spontaneous firing (summarize literature)

			if state == rec.PARK_DBS:
				# 5. Neurochemical effects of DBS?
				raise NotImplementedError()

		else:
			raise NotImplementedError('Unrecognized state %s' % repr(state))

		return somaref, dendrefs


	def _reduce_map_gillies(self):
		"""
		Reduce Gillies model and map synapses.
		"""

		full_model = StnModel.Gillies2005
		red_model = StnModel.Gillies_BranchZip

		# Make sure gillies model is built
		if not self.model_data[full_model]['built']:
			logger.info("Building Gillies original model first...")
			self.build_cell(full_model)
		else:
			logger.warn("Gillies STN model will be modified")

		# Restore conductances
		gillies.reset_channel_gbar()

		# Settings for analyzing electrotonic properties
		Z_freq = 25.
		def stn_setstate():
			""" Initialize cell for analyzing electrical properties """
			h.celsius = 35
			h.v_init = -68.0
			h.set_aCSF(4)
			h.init()

		# Get original model Sections
		somaref = self.model_data[full_model]['sec_refs']['soma']
		dendrefs = self.model_data[full_model]['sec_refs']['dendrites']
		allsecrefs = [somaref] + dendrefs[0] + dendrefs[1]
		soma = somaref.sec

		# Get synapses and NetCon we want to map
		cdict = self.model_data[full_model]['inputs']
		pre_pops = cdict.keys()
		syns_tomap = sum((cdict[pop]['synapses'] for pop in pre_pops), [])
		ncs_tomap = sum((cdict[pop]['syn_NetCons'] for pop in pre_pops), [])

		# Get synapse info
		save_ref_attrs = ['table_index', 'tree_index', 'gid'] # SecRef attributes to save
		pop_mapper = {'pre_pop': lambda syn: self.get_pre_pop(syn, full_model)}
		syn_info = mapsyn.get_syn_info(soma, allsecrefs, Z_freq=Z_freq, 
							init_cell=stn_setstate, save_ref_attrs=save_ref_attrs,
							attr_mappers=pop_mapper, syn_nc_tomap=(syns_tomap, ncs_tomap))
		
		# Create reduced cell
		eq_secs, newsecrefs = marasco.reduce_gillies_incremental(
										n_passes=7, zips_per_pass=100)

		# Reassign SectionRef vars
		somaref = next(ref for ref in newsecrefs if 'soma' in ref.sec.name())
		dendrefs = [[ref] for ref in newsecrefs if not 'soma' in ref.sec.name()]

		# Map synapses to reduced cell
		mapsyn.map_synapses(somaref, newsecrefs, syn_info, stn_setstate, Z_freq,
							method='Vratio')

		# Save inputs
		self._init_con_dict(red_model, pre_pops)
		for syndata in syn_info:
			self.model_data[red_model]['inputs'][syndata.pre_pop]['synapses'].append(syndata.mapped_syn)
			self.model_data[red_model]['inputs'][syndata.pre_pop]['syn_NetCons'].extend(syndata.afferent_netcons)

		return somaref, dendrefs

	def reset_handlers(self, pre_pop):
		"""
		Reset Initialize Handlers for given pre-synaptic input.
		"""
		py_handlers = self.model_data[self.target_model]['inputs'][pre_pop].setdefault('PyInitHandlers', [])
		hoc_handlers = []

		for pyh in py_handlers:
			fih = h.FInitializeHandler(pyh)
			hoc_handlers.append(fih)

		self.model_data[self.target_model]['inputs'][pre_pop]['HocInitHandlers'] = hoc_handlers


	def add_inputs(self, pre_pop, model, **kwargs):
		"""
		Add inputs to dict of existing inputs.
		"""
		pop_inputs = self.model_data[model]['inputs'].get(pre_pop, {})

		for input_type, input_objs in kwargs.iteritems():
			if input_type in pop_inputs:
				# Add to list (don't overwrite)
				pop_inputs[input_type].extend(input_objs)
			else:
				pop_inputs[input_type] = input_objs

		self.model_data[model]['inputs'][pre_pop] = pop_inputs

	def get_pre_pop(self, syn, model):
		"""
		Get pre-synaptic population for given synapse.
		"""
		inputs = self.model_data[model]['inputs']
		gen_pop = (pre_pop for (pre_pop, conn_data) in inputs.iteritems() if syn in conn_data['synapses'])
		return next(gen_pop, None)


	def get_num_syns(self, model):
		"""
		Get total number of synapses that have been creates on given model.
		"""
		num_syn = 0
		for pop in list(Populations):
			pops_inputs = self.model_data[model]['inputs']
			if pop in pops_inputs.keys():
				num_syn += len(pops_inputs[pop]['synapses'])
		
		return num_syn

	def _init_con_dict(self, model, pre_pops):
		"""
		Initialize dict with connection data for given model
		and presynaptic populations.
		"""
		for pop in pre_pops:
			self.model_data[model]['inputs'][pop] = {
				'stimweightvec': [],
				'synapses': [],
				'syn_NetCons': [], # NetCons targetting synapses
				'com_NetCons': [], # NetCons for command & control
				'NetStims': [],
				'HocInitHandlers': [],
				'PyInitHandlers': [],
			}


	def make_inputs(self, stim_protocol):
		"""
		Make the inputs for given stimulation protocol.
		"""
		if self.target_model == StnModel.Gillies_BranchZip:
			logger.info("""\
			\nSkip making inputs for model {}.\
			\nInputs for reduced model should have been mapped from full model.
			""".format(StnModel.Gillies_BranchZip))
			return

		elif self.target_model != StnModel.Gillies2005:
			raise Exception("Found target model '{}'. Only model '{}' is supported as a target model").format(
				self.target_model, StnModel.Gillies2005)

		cc = cpd.CellConnector(self.physio_state, self.rng)

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

		elif stim_protocol == StimProtocol.SINGLE_SYN_GABA:
			# Make single GABA synapse
			proto_simple.make_GABA_inputs(self, 1, connector=cc)

		elif stim_protocol == StimProtocol.SINGLE_SYN_GLU:
			# Make single GLU synapse
			proto_simple.make_GLU_inputs(self, 1, connector=cc)

		elif stim_protocol == StimProtocol.MIN_SYN_BURST:
			# Minimal number of GABA + GLU synapses to trigger burst
			proto_simple.make_GABA_inputs(self, 1)
			proto_simple.make_GLU_inputs(self, 4)

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
			new_inputs = {}


			# Make weight signal representing oscillatory pattern
			ctx_timevec = np.arange(0, self.sim_dur, 0.05) # update every 0.05 ms
			ctx_pattern_freq = 8.0 # frequency [Hz]
			ctx_pattern_phase = 0.0 # anti-phase from GPE
			ctx_radvec = 2*np.pi*ctx_pattern_freq*1e-3*ctx_timevec + ctx_pattern_phase
			
			# Set ON and OFF phase
			duty_ms = 50.0 # max is 1/freq * 1e3
			duty = duty_ms / (1./ctx_pattern_freq*1e3)
			ctx_pattern = signal.square(ctx_radvec, duty)
			ctx_pattern[ctx_pattern<0.0] = 0.05 # fractional noise amplitude

			stimweightvec = h.Vector(ctx_pattern)
			stimtimevec = h.Vector(ctx_timevec)

			# Distribute synapses over dendritic trees
			is_ctx_target = lambda seg: seg.diam <= 1.0			
			dendrites = self.model_data[self.target_model]['sec_refs']['dendrites']
			dend_secrefs = sum(dendrites, [])
			ctx_target_segs = pick_random_segments(dend_secrefs, n_ctx_syn, is_ctx_target, rng=self.rng)

			# Make synapses
			for target_seg in ctx_target_segs:

				# Make poisson spike generator
				stim_rate = 80.0
				stimsource = h.NetStim() # Create a NetStim
				stimsource.interval = stim_rate**-1*1e3 # Interval between spikes
				stimsource.number = 1e9 # max number of spikes
				stimsource.noise = 0.25 # Fractional noise in timing
				# stimsource.noiseFromRandom(stimrand) # Set it to use this random number generator

				# TODO SETPARAM: (CTX) set poisson noise & rate parameters dependent on PhysioState
				#	use references for this (reported in vivo firing rates, traces etc)

				# Make synapse and NetCon
				syn, nc, wvecs = cc.make_synapse((Pop.CTX, Pop.STN), (stimsource, target_seg), 
									'GLUsyn', (NTR.AMPA, NTR.NMDA), use_sources=(Cit.Chu2015,), 
									weight_scales=[stimweightvec], weight_times=[stimtimevec])

				# Save inputs
				extend_dictitem(new_inputs, 'synapses', syn)
				extend_dictitem(new_inputs, 'syn_NetCons', nc)
				extend_dictitem(new_inputs, 'NetStims', stimsource)
				extend_dictitem(new_inputs, 'stimweightvec', wvecs)

			# Save inputs to model
			extend_dictitem(new_inputs, 'stimweightvec', stimweightvec)
			extend_dictitem(new_inputs, 'stimtimevec', stimtimevec)
			self.add_inputs('ctx', self.target_model, **new_inputs)

			####################################################################
			# GPe inputs
			####################################################################

			# Add GPe inputs using Tsodyks-Markram synapses
			n_gpe_syn = 10 # NOTE: one synapse represents a multi-synaptic contact from one GPe axon
			new_inputs = {}

			# Make weight signal representing oscillatory pattern
			gpe_timevec = np.arange(0, self.sim_dur, 0.05) # update every 0.05 ms
			gpe_pattern_freq = 8.0 # frequency [Hz]
			gpe_pattern_phase = np.pi # anti-phase from CTX
			gpe_radvec = 2*np.pi*gpe_pattern_freq*1e-3*gpe_timevec + gpe_pattern_phase
			
			# Set ON and OFF phase
			duty_ms = 80.0 # max is 1/freq * 1e3
			duty = duty_ms / (1./gpe_pattern_freq*1e3)
			gpe_pattern = signal.square(gpe_radvec, duty)
			gpe_pattern[gpe_pattern<0.0] = 0.05 # fractional noise amplitude

			stimweightvec = h.Vector(gpe_pattern)
			stimtimevec = h.Vector(gpe_timevec)

			# Pick random segments in dendrites for placing synapses
			is_gpe_target = lambda seg: seg.diam > 1.0 # select proximal dendrites
			dendrites = self.model_data[self.target_model]['sec_refs']['dendrites']
			dend_secrefs = sum(dendrites, [])
			gpe_target_segs = pick_random_segments(dend_secrefs, n_gpe_syn, is_gpe_target, rng=self.rng)

			# Make synapses
			gpe_wvecs = [stimweightvec]
			for target_seg in gpe_target_segs:

				# Make poisson spike generator
				stim_rate = 100.0 # hz
				stimsource = h.NetStim() # Create a NetStim
				stimsource.interval = stim_rate**-1*1e3 # Interval between spikes
				stimsource.number = 1e9 # max number of spikes
				stimsource.noise = 0.25 # Fractional noise in timing
				# stimsource.noiseFromRandom(stimrand) # Set it to use this random number generator

				# TODO SETPARAM: (GPe) set poisson noise & rate parameters, dependent on PhysioState
				# HallworthBevan2005_DynamicallyRegulate: 
				#		Fig 6: 10 pulses @ 100 Hz and 20 pulses @ 100 Hz triiger rebound bursts
				# Bevan2006_CellularPrinciples:
				#		Fig 2: 10 stimuli @ 100 Hz trigger pause + rebound burst

				# Make synapse and NetCon
				syn, nc, wvecs = cc.make_synapse((Pop.GPE, Pop.STN), (stimsource, target_seg), 
									'GABAsyn', (NTR.GABAA, NTR.GABAB), 
									use_sources=(Cit.Chu2015, Cit.Fan2012, Cit.Atherton2013), 
									weight_scales=[stimweightvec], weight_times=[stimtimevec])

				# Save inputs
				extend_dictitem(new_inputs, 'synapses', syn)
				extend_dictitem(new_inputs, 'syn_NetCons', nc)
				extend_dictitem(new_inputs, 'NetStims', stimsource)
				extend_dictitem(new_inputs, 'stimweightvec', wvecs)

			# Save inputs to model
			extend_dictitem(new_inputs, 'stimtimevec', stimtimevec)
			extend_dictitem(new_inputs, 'stimweightvec', stimweightvec)
			self.add_inputs('gpe', self.target_model, **new_inputs)

		else: # standard action: look up in dict

			try:
				make_inputs_func = self.MAKE_INPUT_FUNCS[stim_protocol]
			except KeyError:
				raise NotImplementedError("Make inputs function for protocol {} not implemented".format(stim_protocol))

			make_inputs_func(self, connector=cc)



	def map_inputs(self, cand_model):
		"""
		Map inputs from target model to candidate model.
		"""
		raise NotImplementedError()

	def rec_GABA_traces(self, protocol, traceSpecs, n_syn=1):
		"""
		Set up recording Vectors

		@param n_syn		number of synaptic traces to record
		"""

		rec_segs = self.model_data[self.target_model]['rec_segs'][protocol]
		model = self.target_model
		
		# Add synapse and segment containing it
		nc_list = self.model_data[model]['inputs']['gpe']['syn_NetCons']
		for i_syn, nc in enumerate(nc_list):
			if i_syn > n_syn-1:
				break

			syn_tag = 'GABAsyn%i' % i_syn
			seg_tag = 'GABAseg%i' % i_syn
			rec_segs[syn_tag] = nc.syn()
			rec_segs[seg_tag] = nc.syn().get_segment()

			# Record synaptic variables
			traceSpecs['gA_GABAsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'g_GABAA'}
			traceSpecs['gB_GABAsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'g_GABAB'}
			traceSpecs['Rrp_GABAsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'Rrp'}
			traceSpecs['Use_GABAsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'Use'}
			traceSpecs['Hill_GABAsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'G'}


	def rec_GLU_traces(self, protocol, traceSpecs, n_syn=1):
		"""
		Set up recording Vectors
		"""
		rec_segs = self.model_data[self.target_model]['rec_segs'][protocol]
		model = self.target_model
		
		# Add synapse and segment containing it
		nc_list = self.model_data[model]['inputs']['ctx']['syn_NetCons']
		for i_syn, nc in enumerate(nc_list):
			if i_syn > n_syn-1:
				break

			syn_tag = 'GLUsyn%i' % i_syn
			seg_tag = 'GLUseg%i' % i_syn
			rec_segs[syn_tag] = nc.syn()
			rec_segs[seg_tag] = nc.syn().get_segment()

			# Record synaptic variables
			traceSpecs['gA_GLUsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'g_AMPA'}
			traceSpecs['gN_GLUsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'g_NMDA'}
			traceSpecs['Rrp_GLUsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'R'}
			traceSpecs['Use_GLUsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'Use'}


	def rec_Vm(self, protocol, traceSpecs):
		"""
		Record membrane voltages in all recorded segments
		"""
		rec_segs = self.model_data[self.target_model]['rec_segs'][protocol]
		for seclabel, seg in rec_segs.iteritems():
			if isinstance(seg, neuron.nrn.Segment):
				traceSpecs['V_'+seclabel] = {'sec':seclabel, 'loc':seg.x, 'var':'v'}


	def rec_traces(self, protocol, recordStep=0.025):
		"""
		Set up recording Vectors to record from relevant pointers
		"""
		# Initialize data
		model = self.target_model
		self.model_data[model]['rec_data'][protocol] = {}

		# Specify sections to record from
		if model == StnModel.Gillies2005:

			somasec = h.SThcell[0].soma
			dendsec = h.SThcell[0].dend1[7]

			# Assign label to each recorded section
			rec_segs = {
				'soma': somasec(0.5), # middle of soma
				'dist_dend': dendsec(0.8), # approximate location along dendrite in fig. 5C
			}

		elif model == StnModel.Gillies_BranchZip:

			somasec = self.model_data[model]['sec_refs']['soma'].sec
			dends = self.model_data[model]['sec_refs']['dendrites'] # list of lists
			dendrefs = sum(dends, []) # flatten it
			dendsec = next(ref.sec for ref in dendrefs if not any(ref.sec.children()))

			# Default recorded segments
			rec_segs = {
				'soma': somasec(0.5), # middle of soma
				'dist_dend': dendsec(0.9), # approximate location along dendrite in fig. 5C
			}

		else:
			raise NotImplementedError("""Recording from other models 
					besides {} not yet implemented""".format(StnModel.Gillies2005))

		# Save recorded segments list
		self.model_data[model]['rec_segs'][protocol] = rec_segs

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


		elif protocol == StimProtocol.SINGLE_SYN_GABA:
			
			# Record synaptic variables
			self.rec_GABA_traces(protocol, traceSpecs)

			# Record membrane voltages
			self.rec_Vm(protocol, traceSpecs)

		elif protocol == StimProtocol.SINGLE_SYN_GLU:
			
			# Record synaptic variables
			self.rec_GLU_traces(protocol, traceSpecs)

			# Record membrane voltages
			self.rec_Vm(protocol, traceSpecs)

		elif protocol == StimProtocol.MIN_SYN_BURST:
			# Record both GABA and GLU synapses
			self.rec_GABA_traces(protocol, traceSpecs)
			self.rec_GLU_traces(protocol, traceSpecs)

			# Record membrane voltages
			self.rec_Vm(protocol, traceSpecs)

		elif protocol == StimProtocol.SYN_PARK_PATTERNED: # pathological input, strong patterned cortical input with strong GPi input in antiphase
			####################################################################
			# Record SYN_PARK_PATTERNED
			####################################################################

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
			gpe_ncs = self.model_data[self.target_model]['inputs']['gpe']['syn_NetCons']
			ctx_ncs = self.model_data[self.target_model]['inputs']['ctx']['syn_NetCons']
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

		else: # standard action: look up in dict

			try:
				rec_trace_func = self.REC_TRACE_FUNCS[protocol]
				rec_trace_func(self, protocol, traceSpecs)
			except KeyError:
				raise NotImplementedError("Recording function for protocol {} not implemented".format(stim_protocol))



		# Prepare dictionary (label -> Section)
		rec_secs = {}
		for seclabel, hobj in rec_segs.iteritems():
			if isinstance(hobj, neuron.nrn.Segment):
				rec_secs[seclabel] = hobj.sec
			else:
				rec_secs[seclabel] = hobj # point process

		# Use trace specs to make Hoc Vectors
		recData, markers = analysis.recordTraces(rec_secs, traceSpecs, recordStep)

		# Save trace specs and recording Vectors
		self.model_data[model]['rec_data'][protocol].update({
			'trace_specs': traceSpecs,
			'trace_data': recData,
			'rec_dt': recordStep,
			'rec_markers': markers,
		})
		

	def _plot_all_Vm(self, model, protocol, fig_per='trace'):
		"""
		Plot all membrane voltages.
		"""
		# Get data
		rec_dict = self.model_data[model]['rec_data'][protocol]
		recData, recordStep = (rec_dict[k] for k in ('trace_data', 'rec_dt'))

		# Plot data
		recV = analysis.match_traces(recData, lambda t: t.startswith('V_'))
		figs_vm = analysis.plotTraces(recV, recordStep, yRange=(-80,40), 
										traceSharex=True, oneFigPer=fig_per)
		return figs_vm


	def _plot_GABA_traces(self, model, protocol, fig_per='trace'):
		"""
		Plot GABA synapse traces.
		"""
		# Get data
		rec_dict = self.model_data[model]['rec_data'][protocol]
		recData, recordStep = (rec_dict[k] for k in ('trace_data', 'rec_dt'))

		# Plot data
		syn_traces = analysis.match_traces(recData, lambda t: re.search(r'GABAsyn', t))
		n, KD = h.n_GABAsyn, h.KD_GABAsyn # parameters of kinetic scheme
		hillfac = lambda x: x**n/(x**n + KD)
		analysis.plotTraces(syn_traces, recordStep, traceSharex=True, title='Synaptic variables',
							traceXforms={'Hill_syn': hillfac})


	def _plot_GLU_traces(self, model, protocol, fig_per='trace'):
		"""
		Plot GABA synapse traces.
		"""
		# Get data
		rec_dict = self.model_data[model]['rec_data'][protocol]
		recData, recordStep = (rec_dict[k] for k in ('trace_data', 'rec_dt'))

		# Plot synaptic variables
		syn_traces = analysis.match_traces(recData, lambda t: re.search(r'GLUsyn', t))
		analysis.plotTraces(syn_traces, recordStep, traceSharex=True, title='Synaptic variables')


	def plot_traces(self, protocol, model=None):
		"""
		Plot relevant recorded traces for given protocol
		"""

		# Get recorded data
		if model is None:
			model = self.target_model

		traceSpecs = self.model_data[model]['rec_data'][protocol]['trace_specs']
		recData = self.model_data[model]['rec_data'][protocol]['trace_data']
		recordStep = self.model_data[model]['rec_data'][protocol]['rec_dt']

		# Plot membrane voltages
		

		# Extra plots depending on simulated protocol
		if protocol == StimProtocol.SPONTANEOUS:

			self._plot_all_Vm(model, protocol)

			# Plot ionic currents, (in)activation variables
			figs, cursors = analysis.plot_currents_activations(self.recData, recordStep)

		elif protocol == StimProtocol.CLAMP_PLATEAU:

			self._plot_all_Vm(model, protocol)

			# Plot ionic currents, (in)activation variables
			figs, cursors = analysis.plot_currents_activations(recData, recordStep)

			# Dendrite currents during burst
			recDend = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.endswith('_d')])
			analysis.cumulPlotTraces(recDend, recordStep, timeRange=burst_time)

		elif protocol == StimProtocol.CLAMP_REBOUND:

			self._plot_all_Vm(model, protocol)

			# Plot ionic currents, (in)activation variables
			figs_soma, cursors_soma = analysis.plot_currents_activations(recData, recordStep, sec_tag='soma')
			figs_dend, cursors_dend = analysis.plot_currents_activations(recData, recordStep, sec_tag='dist_dend')
			figs = figs_soma + figs_dend
			cursors = cursors_soma + cursors_dend

		elif protocol == StimProtocol.SYN_BACKGROUND_HIGH:
			
			self._plot_all_Vm(model, protocol) # only plot membrane voltages

		elif protocol == StimProtocol.SYN_BACKGROUND_LOW:
			
			self._plot_all_Vm(model, protocol) # only plot membrane voltages

		elif protocol == StimProtocol.SINGLE_SYN_GABA:
			
			# Plot membrane voltages (one figure)
			self._plot_all_Vm(model, protocol, fig_per='cell')

			# Plot synaptic variables
			self._plot_GABA_traces(model, protocol)

		elif protocol == StimProtocol.SINGLE_SYN_GLU:

			# Plot membrane voltages (one figure)
			self._plot_all_Vm(model, protocol, fig_per='cell')

			# Plot synaptic variables
			self._plot_GLU_traces(model, protocol)

		elif protocol == StimProtocol.MIN_SYN_BURST:

			# Plot membrane voltages (one figure)
			self._plot_all_Vm(model, protocol, fig_per='cell')

			# Plot synaptic variables
			self._plot_GLU_traces(model, protocol)
			self._plot_GABA_traces(model, protocol)


		elif protocol == StimProtocol.SYN_PARK_PATTERNED:

			V_prox = analysis.match_traces(recData, lambda t: t.startswith('V_prox'))
			V_dist = analysis.match_traces(recData, lambda t: t.startswith('V_dist'))
			V_postsyn = analysis.match_traces(recData, lambda t: t.startswith('V_postsyn'))

			analysis.plotTraces(V_prox, recordStep, yRange=(-80,40), traceSharex=True, 
								title='Proximal sections')
			analysis.plotTraces(V_dist, recordStep, yRange=(-80,40), traceSharex=True,
								title='Distal sections')
			analysis.plotTraces(V_postsyn, recordStep, yRange=(-80,40), traceSharex=True,
								title='Post-synaptic segments')

	def run_sim(self, nthread=1):
		"""
		Run NEURON simulator for `dur` or `self.sim_dur` milliseconds
		with precise measurement of runtime
		"""
		# enable multithreaded execution
		if nthread > 1:
			h.cvode_active(0)
			h.load_file('parcom.hoc')
			pct = h.ParallelComputeTool[0]
			pct.nthread(nthread)
			pct.multisplit(1)
			pct.busywait(1)

		# Simulate
		logger.debug("Simulating...")
		t0 = h.startsw()
		h.run()
		t1 = h.startsw()
		h.stopsw() # or t1=h.startsw(); runtime = t1-t0
		logger.debug("Simulated for {:.6f} seconds".format(t1-t0))


	def init_sim(self, dur=2000., dt=0.025, celsius=35., v_init=-60):
		"""
		Initialize simulation.
		"""
		# Set simulation parameters
		self.sim_dur = dur
		h.tstop = dur
		
		self.sim_dt = dt
		h.dt = dt

		h.celsius = celsius # different temp from paper
		h.v_init = v_init # paper simulations sue default v_init
		gillies.set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

		# Initialize NEURON simulator
		h.init() # calls finitialize()


	def _setup_run_proto(self, proto, stdinit=False):
		"""
		Standard simulation function
		"""
		# Make inputs
		self.make_inputs(proto)

		# Set up recording
		self.rec_traces(proto, recordStep=0.05)

		# Initialize
		if stdinit:
			h.stdinit()
		else:
			self.init_sim()

		# Simulate
		self.run_sim()


	def setup_run_protocol(self, protocol, model=None):
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
			h.init()
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
			h.init()
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
			h.init()
			self.run_sim()

		elif protocol == StimProtocol.MIN_SYN_BURST:

			# Change SKCa conductance
			for sec in h.allsec():
				for seg in sec:
					seg.gk_sKCa = 0.6 * seg.gk_sKCa

			self._setup_run_proto(protocol)

		else:
			# Standard simulation function
			self._setup_run_proto(protocol)

################################################################################
# EXPERIMENTS
################################################################################


def map_protocol_MIN_SYN_BURST():
	"""
	Run the MIN_SYN_BURST protocol, then reduce the cell, map the synapses
	and run the protocol again to compare responses.
	"""
	# Make cell model and evaluator
	evaluator = StnModelEvaluator(StnModel.Gillies2005, PhysioState.NORMAL)
	evaluator.build_cell(StnModel.Gillies2005)
	
	# Run protocol
	proto = StimProtocol.MIN_SYN_BURST
	#evaluator.make_inputs(proto)
	evaluator.setup_run_protocol(proto)
	evaluator.plot_traces(proto)

	###################################
	# Model reduction
	evaluator.build_cell(StnModel.Gillies_BranchZip)
	evaluator.target_model = StnModel.Gillies_BranchZip

	# Run Protocol
	evaluator.setup_run_protocol(proto)
	evaluator.plot_traces(proto)


def run_protocol_MIN_SYN_BURST():
	"""
	Try to elicit a rebound burst using synaptic inputs only.

	SEE ALSO
	- notebook file 'test_SYN_BURST_protocol.ipynb'
	"""
	# Make cell model and evaluator
	evaluator = StnModelEvaluator(StnModel.Gillies2005, PhysioState.NORMAL)
	evaluator.build_cell(StnModel.Gillies2005)

	# Run protocol
	proto = StimProtocol.MIN_SYN_BURST
	evaluator.setup_run_protocol(proto)

	# Plot protocol
	evaluator.plot_traces(proto)

	# # Print synapse list
	# h.topology() # see console from which Jupyter was started
	# print("List of synapses:")
	# all_syns = sum((evaluator.model_data[evaluator.target_model]['inputs'][pop]['synapses'] for pop in ['gpe', 'ctx']),[])
	# for syn in all_syns:
	# 	print("Synapse {} in segment {}".format(syn, syn.get_segment()))

	globals().update(locals())


if __name__ == '__main__':
	# Run Gillies 2005 model
	evaluator = StnModelEvaluator(StnModel.Gillies2005, PhysioState.NORMAL)
	evaluator.build_cell(StnModel.Gillies2005)

	proto = StimProtocol.SYN_BACKGROUND_HIGH
	evaluator.setup_run_protocol(proto)

	# Analyze results
	evaluator.plot_traces(proto)

