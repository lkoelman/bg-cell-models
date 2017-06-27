"""
Evaluation of STN cell models under different physiological and stimulus conditions.
"""

# Standard library modules
import logging
logging.basicConfig(format='%(levelname)s:%(message)s @%(filename)s:%(lineno)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module

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
	SPONTANEOUS # spontaneous firing (no inputs)
	CLAMP_PLATEAU # plateau potential (Gillies 2006, Fig. 10C-D)
	CLAMP_REBOUND # rebound burst (Gillies 2006, Fig. 3-4)
	SYN_BACKGROUND_HIGH # synaptic bombardment, high background activity
	SYN_BACKGROUND_LOW # synaptic bombardment, low background activity
	SYN_PARK_PATTERNED # pathological input, strong patterned cortical input with strong GPi input in antiphase

class StnModel(Enum):
	"""
	STN cell models
	"""
	Gillies2005 = 0
	Miocinovic2006 = 1
	Gillies_GIF = 2
	Gilllies_BranchZip = 3


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

	Inspired by optimization.py/StnCellController, optimization.py/Simulation,
	and bgmodel/models/kang/model.py/BGSim
	"""

	def __init__(self, pstate, target_model):
		self.physio_state = pstate
		self.stim_protocol = sproto

		self.model_data = dict((model, {} for model in list(StnModel)))
		self.target_model = target_model

		somaref, dendrefs = self.build_cell(self.target_model)

		self.model_data[self.target_model]['sec_refs'] = {
			'soma': somaref,
			'dendrites': dendrefs # one list(SectionRef) per dendrite
		}

		self.sim_dur = 1000.
		self.sim_dt = 0.025
		self.rng = np.random.RandomState(25031989)
		

	def build_cell(self, model):
		"""
		Build cell model
		"""
		if model == StnModel.Gillies2005:
			# Make Gillies STN cell
			soma, dends, stims = gillies.stn_cell_gillies()
			somaref, dendL_refs, dendR_refs = gillies.get_stn_refs()
			dendrefs = [dendL_refs, dendR_refs]

		elif model == StnModel.Gilllies_BranchZip:
			# Incremental reduction with 'Branch Zipping' algorithm
			logger.warn("Gillies STN model will be modified if created")
			eq_secs, newsecrefs = marasco.reduce_gillies_incremental(n_passes=7, zips_per_pass=100)
			somaref = next(ref for ref in newsecrefs if 'soma' in ref.sec.name())
			dendrefs = [[ref] for ref in newsecrefs if not 'soma' in ref.sec.name()]

		else:
			raise Exception("Model '' not supported".format(
					model))

		return somaref, dendrefs


	def make_inputs(self, stim_protocol):
		"""
		Make the inputs for given stimulation protocol.
		"""
		if self.target_model != StnModel.Gillies2005:
			raise Exception("Found target model '{}'. Only model '{}' is supported as a target model").format(
				self.target_model, StnModel.Gilllies2005)

		if stim_protocol == StimProtocol.SPONTANEOUS:
			logger.debug("Spontaneous firing has no inputs. Skipping.")

		elif stim_protocol == StimProtocol.CLAMP_PLATEAU:

		elif stim_protocol == StimProtocol.CLAMP_REBOUND:

		elif stim_protocol == StimProtocol.SYN_BACKGROUND_HIGH:

		elif stim_protocol == StimProtocol.SYN_BACKGROUND_LOW:

		elif stim_protocol == StimProtocol.SYN_PARK_PATTERNED:
			# Allocate data for input
			self.model_data[self.target_model]['inputs'] = {}

			# Add cortical input
			# Method:
			# - add netstim and play input signal into weight (see Dura-Berndal arm example)
			# - advantage: low weight in off-period will provide background noise
			# - if no input desired in off-period: periodically turn it on/off using events or other NetStim
			
			n_ctx_syn = 10
			ctx_syns = []
			ctx_ncs = []
			ctx_stims = []

			# Make weight signal
			stimtimevec = np.arange(0, self.sim_dur, 0.05) # update every 0.05 ms
			ctx_pattern_freq = 20.0 # frequency [Hz]
			pattern = np.sin(2*pi*ctx_pattern_freq*1e-3*stimtimevec) # amplitude 1.0
			pattern[pattern<=0] = 0.05 # fractional noise amplitude
			gmax = 0.0015 # synaptic conductance increment [uS], see exp2syn.mod
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
				asyn.tau1 = 1.0 + rng.rand()*50 # EPFL course video 1.4.1
				asyn.tau2 = 19.0 + rng.rand()*30.
				asyn.e = 0.0
				ctx_syns.append(asyn)

				# Make poisson spike generator
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
			gmax = 0.0015 # synaptic conductance increment [uS], see exp2syn.mod
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
				asyn.tau1 = 1.0 + rng.rand()*50 # EPFL course video 1.4.1
				asyn.tau2 = 19.0 + rng.rand()*30. # EPFL course video 1.4.1
				asyn.e = 0.0
				gpe_syns.append(asyn)

				# Make poisson spike generator
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



	def map_inputs(self):
		"""
		Map inputs from target model to candidate model.
		"""
		pass

	def run_candidate(cand_model, cand_protocol):
		"""
		Simulate cell in physiological state, under given stimulation protocol
		"""
		pass
