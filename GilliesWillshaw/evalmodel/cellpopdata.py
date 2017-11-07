"""
Parameters for STN connections

@author		Lucas Koelman
@date		12/07/2017


Based on structure of:

	bgmodel/models/kang/nuclei/cellpopdata.py
	bgmodel/models/kumaravelu/nuclei/cellpopdata.py
	bgmodel/models/netkang/kang_netparams.py

"""

from enum import Enum, IntEnum, unique
import types
import re
from textwrap import dedent


import logging
logging.basicConfig(format='%(levelname)s:%(message)s @%(filename)s:%(lineno)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module

import numpy as np
import neuron
nrn = neuron.nrn # types nrn.Section and nrn.Segment
h = neuron.h

# Load NEURON mechanisms
import os.path
scriptdir, scriptfile = os.path.split(__file__) 
NRN_MECH_PATH = os.path.normpath(os.path.join(scriptdir, '..', 'nrn_mechs')) 
neuron.load_mechanisms(NRN_MECH_PATH) 

from common.conutils import interpretParamSpec

@unique
class PhysioState(IntEnum):
	"""
	Physiological state of the cell
	"""

	# bit flags for states
	PARKINSONIAN =		2**0
	DBS =				2**1
	AWAKE =				2**2
	SLEEP_SWS =			2**3

	# Combinations
	NORMAL = 0
	PARK_DBS = PARKINSONIAN | DBS
	# ALL = 2**32 - 1

	def __contains__(self, item):
		"""
		Bitwise operation to detect if item is subset/member of this state

		NOTE: this only works when item is an Enum item, e.g. check using
		      expression (item_A in item_B)
		"""
		return  (self.value & item.value) == item.value

	def is_subset(self, int_item):
		"""
		Check if this item is a subset of item described by bitflags
		in given integer 'int_item'

		NOTE: this can be used for checking bitflags in an Enum item
		      againt bitflags in an aribrary integer, e.g.: item.is_subset(8-1)
		"""
		return (self.value & int_item) == self.value

	@classmethod
	def from_descr(cls, descr):
		return cls._member_map_[descr.upper()]

	@classmethod
	def from_str(cls, descr):
		return cls._member_map_[descr.upper()]


@unique
class Populations(Enum):
	"""
	Physiological state of the cell
	"""
	STN = 0
	CTX = 1
	GPE = 2
	THA = 4
	PPN = 5

	def to_descr(self):
		return self.name.lower()

	@classmethod
	def from_descr(cls, descr):
		return cls._member_map_[descr.upper()]


@unique
class NTReceptors(Enum):
	"""
	NTReceptors used in synaptic connections
	"""
	AMPA = 0
	NMDA = 1
	GABAA = 2
	GABAB = 3

	@classmethod
	def from_descr(cls, descr):
		return cls._member_map_[descr.upper()]


@unique
class MSRCorrection(Enum):
	"""
	Correction method to take into account multi-synaptic contacts
	(Multi Synapse Rule).
	"""
	SCALE_GSYN_MSR = 0,			# Divide synaptic conductance by average number of contacts
	SCALE_NUMSYN_MSR = 1,		# Number of SYNAPSE objects = number of synapses (observed) divided by average number of contacts
	SCALE_NUMSYN_GSYN = 2,		# Number of SYNAPSE objects = number needed to get total observed synaptic conductance

# MSRC = MSRCorrection

@unique
class StnModel(Enum):
	"""
	STN cell models
	"""
	Gillies2005 = 0
	Miocinovic2006 = 1
	Gillies_GIF = 2
	Gillies_BranchZip = 3
	Gillies_FoldStratford = 4
	Gillies_FoldMarasco = 5

# Indicate which of the STN models are reduced models
ReducedModels = (StnModel.Gillies_GIF, StnModel.Gillies_BranchZip, 
					StnModel.Gillies_FoldMarasco, StnModel.Gillies_FoldStratford)

@unique
class ParameterSource(Enum):
	Default = 0
	CommonUse = 1		# Widely used or commonly accepted values
	Chu2015 = 2
	Baufreton2009 = 3
	Fan2012 = 4
	Atherton2013 = 5
	Kumaravelu2016 = 6
	Custom = 7			# Custom parameters, e.g. for testing
	Gradinaru2009 = 8
	Mallet2016 = 9 # Mallet (2016), Neuron 89
	Nambu2014 = 10
	Bergman2015RetiCh3 = 11


# Shorthands
Pop = Populations
Rec = NTReceptors
Src = ParameterSource


def correct_GABAsyn(syn):
	"""
	Correct parameters of GABAsyn so peak synaptic conductance
	is equal to value of 'gmax_' parameters
	"""

	# Compensate for effect max value Hill factor and U1 on gmax_GABAA and gmax_GABAB
	if isinstance(syn, dict):
		syn['pointprocess']['gmax_GABAA'] /= syn['pointprocess']['U1']
		syn['pointprocess']['gmax_GABAB'] /= 0.21
	else:
		syn.gmax_GABAA = syn.gmax_GABAA / syn.U1
		syn.gmax_GABAB = syn.gmax_GABAB / 0.21


def correct_GLUsyn(syn):
	"""
	Correct parameters of GLUsyn so peak synaptic conductance
	is equal to value of 'gmax_' parameters
	"""

	# Compensate for effect max value Hill factor and U1 on gmax_GABAA and gmax_GABAB
	if isinstance(syn, dict):
		syn['pointprocess']['gmax_AMPA'] /= syn['pointprocess']['U1']
		syn['pointprocess']['gmax_NMDA'] /= syn['pointprocess']['U1']
	else:
		syn.gmax_AMPA = syn.gmax_AMPA / syn.U1
		syn.gmax_NMDA = syn.gmax_NMDA / syn.U1

# Parameter correction functions for synaptic mechanisms
syn_mech_correctors = {
	'GABAsyn' : correct_GABAsyn,
	'GLUsyn' : correct_GLUsyn,
}

class SynInfo(object):
	""" Synapse info bunch/struct """
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)


def get_synapse_data(connector, synapses, netcons):
	"""
	Get list of SynInfo properties containing a reference
	to each synapse, its NetCon and pre-synaptic population.

	@param	synapses	list(Synapse) where Synapse is a wrapped HocObject with
						attributes 'pre_pop' and 'post_pop'

	@return				list(namedtuple)
	"""
	
	syn_list = []

	# Save properties
	for syn in synapses:

		# Get connection parameters
		pre = syn.pre_pop
		post = syn.post_pop
		if not isinstance(pre, Populations):
			pre = Populations.from_descr(pre)
		if not isinstance(post, Populations):
			post = Populations.from_descr(post)

		con_par = connector.getPhysioConParams(pre, post, [Src.Default]) 
		syn_info = SynInfo()

		# HocObjects
		syn_info.orig_syn = syn
		syn_info.afferent_netcons = [nc for nc in netcons if nc.syn().same(syn)]
		
		# meta-information
		syn_info.pre_pop = pre

		# For PSP frequency: need to find the receptor types that this synaptic mechanism implements
		modname = get_mod_name(syn)
		syn_info.mod_name = modname

		syn_receptors = connector.getSynMechReceptors(modname)
		freqs = [con_par[receptor]['f_med_PSP_burst'] for receptor in syn_receptors]
		syn_info.PSP_median_frequency = max(freqs)

		# gbar parameters that need to be scaled
		syn_info.gbar_param_specs = []
		ntrs_params = getNrnConParamMap(modname)
		for ntr, syn_param_specs in ntrs_params.iteritems():
			if 'gbar' in syn_param_specs:
				syn_info.gbar_param_specs.append(syn_param_specs['gbar'])

		# Save SynInfo object
		syn_list.append(syn_info)

	return syn_list


#######################################################################
# Parameter mapping for synaptic mechanisms
#######################################################################

# mapping synapse type -> parameter names
syn_par_maps = {
	'GABAsyn' : {}, # see below
	'GLUsyn' : {}, # see below
	'Exp2Syn' : {}, # see below
}

# Wrapper clases allow setting arbitrary attributes
class Exp2SynWrapper(neuron.hclass(h.Exp2Syn)):
	pass
class GABAsynWrapper(neuron.hclass(h.GABAsyn)):
	pass
class GLUsynWrapper(neuron.hclass(h.GLUsyn)):
	pass
class NetConWrapper(neuron.hclass(h.NetCon)):
	pass

syn_wrapper_classes = {
	'GABAsyn' : GABAsynWrapper,
	'GLUsyn' : GLUsynWrapper,
	'Exp2Syn' : Exp2SynWrapper,
}
# Dynamic creation:
# syn_wrapper_classes = {
# 	modname: type(modname+'Wrapper',(neuron.hclass(getattr(h, modname)),), {})
# 		for modname in syn_par_maps.keys()
# }


# GABAsyn.mod cam be used for GABA-A and GABA-B receptor
syn_par_maps['GABAsyn'] = {
	Rec.GABAA : {
		'Erev': 'syn:Erev_GABAA',
		'tau_rise_g': 'syn:tau_r_GABAA',
		'tau_decay_g': 'syn:tau_d_GABAA',
		'gbar': 'syn:gmax_GABAA',
		'delay': 'netcon:delay',
		'Vpre_threshold': 'netcon:threshold',
		'tau_rec_STD': 'syn:tau_rec', # NOTE: GABAA & GABAB use shared vars for depression/facilitation
		'tau_rec_STP': 'syn:tau_facil',
		'P_release_base': 'syn:U1', # initial release probability (fraction of vesicles in RRP released initially)
	},
	Rec.GABAB : {
		'Erev': 'syn:Erev_GABAB',
		'tau_rise_NT': 'syn:tau_r_GABAB', # in GABAsyn.mod, tau_r represents rise time of NT concentration that kicks off signaling cascade
		'tau_decay_NT': 'syn:tau_d_GABAB', # in GABAsyn.mod, tau_d represents decay time of NT concentration that kicks off signaling cascade
		'gbar': 'syn:gmax_GABAB',
		'delay': 'netcon:delay',
		'Vpre_threshold': 'netcon:threshold',
	},
}

# GABAsyn.mod cam be used for GABA-A and GABA-B receptor
syn_par_maps['GLUsyn'] = {
	Rec.AMPA : {
		'Erev': 'syn:e',
		'tau_rise_g': 'syn:tau_r_AMPA',
		'tau_decay_g': 'syn:tau_d_AMPA',
		'gbar': 'syn:gmax_AMPA',
		'delay': 'netcon:delay',
		'Vpre_threshold': 'netcon:threshold',
		'tau_rec_STD': 'syn:tau_rec', # NOTE: AMPA & NMDA use shared vars for depression/facilitation
		'tau_rec_STP': 'syn:tau_facil',
		'P_release_base': 'syn:U1',
	},
	Rec.NMDA : {
		'Erev': 'syn:e', # AMPA,NMDA have same reversal potential
		'tau_rise_g': 'syn:tau_r_NMDA',
		'tau_decay_g': 'syn:tau_d_NMDA',
		'gbar': 'syn:gmax_NMDA',
		'delay': 'netcon:delay',
		'Vpre_threshold': 'netcon:threshold',
	},
}

# Exp2Syn.mod can be used for any receptor
exp2syn_parmap = {
	'Erev': 'syn:e',
	'tau_rise_g': 'syn:tau1',
	'tau_decay_g': 'syn:tau2',
	'gbar': 'netcon:weight[0]',
	'delay': 'netcon:delay',
	'Vpre_threshold': 'netcon:threshold',
	
}
for rec in list(NTReceptors):
	syn_par_maps['Exp2Syn'][rec] = dict(exp2syn_parmap)


def getNrnConParamMap(mech_name):
	"""
	For given synaptic mechanism (POINT_PROCESS defined in .mod file),
	get mapping from parameter name in dict getPhysioConParams()
	to parameters of the synaptic mechanism and NetCon.

	In other words: how each key in the parameter dictionary
	should be interpreted.
	"""
	return dict(syn_par_maps[mech_name]) # return a copy


def get_mod_name(syn):
	"""
	Get NEURON mechanism name of given synapse object

	@param	syn		HocObject: synapse POINT_PROCESS
	"""
	if hasattr(syn, 'htype'):
		hoc_name = syn.htype.hname() # for wrapped HocObject, mechanism name is in htype attribute
	else:
		hoc_name = syn.hname()
	match_mod = re.search(r'^[a-zA-Z0-9]+', hoc_name)
	modname = match_mod.group()
	return modname


def getSynMechReceptors(mech_name):
	"""
	Get receptor types implemented by the given synaptic mechanism.

	@param	mech_name	str: name of the NEURON mechanism

	@return				list(NTReceptors)
	"""
	return syn_par_maps[mech_name].keys()


def getSynMechParamNames(mech_name):
	"""
	Get parameter names for synaptic mechanism.

	I.e. the parameter names defined in the .mod file that can be changed.

	@return		list(str): list of parameter names declared in mod file
	"""
	ntr_params_names = syn_par_maps[mech_name]
	mech_parnames = []

	# Get all parameter names prefixed by 'syn:'
	for ntr, params_names in ntr_params_names.iteritems():
		for pname in params_names.values():
			matches = re.search(r'^(?P<mech>\w+):(?P<parname>\w+)(\[(?P<idx>\d+)\])?', pname)
			mechtype = matches.group('mech')
			if mechtype == 'syn':
				parname = matches.group('parname')
				mech_parnames.append(parname)

	return mech_parnames


def evalValueSpec(value_spec, rng=None):
	"""
	Evaluate specification of a parameter value in a range of formats.

	@param value_spec	numeric value, function, or dict with parameters of distribution

	@param rng			numpy random object
	
	@return				float
	"""
	if rng is None:
		rng = np.random

	if isinstance(value_spec, (float, int)):
		# parameter is numerical value
		value = value_spec

	elif isinstance(value_spec, types.FunctionType):
		# parameter is a function
		value = value_spec()

	elif isinstance(value_spec, dict):
		# parameter is described by other parameters (e.g. distribution)

		if ('min' in value_spec and 'max' in value_spec):
			lower = value_spec['min']
			upper = value_spec['max']
			value = lower + rng.rand()*(upper-lower)

		elif ('mean' in value_spec and 'deviation' in value_spec):
			lower = value_spec['mean'] - value_spec['deviation']
			upper = value_spec['mean'] + value_spec['deviation']
			value = lower + rng.rand()*(upper-lower)

		elif ('mean' in value_spec and 'stddev' in value_spec):
			value = rng.normal(value_spec['mean'], value_spec['stddev'])

		else:
			raise ValueError('Could not infer distribution from parameters in {}'.format(value_spec))
	return value


class CellConnector(object):
	"""
	Class for storing connection parameters and making connections.
	"""

	def __init__(self, physio_state, rng):
		if isinstance(physio_state, str):
			physio_state = PhysioState.from_descr(physio_state)
		self._physio_state = physio_state
		self._rng = rng

		# make some functions available as instance methods (lazy refactoring)
		self.getSynMechReceptors = getSynMechReceptors
		self.getSynMechParamNames = getSynMechParamNames
		self.getNrnConParamMap = getNrnConParamMap

	def getFireParams(self, pre_pop, phys_state, use_sources, custom_params=None):
		"""
		Get parameters describing firing statistics for given population.
		
		@param use_sources		ordered list of ParameterSource members
								to indicate which literatur sources should
								be used.
		"""

		firing_params = dict(((pop, {}) for pop in list(Populations)))
		fp = firing_params

		# ---------------------------------------------------------------------
		# parameters from Bergman (2015)
		fp[Pop.GPE][Src.Bergman2015RetiCh3] = {}

		fp[Pop.GPE][Src.Bergman2015RetiCh3][PhysioState.NORMAL | PhysioState.AWAKE] = {
			'rate_mean': 60.0, # 50-70 Hz
			'rate_deviation': 10.0,
			'rate_units': 'Hz',
			'pause_dur_mean': 0.6, # average pause duration 0.5-0.7 s in monkeys
			'pause_dur_units': 's',
			'pause_rate_mean': 15.0/60.0, # 10-20 pauses per minute
			'pause_rate_units': 'Hz',
			'pause_rate_dist': 'poisson', # Can control rate NetStim with pause NetStim (stim.number resets, see mod file)
			# NOTE: discarge_dur = 1/pause_rate - pause_dur ~= 4 - 0.6
			'discharge_dur_mean': 1.0, # must be < 1/pause_rate_mean
			'discharge_dur_units': 's',
			'species': 'primates',
		}


		fp[Pop.CTX][Src.Bergman2015RetiCh3] = {}

		fp[Pop.CTX][Src.Bergman2015RetiCh3][PhysioState.NORMAL | PhysioState.AWAKE] = {
			'rate_mean': 2.5,
			'rate_deviation': 2.5,
			'rate_dist': 'poisson',
			'rate_units': 'Hz',
			'species': 'primates',
		}

		# ---------------------------------------------------------------------
		# parameters from Mallet (2016)
		fp[Pop.GPE][Src.Mallet2016] = {}

		fp[Pop.GPE][Src.Mallet2016][PhysioState.NORMAL | PhysioState.AWAKE] = {
			'rate_mean': 47.3,
			'rate_deviation': 6.1,
			'rate_units': 'Hz',
			'species': 'rats',
		}

		# ---------------------------------------------------------------------
		# parameters from Nambu (2014)
		fp[Pop.GPE][Src.Nambu2014] = {}

		fp[Pop.GPE][Src.Nambu2014][PhysioState.NORMAL | PhysioState.AWAKE] = {
			'rate_mean': 65.2,
			'rate_deviation': 25.8,
			'rate_units': 'Hz',
			'species': 'monkey',
			'subspecies': 'macaque',
		}

		# Get final parameters
		fp_final = {} # {Population -> params}
		use_sources = list(use_sources)


		# Use preferred sources to update final parameters
		for citation in reversed(use_sources):

			if citation in firing_params[pre_pop]:

				# Update params if requested Physiological state matches the described state
				for state_described, params in firing_params[pre_pop][citation].iteritems():
					if phys_state.is_subset(state_described):
						fp_final.update(params)

			elif (citation == Src.Custom) and (custom_params is not None):
				# If custom params provided, and custom is in list of preferred params: use them
				fp_final.update(custom_params)

		return fp_final

	def getPhysioConParams(self, pre_pop, post_pop, use_sources, custom_params=None,
					adjust_gsyn_msr=None):
		"""
		Get parameters for afferent connections onto given population,
		in given physiological state.

		@param custom_params	custom parameters in the form of a dict
								{NTR_0: {params_0}, NTR_1: {params_1}} etc.


		@param adjust_gsyn_msr		Number of synapses per axonal contact (int), 
									and whether the synaptic condcutance for each synapse
									should be scaled to take this into account.
									If an int > 0 is given, each synaptic condcutance
									is divided by this number.
		"""

		physio_state = self._physio_state
		rng = self._rng

		if rng is None:
			rng = np.random

		# Initialize parameters dict
		cp = {}

		if post_pop == Pop.STN:

			cp[Pop.CTX] =  {
				Rec.AMPA: {},
				Rec.NMDA: {},
			}

			cp[Pop.GPE] = {
				Rec.GABAA: {},
				Rec.GABAB: {},
			}

			# TODO: correct both gmax/Imax for attenuation from dendrites to soma. 
			#		Do this separately for GPe and CTX inputs since they have to travel different path lengths.

			# TODO SETPARAM: make sure default delay, Vpre etc. is set for each (pre,post,rec) combination

			# gmax calculation:
			# gmax is in [uS] in POINT_PROCESS synapses
			# 	1 [uS] * 1 [mV] = 1 [nA]
			# 	we want ~ -300 [pA] = -0.3 [nA]
			# 	gmax [uS] * (Ermp [mV] - Erev [mV]) = Isyn [nA]
			#
			# 		=> gmax [uS] = Isyn [nA] / (Ermp-Erev) [mV]

			#######################################################################
			# CTX -> STN parameters
			#######################################################################

			Erev = 0.

			# Default parameters
			for ntr in cp[Pop.CTX].keys():
				cp[Pop.CTX][ntr][Src.Default] = {
					'delay': 1.0,
					'Vpre_threshold': 0.0,
				}

			cp[Pop.CTX][Rec.AMPA][Src.Default].update({
				'f_med_PSP_single': 16.87,
				'f_med_PSP_burst': 16.95,
			})

			cp[Pop.CTX][Rec.NMDA][Src.Default].update({
				'f_med_PSP_single': 0.58,
				'f_med_PSP_burst': 2.14,
			})

			# ---------------------------------------------------------------------
			# AMPA from Chu (2015)
			Ermp = -80.
			cp[Pop.CTX][Rec.AMPA][Src.Chu2015] = {
				'Ipeak': -275., 	# peak synaptic current (pA)
				'gbar': -275.*1e-3 / (Ermp - Erev), # gbar calculation (~ 3e-3 uS)
				'tau_rise_g': 1.0,
				'tau_decay_g': 4.0,
				'Erev': Erev,
			}

			# Changes in parkinsonian state
			if physio_state == PhysioState.PARKINSONIAN:
				Ermp = -80. # Fig. 2G
				cp[Pop.CTX][Rec.AMPA][Src.Chu2015].update({
					'Ipeak': -390.,
					'gbar': -390.*1e-3 / (Ermp - Erev),	
				})
			park_gain = 390./275. # increase in Parkinsonian condition

			# NMDA from Chu (2015)
			Ermp = 30. # NMDA EPSC measured at 30 mV rmp to remove Mg2+ block
			I_peak = 210. # opposite sign: see graph
			cp[Pop.CTX][Rec.NMDA][Src.Chu2015] = {
				'Ipeak': I_peak, 	# peak synaptic current (pA)
				'gbar': I_peak * 1e-3 / (Ermp - Erev), # gbar calculation
				'tau_rise_g': 3.7,
				'tau_decay_g': 80.0,
				'Erev': Erev,
			}

			# Changes in parkinsonian state
			if physio_state == PhysioState.PARKINSONIAN:
				Ermp = -80. # Fig. 2G
				# NOTE: increase peak conductance by same factor as AMPA
				cp[Pop.CTX][Rec.NMDA][Src.Chu2015].update({
					'Ipeak': park_gain * cp[Pop.CTX][Rec.NMDA][Src.Chu2015]['Ipeak'],
					'gbar':  park_gain * cp[Pop.CTX][Rec.NMDA][Src.Chu2015]['Ipeak'],	
				})

			# ---------------------------------------------------------------------
			# Parameters Gradinaru (2009)

			cp[Pop.CTX][Rec.AMPA][Src.Gradinaru2009] = {
				'tau_rec_STD': 200.,
				'tau_rec_STP': 1., # no facilitation
				'P_release_base': 0.7,
			}


			# ---------------------------------------------------------------------
			# Default parameters

			# Set params Chu (2015) as default parameters
			cp[Pop.CTX][Rec.AMPA][Src.Default].update(cp[Pop.CTX][Rec.AMPA][Src.Chu2015])
			cp[Pop.CTX][Rec.NMDA][Src.Default].update(cp[Pop.CTX][Rec.NMDA][Src.Chu2015])
			
			# Modification to NMDA conductance
			#	-> NMDA conductance is typically 70% of that of AMPA (see EPFL MOOC)
			cp[Pop.CTX][Rec.NMDA][Src.Default]['gbar'] = 0.7 * cp[Pop.CTX][Rec.AMPA][Src.Default]['gbar']
			
			
			
			#######################################################################
			# GPe -> STN parameters
			#######################################################################

			# ---------------------------------------------------------------------
			# Default parameters
			Erev_GABAA = -85.
			Erev_GABAB = -93.
			Ermp = -70.

			cp[Pop.GPE][Rec.GABAA][Src.Default] = {
				'Erev': Erev_GABAA,
				'delay': 1.0,
				'Vpre_threshold': 0.0,
				'f_med_PSP_single': 0.19, # median frequency of PSP triggered by burst
				'f_med_PSP_burst': 0.05, # median frequency of PSP triggered by single spike
			}

			cp[Pop.GPE][Rec.GABAB][Src.Default] = {
				'Erev': Erev_GABAB,
				'gbar': 350.*1e-3 / (Ermp - Erev_GABAB), # Chu2015 in healthy state
				'delay': 1.0,
				'Vpre_threshold': 0.0,
				'f_med_PSP_single': 4.22,
				'f_med_PSP_burst': 8.72,
			}

			# ---------------------------------------------------------------------
			# Parameters from Chu (2015)
			cp[Pop.GPE][Rec.GABAA][Src.Chu2015] = {
				'Ipeak': 350., 	# peak synaptic current (pA)
				'gbar': 350.*1e-3 / (Ermp - Erev_GABAA), # gbar calculation
				'tau_rise_g': 2.6,
				'tau_decay_g': 5.0,
			}
			if physio_state == PhysioState.PARKINSONIAN:
				cp[Pop.GPE][Rec.GABAA][Src.Chu2015].update({
					'Ipeak': 450.,
					'gbar': 450.*1e-3 / (Ermp - Erev_GABAA),
					'tau_rise_g': 3.15,
					'tau_decay_g': 6.5,
				})

			# ---------------------------------------------------------------------
			# Parameters from Fan (2012)
			cp[Pop.GPE][Rec.GABAA][Src.Fan2012] = {
				'gbar': {
					'mean': 7.03e-3,
					'deviation': 3.10e-3,
					'units': 'uS'
					},
				'tau_rise_g': 1.0,
				'tau_decay_g': 6.0,
			}
			if physio_state == PhysioState.PARKINSONIAN:
				cp[Pop.GPE][Rec.GABAA][Src.Fan2012].update({
					'gbar': {
						'mean': 11.17e-3,
						'deviation': 5.41e-3,
						'units': 'uS'
						},
					'tau_rise_g': 1.0,
					'tau_decay_g': 8.0,
				})

			# ---------------------------------------------------------------------
			# Parameters from Atherton (2013)
			cp[Pop.GPE][Rec.GABAA][Src.Atherton2013] = {
				'tau_rec_STD': 1730., # See Fig. 2
				'tau_rec_STP': 1.0, # no STP: very fast recovery
				'P_release_base': 0.5, # Fitted
			}

			# ---------------------------------------------------------------------
			# Parameters from Baufreton (2009)
			cp[Pop.GPE][Rec.GABAA][Src.Baufreton2009] = {
				'Ipeak': {
						'min': 375.,
						'max': 680.,
						'units': 'pA',
					},
				'gbar': 530.*1e-3 / (Ermp - Erev_GABAA), # gbar calculation
				'tau_rise_g': 2.6,
				'tau_decay_g': 5.0,
			}

		# Make final dict with only (receptor -> params)
		cp_final = {}
		use_sources = list(use_sources)
		use_sources.append(Src.Default) # this source was used for default parameters
		use_sources.append(Src.CommonUse) # source used for unreferenced parameters from other models

		for receptor in cp[pre_pop].keys(): # all receptors involved in this connection
			cp_final[receptor] = {}

			# Use preferred sources to update final parameters
			for citation in reversed(use_sources):
				
				if ((citation == Src.Custom) and (custom_params is not None)
					and (receptor in custom_params)):
					ntr_params = custom_params[receptor]

				elif (citation in cp[pre_pop][receptor]):
					ntr_params = cp[pre_pop][receptor][citation]

				else:
					continue # citation doesn't provide parameters about this receptor

				# Successively overwrite with params of each source
				cp_final[receptor].update(ntr_params)

			# Adjust for multi synapse rule
			if (adjust_gsyn_msr) and ('gbar' in cp_final[receptor]):
				cp_final[receptor]['gbar'] = cp_final[receptor]['gbar']/adjust_gsyn_msr

		return cp_final


	def getNrnObjParams(self, nrn_mech_name, physio_params):
		"""
		Get parameters to assign to NEURON objects.

		@return		dictionary {nrn_obj_type: {attr_name: value} }
					
					where nrn_obj_type is one of:

						- 'pointprocess',
						- 'netcon'
						- 'netstim'

					and the inner dictionary are parameter names and values for 
					these object types.

		USAGE:

			physio_params = cc.getPhysioConParams(pre_pop, post_pop, use_sources)
			mech = 'Exp2Syn'
			nrn_params = cc.getNrnObjParams(mech, physio_params)
		
		"""
		# Get NT Receptors used in connection
		receptors = physio_params.keys()

		# Get mapping from physiological to NEURON mechanism parameters
		nrn_param_map = getNrnConParamMap(nrn_mech_name)

		# keep track of parameters that are assigned
		nrn_param_values = {
			'pointprocess':	{},
			'netcon':		{},
			'netstim':		{},
		}

		# For each NT receptor, look up the physiological connection parameters,
		# and translate them to a parameter of the synaptic mechanism or NetCon
		for rec in receptors:

			physio_to_nrn = nrn_param_map[rec] 
			physio_to_values = physio_params[rec]

			# Translate each physiological parameter to NEURON parameter
			for physio_name, nrn_param_spec in physio_to_nrn.iteritems():

				# Check if parameters is available from given sources
				if not physio_name in physio_to_values:
					logger.anal("Parameter {dictpar} not found for connection. "
								"This means that parameter {mechpar} will not be set\n".format(
								dictpar=physio_name, mechpar=nrn_param_spec))
					continue

				mech_type, param_name, param_index = interpretParamSpec(nrn_param_spec)

				# Get the actual parameter value
				value_spec = physio_to_values[physio_name]
				value = evalValueSpec(value_spec, self._rng)

				# Determine target (synapse or NetCon)
				if mech_type == 'syn':
					nrn_dict = nrn_param_values['pointprocess']
				else:
					nrn_dict = nrn_param_values[mech_type]

				# Set attribute
				if param_index is None:
					nrn_dict[param_name] = value
				else:
					values = nrn_dict.get(param_name, None)
					if values is None:
						values = [0.0] * (param_index+1)
						nrn_dict[param_name] = values
					elif len(values) <= param_index:
						values.extend([0.0] * (param_index+1-len(values)))
					values[param_index] = value

		# Apply possible corrections to synaptic parameters
		correct_syn = syn_mech_correctors.get(nrn_mech_name, None)
		if correct_syn is not None:
			correct_syn(nrn_param_values)

		return nrn_param_values


	def make_synapse(self, pre_post_pop, pre_post_obj, syn_type, receptors, 
						use_sources=None, custom_conpar=None, custom_synpar=None,
						con_par_data=None, weight_scales=None, weight_times=None):
		"""
		Insert synapse POINT_PROCESS in given section.


		USAGE:

		- either provide argument 'use_sources' to fetch connection parameters
		  or provide them yourself in argument 'con_par_data'

		
		ARGUMENTS:
		
		@param pre_post_pop		tuple(str, str) containing keys for pre-synaptic
								and post-synaptic populations, e.g.
								(Populations.GPE, Populations.STN)

		@param pre_post_obj		tuple(object, object) containing source and target object
								for synaptic connection. Synapse will be inserted into
								target object

		@param custom_conpar	Custom physiological parameters for the connection,
								in the form of a dict {NTR_0: {params_0}, NTR_1: {params_1}}

		@param custom_synpar	Custom mechanism parameters for the synaptic mechanism,
								in the form of a dict {param_name: param_value, ...}


		@param weight_scales	Scale factors for the weights in range (0,1). 
								This must be an iterable: NetCon.weight[i] is scaled by the i-th value. 
								If a vector	is given, it is scaled and played into the weight.

		@effect					creates an Exp2Syn with an incoming NetCon with
								weight equal to maximum synaptic conductance
		"""

		pre_pop, post_pop = pre_post_pop
		pre_obj, post_obj = pre_post_obj

		if not isinstance(post_obj, nrn.Segment):
			raise ValueError("Post-synaptic object {} is not of type nrn.Segment".format(repr(post_obj)))

		# Get synapse type constructor and make it
		syn_ctor = syn_wrapper_classes.get(syn_type, getattr(h, syn_type))
		syn = syn_ctor(post_obj)

		# Store some data on the synapse
		if hasattr(syn, 'htype'): # check if wrapper class (see nrn/lib/python/hclass.py)
			syn.pre_pop = pre_pop.name
			syn.post_pop = post_pop.name

		# Make NetCon connection
		if isinstance(pre_obj, nrn.Section):
			# Biophysical cells need threshold detection to generate events
			nc = NetConWrapper(pre_obj(0.5)._ref_v, syn, sec=pre_obj)
		else:
			# Source object is POINT_PROCESS or other event-generating objcet
			nc = NetConWrapper(pre_obj, syn)
		nc.pre_pop = pre_pop.name
		nc.post_pop = post_pop.name

		# Get physiological parameter descriptions
		if con_par_data is None:
			con_par_data = self.getPhysioConParams(pre_pop, post_pop, use_sources, custom_conpar)
		
		# Get mapping from physiological to NEURON mechanism parameters
		syn_par_map = getNrnConParamMap(syn_type)

		# keep track of parameters that are assigned
		syn_assigned_pars = []
		netcon_assigned_pars = []

		# For each NT receptor, look up the physiological connection parameters,
		# and translate them to a parameter of the synaptic mechanism or NetCon
		for rec in receptors:
			parname_map = syn_par_map[rec] # how each connection parameter is mapped to mechanism parameter
			phys_params = con_par_data[rec] # physiological parameters from given sources

			for phys_parname, mech_parspec in parname_map.iteritems():

				# Check if parameters is available from given sources
				if not phys_parname in phys_params:
					logger.anal(dedent("""\
							Parameter {dictpar} not found for connection ({pre},{post},{rec}).
							This means that parameter {mechpar} will not be set\n""").format(
								dictpar=phys_parname, pre=pre_pop, post=post_pop, 
								rec=rec, mechpar=mech_parspec))
					continue

				# Interpret parameter specification (what is the parameter?)
				mechtype, mech_parname, paridx = interpretParamSpec(mech_parspec)

				# Get the actual parameter value
				value_spec = phys_params[phys_parname]
				par_val = evalValueSpec(value_spec, self._rng)

				# Determine target (synapse or NetCon)
				if mechtype == 'syn':
					target = syn
					syn_assigned_pars.append(mech_parname)
				
				elif mechtype == 'netcon':
					target = nc
					netcon_assigned_pars.append(mech_parname)
				else:
					raise ValueError("Cannot set attribute of unknown mechanism type {}".format(mechtype))

				# Set attribute
				if paridx is None:
					setattr(target, mech_parname, par_val)
				else:
					getattr(target, mech_parname)[int(paridx)] = par_val

		# Custom synaptic mechanism parameters
		if custom_synpar is not None:
			for pname, pval in custom_synpar.iteritems():
				setattr(syn, pname, pval)

		# Apply possible corrections to synaptic parameters
		correct_syn = syn_mech_correctors.get(syn_type, None)
		if correct_syn is not None:
			correct_syn(syn)

		# Set weights
		weight_vecs = []
		if (weight_scales is None) and (not 'weight' in netcon_assigned_pars):
			# Weight was never assigned, assume synapse has gmax attribute
			nc.weight[0] = 1.0

		elif (weight_scales is not None):
			for i_w, weight in enumerate(weight_scales):

				if isinstance(weight, (int, float)):
					# If weight was assigned as part of synapse parameter, scale it
					if ('weight' in netcon_assigned_pars):
						nc.weight[i_w] = nc.weight[i_w] * weight
					else:
						nc.weight[i_w] = weight

				else: # weight is h.Vector
					# If weight was assigned as part of synapse parameter, scale it
					if ('weight' in netcon_assigned_pars):
						logger.anal("Weight was assigned as part of synapse parameters. New weight Vector will be allocated.")
						wvec = h.Vector()
						wvec.copy(weight)
						wvec.mul(nc.weight[i_w]) # Vector.mul() : multiply in-place
					else:
						wvec = weight

					# Play Vector into weight
					timevec = weight_times[i_w]
					wvec.play(nc._ref_weight[i_w], timevec)
					weight_vecs.append(wvec)

		# Return refs to objects that need to stay alive
		return syn, nc, weight_vecs