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
	ALL = int(2**32 - 1)

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


@unique
class NTReceptors(Enum):
	"""
	NTReceptors used in synaptic connections
	"""
	AMPA = 0
	NMDA = 1
	GABAA = 2
	GABAB = 3


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
	syn.gmax_GABAA = syn.gmax_GABAA / syn.U1
	syn.gmax_GABAB = syn.gmax_GABAB / 0.21


def correct_GLUsyn(syn):
	"""
	Correct parameters of GLUsyn so peak synaptic conductance
	is equal to value of 'gmax_' parameters
	"""

	# Compensate for effect max value Hill factor and U1 on gmax_GABAA and gmax_GABAB
	syn.gmax_AMPA = syn.gmax_AMPA / syn.U1
	syn.gmax_NMDA = syn.gmax_NMDA / syn.U1


class CellConnector(object):
	"""
	Class for storing connection parameters and making connections.
	"""

	def __init__(self, physio_state, rng):
		self._physio_state = physio_state
		self._rng = rng


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
			'rate_mean': 60.0,
			'rate_deviation': 10.0,
			'rate_units': 'Hz',
			'pause_mean': 0.6, # average pause duration 0.5-0.7 s in monkeys
			'pause_units': 's',
			'pause_dist': 'poisson', # Can control rate NetStim with pause NetStim (stim.number resets, see mod file)
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

	def getConParams(self, pre_pop, post_pop, use_sources, custom_params=None):
		"""
		Get parameters for afferent connections onto given population,
		in given physiological state.

		@param custom_params	custom parameters in the form of a dict
								{NTR_0: {params_0}, NTR_1: {params_1}} etc.
		"""

		pop = Populations

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
				cp[Pop.CTX][Rec.AMPA][Src.Chu2015].update({
					'Ipeak': park_gain * cp[Pop.CTX][Rec.NMDA][Src.Chu2015]['Ipeak'],
					'gbar':  park_gain * cp[Pop.CTX][Rec.NMDA][Src.Chu2015]['Ipeak'],	
				})

			# ---------------------------------------------------------------------
			# Default parameters

			# Copy params Chu (2015)
			cp[Pop.CTX][Rec.AMPA][Src.Default] = dict(cp[Pop.CTX][Rec.AMPA][Src.Chu2015])
			cp[Pop.CTX][Rec.NMDA][Src.Default] = dict(cp[Pop.CTX][Rec.NMDA][Src.Chu2015])
			
			# Modification to NMDA conductance
			#	-> NMDA conductance is typically 70% of that of AMPA (see EPFL MOOC)
			cp[Pop.CTX][Rec.NMDA][Src.Default]['gbar'] = 0.7 * cp[Pop.CTX][Rec.AMPA][Src.Default]['gbar']
			
			
			# ---------------------------------------------------------------------
			# Parameters Gradinaru (2009)

			cp[Pop.CTX][Rec.AMPA][Src.Gradinaru2009] = {
				'tau_rec_STD': 200.,
				'tau_rec_STP': 1., # no facilitation
				'P_release_base': 0.7,
			}

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
			}

			cp[Pop.GPE][Rec.GABAB][Src.Default] = {
				'Erev': Erev_GABAB,
				'gbar': 350.*1e-3 / (Ermp - Erev_GABAB), # Chu2015 in healthy state
				'delay': 1.0,
				'Vpre_threshold': 0.0,
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

		return cp_final

	#######################################################################
	# Parameter mapping for synaptic mechanisms
	#######################################################################

	# mapping synapse type -> parameter names
	syn_par_maps = {
		'GABAsyn' : {}, # see below
		'GLUsyn' : {}, # see below
		'Exp2Syn' : {}, # see below
	}

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


	# Parameter correction functions for synaptic mechanisms
	syn_mech_correctors = {
		'GABAsyn' : correct_GABAsyn,
		'GLUsyn' : correct_GLUsyn,
	}


	def getSynParamMap(self, syn_type):
		"""
		For given synaptic mechanism (POINT_PROCESS defined in .mod file),
		get mapping from parameter name in dict getConParams()
		to parameters of the synaptic mechanism and NetCon.

		In other words: how each key in the parameter dictionary
		should be interpreted.
		"""
		return dict(self.syn_par_maps[syn_type]) # return a copy


	def getSynMechParamNames(self, syn_type):
		"""
		Get parameter names for synaptic mechanism
		"""
		ntr_params_names = self.syn_par_maps[syn_type]
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


	def getSynCorrector(self, syn_type):
		"""
		Get parameter correction functions for given synapse type.
		"""
		
		return self.syn_mech_correctors.get(syn_type, None)


	def make_synapse(self, pre_post_pop, pre_post_obj, syn_type, receptors, 
						use_sources=None, custom_conpar=None, custom_synpar=None,
						con_par_data=None,
						weight_scales=None, weight_times=None):
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

		rng = self._rng
		pre_pop, post_pop = pre_post_pop
		pre_obj, post_obj = pre_post_obj

		if not isinstance(post_obj, nrn.Segment):
			raise ValueError("Post-synaptic object {} is not of type nrn.Segment".format(repr(post_obj)))

		# Make synapse
		syn_ctor = getattr(h, syn_type) # constructor for synapse type
		syn = syn_ctor(post_obj)

		# Make NetCon connection
		if isinstance(pre_obj, nrn.Section):
			# Biophysical cells need threshold detection to generate events
			nc = h.NetCon(pre_obj(0.5)._ref_v, receiver, sec=pre_obj)

		else:
			# Source object is POINT_PROCESS or other event-generating objcet
			nc = h.NetCon(pre_obj, syn)

		# Get connnection parameters and how to use them
		if con_par_data is None:
			con_par_data = self.getConParams(pre_pop, post_pop, use_sources, custom_conpar)
		syn_par_map = self.getSynParamMap(syn_type) # how they map to mechanism parameters

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
					logger.debug(dedent("""\
							Parameter {dictpar} not found for connection ({pre},{post},{rec}).
							This means that parameter {mechpar} will not be set\n""").format(
								dictpar=phys_parname, pre=pre_pop, post=post_pop, 
								rec=rec, mechpar=mech_parspec))
					continue

				matches = re.search(r'^(?P<mech>\w+):(?P<parname>\w+)(\[(?P<idx>\d+)\])?', mech_parspec)
				mech_parname = matches.group('parname')
				paridx = matches.group('idx')
				mechtype = matches.group('mech')

				# Get the actual parameter value
				par_data = phys_params[phys_parname]
				par_val = None

				# Make sure it is a numerical value
				if isinstance(par_data, (float, int)):
					# parameter is numerical value
					par_val = par_data

				elif isinstance(par_data, types.FunctionType):
					# parameter is a function
					par_val = par_data()

				elif isinstance(par_data, dict):
					# parameter is described by other parameters (e.g. distribution)

					if ('min' in par_data and 'max' in par_data):
						lower = par_data['min']
						upper = par_data['max']
						par_val = lower + rng.rand()*(upper-lower)

					elif ('mean' in par_data and 'deviation' in par_data):
						lower = par_data['mean'] - par_data['deviation']
						upper = par_data['mean'] + par_data['deviation']
						par_val = lower + rng.rand()*(upper-lower)

					elif ('mean' in par_data and 'stddev' in par_data):
						par_val = rng.normal(par_data['mean'], par_data['stddev'])

					else:
						raise ValueError('Could not infer distribution from parameters in {}'.format(par_data))

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
		correct_syn = self.getSynCorrector(syn_type)
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
						logger.warning("Weight was assigned as part of synapse parameters. New weight Vector will be allocated.")
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