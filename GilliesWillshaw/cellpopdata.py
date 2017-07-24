"""
Parameters for STN connections

@author		Lucas Koelman
@date		12/07/2017


Based on structure of:

	bgmodel/models/kang/nuclei/cellpopdata.py
	bgmodel/models/kumaravelu/nuclei/cellpopdata.py
	bgmodel/models/netkang/kang_netparams.py

"""

from enum import Enum
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

class PhysioState(Enum):
	"""
	Physiological state of the cell
	"""
	NORMAL = 0
	PARKINSONIAN = 1
	PARK_DBS = 2


class Populations(Enum):
	"""
	Physiological state of the cell
	"""
	STN = 0
	CTX = 1
	GPE = 2


class NTReceptors(Enum):
	"""
	NTReceptors used in synaptic connections
	"""
	AMPA = 0
	NMDA = 1
	GABAA = 2
	GABAB = 3


class ParameterSource(Enum):
	Default = 0
	CommonUse = 1		# Widely used or commonly accepted values
	Chu2015 = 2
	Baufreton2009 = 3
	Fan2012 = 4
	Atherton2013 = 5
	Kumaravelu2016 = 6
	Custom = 7			# Custom parameters, e.g. for testing

# Shorthands
Pop = Populations
Rec = NTReceptors
Src = ParameterSource

class CellConnector(object):
	"""
	Class for storing connection parameters and making connections.
	"""

	def __init__(self, physio_state, rng):
		self._physio_state = physio_state
		self._rng = rng

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

			# TODO SETPARAM: add params STD/STP dynamics for GPE-STN and CTX-STN synapses

			# gmax calculation:
			# gmax is in [uS] in POINT_PROCESS synapses
			# 	1 [uS] * 1 [mV] = 1 [nA]
			# 	we want ~ -300 [pA] = -0.3 [nA]
			# 	gmax [uS] * (-68 [mV] - 0 [mV]) = gmax * -68 [nA]
			# 	gmax * -68 = -0.3 <=> gmax = 0.3/68 = 0.044 [uS]

			#######################################################################
			# CTX -> STN parameters
			Erev = 0.
			Ermp = -70.

			# Default parameters
			for ntr in cp[Pop.CTX].keys():
				cp[Pop.CTX][ntr][Src.Default] = {
					'delay': 1.0,
					'Vpre_threshold': 0.0,
				}

			# ---------------------------------------------------------------------
			# Parameters from Chu (2015)
			cp[Pop.CTX][Rec.AMPA][Src.Chu2015] = {
				'Ipeak': -275., 	# peak synaptic current (pA)
				'gbar': -275.*1e-3 / (Ermp - Erev), # gbar calculation
				'tau_rise_g': 1.0,
				'tau_decay_g': 4.0,
				'Erev': Erev,
			}
			cp[Pop.CTX][Rec.NMDA][Src.Chu2015] = {
				'Ipeak': -270., 	# peak synaptic current (pA)
				'gbar': -270.*1e-3 / (Ermp - Erev), # gbar calculation
				'tau_rise_g': 1.0,
				'tau_decay_g': 50.0,
				'Erev': Erev,
			}
			if physio_state == PhysioState.PARKINSONIAN:
				cp[Pop.CTX][Rec.AMPA][Src.Chu2015].update({
					'Ipeak': -390.,
					'gbar': -390.*1e-3 / (Ermp - Erev),	
				})
			

			#######################################################################
			# GPe -> STN parameters

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

	# mapping synapse type -> parameter names
	syn_par_maps = {
		'GABAsyn' : {}, # see below
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
			'P_release_base': 'syn:U1',
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

	# Exp2Syn.moc can be used for any receptor
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


	def make_synapse(self, pre_post_pop, pre_post_obj, syn_type, receptors, 
						use_sources, custom_conpar=None, custom_synpar=None,
						weight_scales=None, weight_times=None):
		"""
		Insert synapse POINT_PROCESS in given section.

		@param pre_post_pop		tuple(str, str) containing keys for pre-synaptic
								and post-synaptic populations, e.g.
								(Populations.GPE, Populations.STN)

		@param pre_post_obj		tuple(object, object) containing source and target object
								for synaptic connection. Synapse will be inserted into
								target object

		@param weight_scales	Scale factors for the weights in range (0,1). 
								This must be an iterable: NetCon.weight[i] is scaled by the i-th value. 
								If a vector	is given, it is scaled and played into the weight.

		@param custom_conpar	Custom physiological parameters for the connection,
								in the form of a dict {NTR_0: {params_0}, NTR_1: {params_1}}

		@param custom_synpar	Custom mechanism parameters for the synaptic mechanism,
								in the form of a dict {param_name: param_value, ...}

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
			nc = hoc.NetCon(pre_obj(0.5)._ref_v, receiver, sec=pre_obj)

		else:
			# Source object is POINT_PROCESS or other event-generating objcet
			nc = h.NetCon(pre_obj, syn)


		# Set each parameter on the synapse object
		phys_data = self.getConParams(pre_pop, post_pop, use_sources, custom_conpar)
		syn_par_map = self.getSynParamMap(syn_type)

		# keep track of parameters that are assigned
		syn_assigned_pars = []
		netcon_assigned_pars = []

		# For each NT receptor, look up the physiological connection parameters,
		# and translate them to a parameter of the synaptic mechanism or NetCon
		for rec in receptors:
			parname_map = syn_par_map[rec] # how each connection parameter is mapped to mechanism parameter
			phys_params = phys_data[rec] # physiological parameters from given sources

			for phys_parname, mech_parspec in parname_map.iteritems():

				# Check if parameters is available from given sources
				if not phys_parname in phys_params:
					logger.warning(dedent("""\
							Parameter {dictpar} not found for connection ({pre},{post},{rec}).
							This means that parameter {mechpar} will not be set""").format(
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

				# Set the attribute
				if mechtype == 'syn':
					target = syn
					syn_assigned_pars.append(mech_parname)
				
				elif mechtype == 'netcon':
					target = nc
					netcon_assigned_pars.append(mech_parname)
				else:
					raise ValueError("Cannot set attribute of unknown mechanism type {}".format(mechtype))

				if paridx is None:
					setattr(target, mech_parname, par_val)
				else:
					getattr(target, mech_parname)[int(paridx)] = par_val

		# Custom synaptic mechanism parameters
		for pname, pval in custom_synpar.iteritems():
			setattr(syn, pname, pval)

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