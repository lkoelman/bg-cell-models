"""
Protocols for optimization of reduced neuron model using BluePyOpt.

@author	Lucas Koelman

@date	25/09/2017


NOTES

- see SETPARAM comment for parameters that should be set by user

"""

import collections

import numpy as np

import bluepyopt.ephys as ephys
from bpop_extensions import PhysioProtocol, NrnSpaceClamp, NrnNamedSecLocation
from bpop_protocol_ext import SelfContainedProtocol

from evalmodel import (
	cellpopdata as cpd,
	proto_common, proto_background
)
from evalmodel.proto_common import StimProtocol

Pop = cpd.Populations
NTR = cpd.NTReceptors
Ref = cpd.ParameterSource
StnModel = cpd.StnModel

from gillies_model import set_aCSF

import logging
logger = logging.getLogger('bpop_ext')


################################################################################
# PROTOCOLS
################################################################################

# Location for current clamp
soma_center_loc = ephys.locations.NrnSeclistCompLocation(
				name			= 'soma_center_loc',
				seclist_name	= 'somatic',
				sec_index		= 0,		# index in SectionList
				comp_x			= 0.5)	# x-location in Section


class BpopProtocolWrapper(object):
	"""
	Common attributes:

		ephys_protocol:		PhysioProtocol instance
		
		proto_vars:			dict with protocol variables
		
		response_interval:	expected time interval of response
	"""

	# SETPARAM: spike threshold for cell or specific protocol
	spike_threshold = -10.0
	
	@classmethod
	def make(cls, stim_proto, model_type=None):
		"""
		Instantiate protocol one of the protocol objects defined in this model.
		"""
		wrapper_class = PROTOCOL_WRAPPERS[stim_proto]
		return wrapper_class(model_type)


	def get_mechs_params(self):
		"""
		Get all ephys.mechanisms and ephys.parameters used by the protocols.

		These need to be assigned to the cell model to run the protocol.

		@return		tuple(mechs, params) containing a list of ephys.mechanisms 
					and ephys.parameters respectively
		"""

		proto_mechs = self.proto_vars.get('pp_mechs', []) + \
		              self.proto_vars.get('range_mechs', [])

		proto_params = self.proto_vars.get('pp_mech_params', [])

		return proto_mechs, proto_params


	@classmethod
	def all_mechs_params(cls, proto_wrappers):
		"""
		Concatenate all mechanisms and all parametes for given protocol wrappers.

		This is useful for assigning to a cell model that must be able to run multiple protocols.
		"""
		all_mechs, all_params = [], []
		for proto in proto_wrappers:
			mechs, params = proto.get_mechs_params()
			all_mechs.extend(mechs)
			all_params.extend(params)

		return all_mechs, all_params

################################################################################
# PLATEAU protocol

def init_plateau(sim, model):
	"""
	Initialize simulator to run plateau protocol

	NOTE: function must be declared at top-level of module in order to be pickled
	"""
	h = sim.neuron.h
	h.celsius = 30
	h.v_init = -60
	h.set_aCSF(4) # gillies_model.set_aCSF(4) # if called before model.instantiate()
	h.init()


class BpopPlateauProtocol(BpopProtocolWrapper):
	"""
	Functions for setting up plateau protocol.
	"""

	IMPL_PROTO = StimProtocol.CLAMP_PLATEAU

	def __init__(self, stn_model_type=None):
		"""
		Initialize all protocol variables for given model type

		@param stn_model_type		cellpopdata.StnModel enum instance

		@post						following attributes will be available on this object:
									- ephys_protocol: PhysioProtocol instance
									- proto_vars: dict with protocol variables
									- response_interval: expected time interval of response
		"""
		pass # protol specification independent of model type

	# stimulus parameters
	I_hyper = -0.17			# hyperpolarize to -70 mV (see fig. 10C)
	I_depol = I_hyper + 0.2	# see fig. 10D: 0.2 nA (=stim.amp) over hyperpolarizing current

	del_depol = 1000
	dur_depol = 50			# see fig. 10D, top right
	dur_total = 2000

	# stimulus interval for eFEL features
	plat_start = del_depol - 50
	plat_stop = del_depol + 200
	response_interval = (plat_start, plat_stop)

	stim1_hyp = ephys.stimuli.NrnSquarePulse(
					step_amplitude	= I_hyper,
					step_delay		= 0,
					step_duration	= del_depol,
					location		= soma_center_loc,
					total_duration	= del_depol)

	stim2_dep = ephys.stimuli.NrnSquarePulse(
					step_amplitude	= I_depol,
					step_delay		= del_depol,
					step_duration	= dur_depol,
					location		= soma_center_loc,
					total_duration	= del_depol + dur_depol)

	stim3_hyp = ephys.stimuli.NrnSquarePulse(
					step_amplitude	= I_hyper,
					step_delay		= del_depol + dur_depol,
					step_duration	= del_depol,
					location		= soma_center_loc,
					total_duration	= dur_total)

	plat_rec1 = ephys.recordings.CompRecording(
					name			= '{}.soma.v'.format(IMPL_PROTO.name),
					location		= soma_center_loc,
					variable		= 'v')

	ephys_protocol = PhysioProtocol(
					name		= IMPL_PROTO.name, 
					stimuli		= [stim1_hyp, stim2_dep, stim3_hyp],
					recordings	= [plat_rec1],
					init_func	= init_plateau)

	proto_vars = {
		# 'pp_mechs':			[],
		# 'pp_comp_locs':		[],
		# 'pp_target_locs':	[],
		# 'pp_mech_params':	[],
		'stims':			[stim1_hyp, stim2_dep, stim3_hyp],
		'recordings':		[plat_rec1],
		# 'range_mechs':		[],
	}

	# Characterizing features and parameters for protocol
	characterizing_feats = {
		### SPIKE TIMING RELATED ###
		'Spikecount': {			# (int) The number of peaks during stimulus
			'weight':	2.0,
		},
		# 'mean_frequency',		# (float) the mean frequency of the firing rate
		# 'burst_mean_freq',	# (array) The mean frequency during a burst for each burst
		'adaptation_index': {	# (float) Normalized average difference of two consecutive ISIs
			'weight':	1.0,
			'double':	{'spike_skipf': 0.0},
			'int':		{'max_spike_skip': 0},
		},
		'ISI_CV': {				# (float) coefficient of variation of ISI durations
			'weight':	1.0,
		},
		# 'ISI_log',			# no documentation
		'Victor_Purpura_distance': {
			'weight':	2.0,
			'double': { 'spike_shift_cost' : 1.0 },
		},
		### SPIKE SHAPE RELATED ###
		# 'AP_duration_change': {	# (array) Difference of the durations of the second and the first action potential divided by the duration of the first action potential
		# 	'weight':	1.0,
		# },
		# 'AP_duration_half_width_change': { # (array) Difference of the FWHM of the second and the first action potential divided by the FWHM of the first action potential
		# 	'weight':	1.0,
		# },
		# 'AP_rise_time': {		# (array) Time from action potential onset to the maximum
		# 	'weight':	1.0,
		# },
		# 'AP_rise_rate':	{		# (array) Voltage change rate during the rising phase of the action potential
		# 	'weight':	1.0,
		# },
		# 'AP_height': {			# (array) The voltages at the maxima of the peak
		# 	'weight':	1.0,
		# },
		# 'AP_amplitude': {		# (array) The relative height of the action potential
		# 	'weight':	1.0,
		# },
		# 'spike_half_width':	{	# (array) The FWHM of each peak
		# 	'weight':	1.0,
		# },
		# 'AHP_time_from_peak': {	# (array) Time between AP peaks and AHP depths
		# 	'weight':	1.0,
		# },
		# 'AHP_depth': {			# (array) relative voltage values at the AHP
		# 	'weight':	1.0,
		# },
		'min_AHP_values': {		# (array) Voltage values at the AHP
			'weight':	2.0,	# this feature recognizes that there is an elevated plateau
		},
	}

################################################################################
# REBOUND protocol

def init_rebound(sim, model):
	"""
	Initialize simulator to run rebound protocol

	NOTE: function must be declared at top-level of module in order to be pickled
	"""
	h = sim.neuron.h
	h.celsius = 35
	h.v_init = -60
	h.set_aCSF(4)
	h.init()


class BpopReboundProtocol(BpopProtocolWrapper):
	"""
	Functions for setting up rebound protocol.
	"""

	IMPL_PROTO = StimProtocol.CLAMP_REBOUND

	def __init__(self, stn_model_type=None):
		"""
		Initialize all protocol variables for given model type

		@param stn_model_type		cellpopdata.StnModel enum instance

		@post						following attributes will be available on this object:
									- ephys_protocol: PhysioProtocol instance
									- proto_vars: dict with protocol variables
									- response_interval: expected time interval of response
		"""
		pass # protol specification independent of model type

	dur_hyper = 500
	dur_burst = 150 # SETPARAM: approximate duration of burst
	reb_start, reb_stop = dur_hyper-50, dur_hyper+dur_burst # SETPARAM: stimulus interval for eFEL features
	response_interval = (reb_start, reb_stop)

	reb_clmp1 = NrnSpaceClamp(
					step_amplitudes	= [0, 0, -75],
					step_durations	= [0, 0, dur_hyper],
					total_duration	= reb_stop,
					location		= soma_center_loc)

	reb_rec1 = ephys.recordings.CompRecording(
					name			= '{}.soma.v'.format(IMPL_PROTO.name),
					location		= soma_center_loc,
					variable		= 'v')

	ephys_protocol = PhysioProtocol(
					name		= IMPL_PROTO.name, 
					stimuli		= [reb_clmp1],
					recordings	= [reb_rec1],
					init_func	= init_rebound)

	proto_vars = {
		# 'pp_mechs':			[],
		# 'pp_comp_locs':		[],
		# 'pp_target_locs':	[],
		# 'pp_mech_params':	[],
		'stims':			[reb_clmp1],
		'recordings':		[reb_rec1],
		# 'range_mechs':		[],
	}

	# Characterizing features and parameters for protocol
	characterizing_feats = {
		### SPIKE TIMING RELATED ###
		'Spikecount': {			# (int) The number of peaks during stimulus
			'weight':	2.0,
		},
		# 'mean_frequency',		# (float) the mean frequency of the firing rate
		# 'burst_mean_freq',	# (array) The mean frequency during a burst for each burst
		'adaptation_index': {	# (float) Normalized average difference of two consecutive ISIs
			'weight':	1.0,
			'double':	{'spike_skipf': 0.0},
			'int':		{'max_spike_skip': 0},
		},
		'ISI_CV': {				# (float) coefficient of variation of ISI durations
			'weight':	1.0,
		},
		# 'ISI_log',			# no documentation
		'Victor_Purpura_distance': {
			'weight':	2.0,
			'double': { 'spike_shift_cost' : 1.0 },
		},
		### SPIKE SHAPE RELATED ###
		# 'AP_duration_change': {	# (array) Difference of the durations of the second and the first action potential divided by the duration of the first action potential
		# 	'weight':	1.0,
		# },
		# 'AP_duration_half_width_change': { # (array) Difference of the FWHM of the second and the first action potential divided by the FWHM of the first action potential
		# 	'weight':	1.0,
		# },
		# 'AP_rise_time': {		# (array) Time from action potential onset to the maximum
		# 	'weight':	1.0,
		# },
		# 'AP_rise_rate':	{		# (array) Voltage change rate during the rising phase of the action potential
		# 	'weight':	1.0,
		# },
		# 'AP_height': {			# (array) The voltages at the maxima of the peak
		# 	'weight':	1.0,
		# },
		# 'AP_amplitude': {		# (array) The relative height of the action potential
		# 	'weight':	1.0,
		# },
		# 'spike_half_width':	{	# (array) The FWHM of each peak
		# 	'weight':	1.0,
		# },
		# 'AHP_time_from_peak': {	# (array) Time between AP peaks and AHP depths
		# 	'weight':	1.0,
		# },
		# 'AHP_depth': {			# (array) relative voltage values at the AHP
		# 	'weight':	1.0,
		# },
		'min_AHP_values': {		# (array) Voltage values at the AHP
			'weight':	2.0,	# this feature recognizes that there is an elevated plateau
		},
	}

################################################################################
# SYNAPTIC protocol

# - References:
# 	- see https://github.com/BlueBrain/BluePyOpt/blob/master/examples/expsyn/

def init_burstsyn(sim, icell):
	"""
	Initialize simulator to run synaptic burst protocol

	NOTE: function must be declared at top-level of module in order to be pickled
	"""
	h = sim.neuron.h

	h.celsius = 35
	h.v_init = -60
	set_aCSF(4) # gillies_model.set_aCSF(4) # if called before model.instantiate()
	
	logger.debug("Set aCSF concentrations.")

	# lower sKCa conductance to promote bursting
	for sec in icell.all:
		for seg in sec:
			seg.gk_sKCa = 0.6 * seg.gk_sKCa

	# BluePyOpt never calls h.init()/stdinit()/finitialize()
	h.init()


class BpopSynBurstProtocol(BpopProtocolWrapper):
	"""
	Functions for setting up plateau protocol.
	"""

	IMPL_PROTO = StimProtocol.MIN_SYN_BURST

	rng = np.random.RandomState(25031989)
	cc = cpd.CellConnector(cpd.PhysioState.NORMAL, rng)

	# TODO: adjust start of spike volleys and times here, also adjust CSF ioninc concentrations
	synburst_start = 700
	synburst_stop = synburst_start + 350
	response_interval = (synburst_start, synburst_stop) # interval to analyze eFEL features

	def __init__(self, stn_model_type=None):
		"""
		Initialize all protocol variables for given model type

		@param stn_model_type		cellpopdata.StnModel enum instance

		@post						following attributes will be available on this object:
									- ephys_protocol: PhysioProtocol instance
									- proto_vars: dict with protocol variables
									- response_interval: expected time interval of response
		"""

		# SETPARAM: synapse locations (name of icell SectionList & index)
		if stn_model_type == StnModel.Gillies2005:
			# TODO: fix locations and weights for full model
			self.synapse_info = {
				'GLUsyn':	[('SThcell[0].dend0[11]', 0.25, 1.0), 
							('SThcell[0].dend0[16]', 0.625, 1.0), 
							('SThcell[0].dend0[19]', 0.875, 1.0), 
							('SThcell[0].dend0[11]', 0.416667, 1.0)],
				
				'GABAsyn':	[('SThcell[0].dend0[2]', 0.75, 1.0)],
							# ('SThcell[0].dend0[2]', 0.75)],
			}
		elif stn_model_type == StnModel.Gillies_FoldMarasco:
			self.synapse_info = {
				'GLUsyn':	[('zipF_zipE_zipD_SThcell0dend01', 0.9375, 0.996135509079), 
							('zipG_zipF_zipE_zipD_SThcell0dend02', 1, 0.93307003268), 
							('zipG_zipF_zipE_zipD_SThcell0dend02', 1, 0.840762949423), 
							('zipG_zipF_zipE_zipD_SThcell0dend01', 1, 0.956082747602)],
				
				'GABAsyn':	[('SThcell[0].dend0[2]', 0.75, 0.985420572679)],
							# ('SThcell[0].dend0[2]', 0.75)],
			}
		else:
			raise NotImplementedError("Protocol not implemented for model {}".format(stn_model_type))

		# Make elements of the protocol
		self.proto_vars = self.make_synburst_proto_vars()

		self.ephys_protocol = PhysioProtocol(
							name		= self.IMPL_PROTO.name, 
							stimuli		= self.proto_vars['stims'],
							recordings	= self.proto_vars['recordings'],
							init_func	= init_burstsyn)


	def get_syn_mechs_params(self, mech_name, stim_pp, nrn_params, proto_vars):
		"""
		Get ephys.mechanisms, ephys.locations, ephys.parameters objects for 
		synapses of given mechanism.

		@param	mech_name		synaptic mechanism name

		@param	stim_bp			BluePyOpt.ephys.stimuli.NrnNetStimStimulus object

		@param	nrn_params		dictionary with NEURON object parameters for this connection type

		@param	proto_vars		dictionary to store BluePyOpt objects created
		"""

		# Get compartmental locations for GLU synapses
		# TODO: make locations here and copy weights from evalmodel implementation
		loc_data = self.synapse_info[mech_name]
		comp_locs = []
		for i_loc, (sec_name, sec_x, nc_weight) in enumerate(loc_data):

			loc = NrnNamedSecLocation(
					name			= '{}_loc_{}'.format(mech_name, i_loc),
					sec_name		= sec_name,
					comp_x			= sec_x)

			comp_locs.append(loc)

		proto_vars['pp_comp_locs'].extend(comp_locs)

		# Make synapse at locations
		syn_pp = ephys.mechanisms.NrnMODPointProcessMechanism(
						name		= '{}_synapses'.format(mech_name),
						suffix		= mech_name,
						locations	= comp_locs)

		proto_vars['pp_mechs'].append(syn_pp)

		# Make 'target location' for synapse parameters & NetStims
		loc_pp = ephys.locations.NrnPointProcessLocation(
						name			= '{}_locs'.format(mech_name),
						pprocess_mech	= syn_pp, # NOTE: this encapsulates multiple distributed instances of the point process mechaism
						comment			= "location of {}".format(syn_pp.name))

		proto_vars['pp_comp_locs'].append(loc_pp)
			

		# Make BluePyOpt parameter objects
		for param_name, param_value in nrn_params['pointprocess'].iteritems():

			syn_param = ephys.parameters.NrnPointProcessParameter(                   
							name			= "{}_{}".format(mech_name, param_name),
							param_name		= param_name,
							value			= param_value,
							bounds			= [param_value, param_value],
							frozen			= True,
							locations		= [loc_pp]) # one loc can encapsulate multiple locations

			proto_vars['pp_mech_params'].append(syn_param)
		
		# Save stimulus with updated locations
		stim_pp.weight = sum([weight for _,_,weight in loc_data]) / len(loc_data) # TODO: now this is average weight because of limitations of ephys.NetStim -> change this
		stim_pp.locations.append(loc_pp)
		proto_vars['stims'].append(stim_pp)


	def make_synburst_proto_vars(self):
		"""
		Make minimal set of synapses, with locations and parameters that generate
		a burst in STN.
		"""

		# All variables for protocol (keep alive)
		proto_vars = {
			'pp_mechs':		[],
			'pp_comp_locs':		[],
			'pp_target_locs':	[],
			'pp_mech_params':	[],
			'stims':			[],
			'recordings':		[],
			#'range_mechs':		[], # not used in this case
		}

		############################################################################
		# Recordings

		# Recording for EFeatures
		proto_rec = ephys.recordings.CompRecording(
						name			= '{}.soma.v'.format(self.IMPL_PROTO.name),
						location		= soma_center_loc,
						variable		= 'v')

		proto_vars['recordings'].append(proto_rec)
		
		############################################################################
		# Get compartmental locations for GLU synapses
		mech_name = 'GLUsyn'

		# Get physiological parameters
		physio_params = self.cc.getPhysioConParams(Pop.CTX, Pop.STN, [Ref.Default, Ref.Chu2015, Ref.Gradinaru2009])
		
		# Get NEURON parameters
		nrn_params = self.cc.getNrnObjParams(mech_name, physio_params)

		# SETPARAM: set parameters from burst experiment
		nrn_params['pointprocess'].update({ # copied from printed values in notebook
									'gmax_AMPA': 0.0049107, #0.0034375,
									'gmax_NMDA': 0.0034375, #0.00240625,
									})

		# Spike generator for GLU synapses
		stim_glu = ephys.stimuli.NrnNetStimStimulus(
						locations		= [],		# assigned later
						total_duration	= self.synburst_stop,# total duration of stim protocol
						interval		= 100.0**-1 * 1e3, # 100 Hz to ms
						number			= 5,
						start			= self.synburst_start + 150,
						noise			= 0,
						weight			= None)		# assigned later

		# Make GLU synapses (put everything in proto_vars)
		self.get_syn_mechs_params(mech_name, stim_glu, nrn_params, proto_vars)

		############################################################################
		# Get compartmental locations for GABA synapses
		mech_name = 'GABAsyn'

		# Get physiological parameters
		physio_params = self.cc.getPhysioConParams(
								Pop.GPE, Pop.STN, 
								[Ref.Custom, Ref.Chu2015, Ref.Fan2012, Ref.Atherton2013])
		
		# Get NEURON parameters
		nrn_params = self.cc.getNrnObjParams(mech_name, physio_params)
		
		# SETPARAM: set parameters from burst experiment
		nrn_params['pointprocess'].update({ # copied from printed values in notebook
									'gmax_GABAA': 0.0466666,
									'gmax_GABAB': 0.0724637,
									})

		# Spike generator for GABA synapses
		stim_gaba = ephys.stimuli.NrnNetStimStimulus(
						locations		= [],		# assigned later
						total_duration	= self.synburst_stop,# total duration of stim protocol
						interval		= 100.0**-1 * 1e3, # 100 Hz to ms
						number			= 5,
						start			= self.synburst_start,
						noise			= 0,
						weight			= None)		# assigned later

		# Make GLU synapses (put everything in proto_vars)
		self.get_syn_mechs_params(mech_name, stim_gaba, nrn_params, proto_vars)

		return proto_vars


	# Characterizing features and parameters for protocol
	characterizing_feats = {
		### SPIKE TIMING RELATED ###
		'Spikecount': {			# (int) The number of peaks during stimulus
			'weight':	2.0,
		},
		'adaptation_index': {	# (float) Normalized average difference of two consecutive ISIs
			'weight':	1.0,
			'double':	{'spike_skipf': 0.0},
			'int':		{'max_spike_skip': 0},
		},
		'ISI_CV': {				# (float) coefficient of variation of ISI durations
			'weight':	1.0,
		},
		'Victor_Purpura_distance': {
			'weight':	2.0,
			'double': { 'spike_shift_cost' : 1.0 },
		},
		### SPIKE SHAPE RELATED ###
		# 'AP_duration_half_width_change': { # (array) Difference of the FWHM of the second and the first action potential divided by the FWHM of the first action potential
		# 	'weight':	1.0,
		# },
		# 'AP_rise_rate':	{		# (array) Voltage change rate during the rising phase of the action potential
		# 	'weight':	1.0,
		# },
		# 'AP_height': {			# (array) The voltages at the maxima of the peak
		# 	'weight':	1.0,
		# },
		# 'AP_amplitude': {		# (array) The relative height of the action potential
		# 	'weight':	1.0,
		# },
		# 'spike_half_width':	{	# (array) The FWHM of each peak
		# 	'weight':	1.0,
		# },
		# 'AHP_time_from_peak': {	# (array) Time between AP peaks and AHP depths
		# 	'weight':	1.0,
		# },
		# 'AHP_depth': {			# (array) relative voltage values at the AHP
		# 	'weight':	1.0,
		# },
		'min_AHP_values': {		# (array) Voltage values at the AHP
			'weight':	2.0,	# this feature recognizes that there is an elevated plateau
		},
	}

################################################################################
# REBOUND protocol

def rng_getter(setup_kwargs):
	"""
	Function to get Numpy.Random object for stimulation protocol setup functions.
	"""
	import numpy as np
	base_seed = setup_kwargs['base_seed']
	return np.random.RandomState(base_seed)


def connector_getter(setup_kwargs):
	"""
	Function to get CellConnector for stimulation protocol setup functions.
	"""
	from evalmodel import cellpopdata as cpd
	physio_state = setup_kwargs['physio_state']
	rng = setup_kwargs['rng']
	return cpd.CellConnector(physio_state, rng)


class BpopBackgroundProtocol(BpopProtocolWrapper):
	"""
	Functions for setting up rebound protocol.
	"""

	IMPL_PROTO = StimProtocol.SYN_BACKGROUND_HIGH

	def __init__(self, stn_model_type=None):
		"""
		Initialize all protocol variables for given model type

		@param stn_model_type		cellpopdata.StnModel enum instance

		@post						following attributes will be available on this object:
									- ephys_protocol: PhysioProtocol instance
									- proto_vars: dict with protocol variables
									- response_interval: expected time interval of response
		"""
		pass # protol specification independent of model type


	# TODO: find out how to make it run long enough if no ephys Stimuli present (set in ExtProtocol?)
	sim_end = 2000.0
	response_interval = (300.0, sim_end)

	bg_recV = ephys.recordings.CompRecording(
					name			= '{}.soma.v'.format(IMPL_PROTO.name),
					location		= soma_center_loc,
					variable		= 'v')

	# Protocol setup
	proto_setup_funcs = [
		proto_background.make_inputs_ctx_impl, 
		proto_background.make_inputs_gpe_impl
	]

	proto_init_funcs = [
		proto_background.init_sim_impl
	]

	proto_setup_kwargs_const = {
		'base_seed': 25031989,
		'gid': 1,
		'do_map_synapses': True,
		'physio_state': cpd.PhysioState.NORMAL.name,
	}

	proto_setup_kwargs_getters = collections.OrderedDict([
		('rng', rng_getter),
		('connector', connector_getter),
	])

	# Recording and plotting traces
	proto_rec_funcs = [
		proto_background.rec_spikes,
	]

	proto_plot_funcs = [
		proto_common.plot_all_spikes,
	]

	ephys_protocol = SelfContainedProtocol(
						name				= IMPL_PROTO.name, 
						recordings			= [bg_recV],
						total_duration		= sim_end,
						proto_init_funcs			= proto_init_funcs,
						proto_setup_funcs_pre		= proto_setup_funcs,
						proto_setup_kwargs_const	= proto_setup_kwargs_const,
						proto_setup_kwargs_getters	= proto_setup_kwargs_getters,
						rec_traces_funcs	= proto_rec_funcs,
						plot_traces_funcs	= proto_plot_funcs)

	proto_vars = {
		'pp_mechs':			[],
		'pp_comp_locs':		[],
		'pp_target_locs':	[],
		'pp_mech_params':	[],
		'stims':			[],
		'recordings':		[bg_recV],
		'range_mechs':		[],
	}

	# Characterizing features and parameters for protocol
	characterizing_feats = {
		'Victor_Purpura_distance': {
			'weight':	2.0,
			'double': { 'spike_shift_cost' : 1.0 },
		},
	}

# ==============================================================================
# EXPORTED variables

PROTOCOL_WRAPPERS = {
	StimProtocol.CLAMP_PLATEAU: BpopPlateauProtocol,
	StimProtocol.CLAMP_REBOUND: BpopReboundProtocol,
	StimProtocol.MIN_SYN_BURST: BpopSynBurstProtocol,
	StimProtocol.SYN_BACKGROUND_HIGH: BpopBackgroundProtocol,
}
