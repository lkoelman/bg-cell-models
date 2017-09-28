"""
Protocols for optimization of reduced neuron model using BluePyOpt.

@author	Lucas Koelman

@date	25/09/2017


NOTES

- see SETPARAM comment for parameters that should be set by user

"""

import numpy as np

import bluepyopt.ephys as ephys
from bpop_extensions import PhysioProtocol, NrnSpaceClamp

from evalmodel import cellpopdata as cpd
Pop = cpd.Populations
NTR = cpd.NTReceptors
Ref = cpd.ParameterSource


################################################################################
# PROTOCOLS
################################################################################

# Location for current clamp
soma_center_loc = ephys.locations.NrnSeclistCompLocation(
				name			='soma_center_loc',
				seclist_name	='somatic',
				sec_index		=0,		# index in SectionList
				comp_x			=0.5)	# x-location in Section

################################################################################
# PLATEAU protocol

# stimulus parameters
I_hyper = -0.17			# hyperpolarize to -70 mV (see fig. 10C)
I_depol = I_hyper + 0.2	# see fig. 10D: 0.2 nA (=stim.amp) over hyperpolarizing current

del_depol = 1000
dur_depol = 50			# see fig. 10D, top right
dur_total = 2000

# stimulus interval for eFEL features
plat_start = del_depol - 50
plat_stop = del_depol + 200

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
				name			= 'plateau.soma.v',
				location		= soma_center_loc,
				variable		= 'v')

def init_plateau(sim, model):
	""" Initialize simulator to run plateau protocol """
	h = sim.neuron.h
	h.celsius = 30
	h.v_init = -60
	h.set_aCSF(4) # gillies_model.set_aCSF(4) # if called before model.instantiate()

plateau_protocol = PhysioProtocol(
					name		= 'plateau', 
					stimuli		= [stim1_hyp, stim2_dep, stim3_hyp],
					recordings	= [plat_rec1],
					init_func	= init_plateau)

plateau_proto_vars = {
	# 'pp_mechs':			[],
	# 'pp_comp_locs':		[],
	# 'pp_target_locs':	[],
	# 'pp_mech_params':	[],
	'stims':			[stim1_hyp, stim2_dep, stim3_hyp],
	'recordings':		[plat_rec1],
	# 'range_mechs':		[],
}

# Characterizing features and parameters for protocol
plateau_characterizing_feats = {
	'Spikecount': {			# (int) The number of peaks during stimulus
		'weight':	1.0,
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
	'AP_duration_change': {	# (array) Difference of the durations of the second and the first action potential divided by the duration of the first action potential
		'weight':	1.0,
	},
	'AP_duration_half_width_change': { # (array) Difference of the FWHM of the second and the first action potential divided by the FWHM of the first action potential
		'weight':	1.0,
	},
	'AP_rise_time': {		# (array) Time from action potential onset to the maximum
		'weight':	1.0,
	},
	'AP_rise_rate':	{		# (array) Voltage change rate during the rising phase of the action potential
		'weight':	1.0,
	},
	'AP_height': {			# (array) The voltages at the maxima of the peak
		'weight':	1.0,
	},
	'AP_amplitude': {		# (array) The relative height of the action potential
		'weight':	1.0,
	},
	'spike_half_width':	{	# (array) The FWHM of each peak
		'weight':	1.0,
	},
	'AHP_time_from_peak': {	# (array) Time between AP peaks and AHP depths
		'weight':	1.0,
	},
	'AHP_depth': {			# (array) relative voltage values at the AHP
		'weight':	1.0,
	},
	'min_AHP_values': {		# (array) Voltage values at the AHP
		'weight':	1.0,
	},
}

################################################################################
# REBOUND protocol

dur_hyper = 500

reb_clmp1 = NrnSpaceClamp(
				step_amplitudes	= [0, 0, -75],
				step_durations	= [0, 0, dur_hyper],
				total_duration	= 2000,
				location		= soma_center_loc)


reb_rec1 = ephys.recordings.CompRecording(
				name			= 'rebound.soma.v',
				location		= soma_center_loc,
				variable		= 'v')

def init_rebound(sim, model):
	""" Initialize simulator to run rebound protocol """
	h = sim.neuron.h
	h.celsius = 35
	h.v_init = -60
	h.set_aCSF(4) # gillies_model.set_aCSF(4) # if called before model.instantiate()


# stimulus interval for eFEL features
reb_start = dur_hyper - 50
reb_stop = dur_hyper + 1000

rebound_protocol = PhysioProtocol(
					name		= 'rebound', 
					stimuli		= [reb_clmp1],
					recordings	= [reb_rec1],
					init_func	= init_rebound)

rebound_proto_vars = {
	# 'pp_mechs':			[],
	# 'pp_comp_locs':		[],
	# 'pp_target_locs':	[],
	# 'pp_mech_params':	[],
	'stims':			[reb_clmp1],
	'recordings':		[reb_rec1],
	# 'range_mechs':		[],
}

# Characterizing features and parameters for protocol
rebound_characterizing_feats = {
	'Spikecount': {			# (int) The number of peaks during stimulus
		'weight':	1.0,
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
	'AP_duration_change': {	# (array) Difference of the durations of the second and the first action potential divided by the duration of the first action potential
		'weight':	1.0,
	},
	'AP_duration_half_width_change': { # (array) Difference of the FWHM of the second and the first action potential divided by the FWHM of the first action potential
		'weight':	1.0,
	},
	'AP_rise_time': {		# (array) Time from action potential onset to the maximum
		'weight':	1.0,
	},
	'AP_rise_rate':	{		# (array) Voltage change rate during the rising phase of the action potential
		'weight':	1.0,
	},
	'AP_height': {			# (array) The voltages at the maxima of the peak
		'weight':	1.0,
	},
	'AP_amplitude': {		# (array) The relative height of the action potential
		'weight':	1.0,
	},
	'spike_half_width':	{	# (array) The FWHM of each peak
		'weight':	1.0,
	},
	'AHP_time_from_peak': {	# (array) Time between AP peaks and AHP depths
		'weight':	1.0,
	},
	'AHP_depth': {			# (array) relative voltage values at the AHP
		'weight':	1.0,
	},
	'min_AHP_values': {		# (array) Voltage values at the AHP
		'weight':	1.0,
	},
}

################################################################################
# SYNAPTIC protocol

# - References:
# 	- see https://github.com/BlueBrain/BluePyOpt/blob/master/examples/expsyn/

# TODO: custom location object dependent on passive transfer impedance?

rng = np.random.RandomState(25031989)
cc = cpd.CellConnector(cpd.PhysioState.NORMAL, rng)

# SETPARAM: synapse locations (name of icell SectionList & index)
sec_index_x = {
	'GLUsyn':	[('dendritic', 5, 0.9375), 
				('dendritic', 10, 1), 
				('dendritic', 10, 1), 
				('dendritic', 4, 1)],
	
	'GABAsyn':	[('dendritic', 7, 0.75)], # SETPARAM: GABA synapse locations
}

burst_syn_locs = {ppmech: [] for ppmech in sec_index_x.keys()}

# Make NrnSeclistCompLocation for each synapse
for ppmech, loc_data in sec_index_x.iteritems():
	for i_loc, (sec_list, sec_index, sec_x) in enumerate(loc_data):

		loc = ephys.locations.NrnSeclistCompLocation(
						name			= '{}_loc_{}'.format(ppmech, i_loc),
						seclist_name	= sec_list,
						sec_index		= sec_index,
						comp_x			= sec_x)

		burst_syn_locs[ppmech].append(loc)


def get_syn_mechs_params(mech_name, stim_pp, nrn_params, proto_vars):
	"""
	Get ephys.mechanisms, ephys.locations, ephys.parameters objects for 
	synapses of given mechanism.

	@param	mech_name		synaptic mechanism name

	@param	stim_bp			BluePyOpt.ephys.stimuli.NrnNetStimStimulus object

	@param	nrn_params		dictionary with NEURON object parameters for this connection type

	@param	proto_vars		dictionary to store BluePyOpt objects created
	"""

	# Get compartmental locations for GLU synapses
	comp_locs = burst_syn_locs[mech_name]
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
	stim_pp.weight = nrn_params['netcon']['weight']
	stim_pp.locations.append(loc_pp)
	proto_vars['stims'].append(stim_pp)


def make_synburst_proto_vars():
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
		'range_mechs':		[], # not used in this case
	}

	# Recording for EFeatures
	proto_rec = ephys.recordings.CompRecording(
					name			= 'synburst.soma.v',
					location		= soma_center_loc,
					variable		= 'v')

	proto_vars['recordings'].append(proto_rec)
	
	############################################################################
	# Get compartmental locations for GLU synapses
	mech_name = 'GLUsyn'

	# Get physiological parameters
	physio_params = cc.getPhysioConParams(Pop.CTX, Pop.STN, [Ref.Default, Ref.Chu2015, Ref.Gradinaru2009])
	
	# Get NEURON parameters
	nrn_params = cc.getNrnObjParams(mech_name, physio_params)

	# SETPARAM: set parameters from burst experiment
	nrn_params['pointprocess'].update({ # copied from printed values in notebook
								'gmax_AMPA': 0.0049107, #0.0034375,
								'gmax_NMDA': 0.0034375, #0.00240625,
								})
	nrn_params['netcon']['weight'] = 1.0 # scaled by mapsyn.py : set value after scaling!

	# Spike generator for GLU synapses
	stim_glu = ephys.stimuli.NrnNetStimStimulus(
					locations		= [],		# assigned later
					total_duration	= 1250,		# total duration of stim protocol
					interval		= 100.0**-1 * 1e3, # 100 Hz to ms
					number			= 5,
					start			= 850,
					noise			= 0,
					weight			= None)		# assigned later

	# Make GLU synapses (put everything in proto_vars)
	get_syn_mechs_params(mech_name, stim_glu, nrn_params, proto_vars)

	############################################################################
	# Get compartmental locations for GABA synapses
	mech_name = 'GABAsyn'

	# Get physiological parameters
	physio_params = cc.getPhysioConParams(
							Pop.GPE, Pop.STN, 
							[Ref.Custom, Ref.Chu2015, Ref.Fan2012, Ref.Atherton2013])
	
	# Get NEURON parameters
	nrn_params = cc.getNrnObjParams(mech_name, physio_params)
	
	# SETPARAM: set parameters from burst experiment
	nrn_params['pointprocess'].update({ # copied from printed values in notebook
								'gmax_GABAA': 0.0466666,
								'gmax_GABAB': 0.0724637,
								})
	nrn_params['netcon']['weight'] = 0.95 # scaled by mapsyn.py : set value after scaling!

	# Spike generator for GABA synapses
	stim_gaba = ephys.stimuli.NrnNetStimStimulus(
					locations		= [],		# assigned later
					total_duration	= 1250,		# total duration of stim protocol
					interval		= 100.0**-1 * 1e3, # 100 Hz to ms
					number			= 8,
					start			= 700,
					noise			= 0,
					weight			= None)		# assigned later

	# Make GLU synapses (put everything in proto_vars)
	get_syn_mechs_params(mech_name, stim_gaba, nrn_params, proto_vars)

	return proto_vars


# Cell initialization function
def init_burstsyn(sim, icell):
	"""
	Initialize simulator to run synaptic burst protocol
	"""
	h = sim.neuron.h
	h.celsius = 35
	h.v_init = -60
	h.set_aCSF(4) # gillies_model.set_aCSF(4) # if called before model.instantiate()

	# lower sKCa conductance to promote bursting
	for sec in icell.all:
		for seg in sec:
			seg.gk_sKCa = 0.6 * seg.gk_sKCa


# Make elements of the protocol
synburst_proto_vars = make_synburst_proto_vars()

# TODO: adjust start of spike volleys and times here, also adjust CSF ioninc concentrations
synburst_start, synburst_stop = 750, 1250

synburst_protocol = PhysioProtocol(
						name		= 'synburst', 
						stimuli		= synburst_proto_vars['stims'],
						recordings	= synburst_proto_vars['recordings'],
						init_func	= init_burstsyn)

# Characterizing features and parameters for protocol
synburst_characterizing_feats = {
	'Spikecount': {			# (int) The number of peaks during stimulus
		'weight':	1.0,
	},
	'adaptation_index': {	# (float) Normalized average difference of two consecutive ISIs
		'weight':	1.0,
		'double':	{'spike_skipf': 0.0},
		'int':		{'max_spike_skip': 0},
	},
	'ISI_CV': {				# (float) coefficient of variation of ISI durations
		'weight':	1.0,
	},
	'AP_duration_half_width_change': { # (array) Difference of the FWHM of the second and the first action potential divided by the FWHM of the first action potential
		'weight':	1.0,
	},
	'AP_rise_rate':	{		# (array) Voltage change rate during the rising phase of the action potential
		'weight':	1.0,
	},
	'AP_height': {			# (array) The voltages at the maxima of the peak
		'weight':	1.0,
	},
	'AP_amplitude': {		# (array) The relative height of the action potential
		'weight':	1.0,
	},
	'spike_half_width':	{	# (array) The FWHM of each peak
		'weight':	1.0,
	},
	'AHP_time_from_peak': {	# (array) Time between AP peaks and AHP depths
		'weight':	1.0,
	},
	'AHP_depth': {			# (array) relative voltage values at the AHP
		'weight':	1.0,
	},
	'min_AHP_values': {		# (array) Voltage values at the AHP
		'weight':	1.0,
	},
}

# ==============================================================================
# EXPORTED variables

# SETPARAM: characteristic response features and their weights for each protocol
proto_characteristics_feats = {
	plateau_protocol : plateau_characterizing_feats,
	rebound_protocol : rebound_characterizing_feats,
	synburst_protocol: synburst_characterizing_feats,
}

# SETPARAM: stimulus [start, stop] times for feature calculation
proto_response_intervals = {
	plateau_protocol: (plat_start, plat_stop),
	rebound_protocol: (reb_start, reb_stop),
	synburst_protocol: (synburst_start, synburst_stop),
}

# BluePyOpt variables for each protocol
proto_vars = {
	synburst_protocol:	synburst_proto_vars,
	plateau_protocol:	plateau_proto_vars,
	rebound_protocol:	rebound_proto_vars,
}