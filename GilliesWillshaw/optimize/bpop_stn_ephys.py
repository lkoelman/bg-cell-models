"""
Optimization of reduced neuron model using BluePyOpt.

@author	Lucas Koelman

@date	12/09/2017



ARCHITECTURE

- How to make it incremental
	
	- old strategy: strategy in optimize_incremental.py is to make number of reduction passes a parameter of the optimization
		
		- if this is higher than current number of passes: re-initialize cell model and reduce again
		
		- the run() method calls build_candidate(genes) and then runs the protocol
	

	- new strategy A: one Optimization for one cell model created using EPhys module 

		- start from single collapse, save parameters, and write methods to transfer the 'DNA'/parameters to another cell model (re-use DNA from previous optimization)

		- will need to implement own parameters (that the DNA codes for)
			+ control everything via custom MetaParameters
			+ these can be transferred across cell models

		- CONS:
			* have to fit everything into straightjacket of BluePyOpt classes, will will require writing lots of new code, and hinder re-use of existing code

		- PROS:
			* can re-use Bpop examples and code structure for optimization



- Overview of optimization using bluepyopt.EPhys classes:

	- see examples:
		- 'simplecell.ipynb' at https://github.com/BlueBrain/BluePyOpt
		- 'L5PC.ipynb' at https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/L5PC.ipynb
		- MOOC Week 5 Graded exercise (simdata folder)

	- create cell model

	- define parameters

		+ control everything via custom MetaParameters

	- for each StimulationProtocol: create one ephys.protocols.SweepProtocol
		
		+ allows to add generic stimuli
			- see http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.protocols.html
		
		+ can have multiple Stimuli objects + recording objects
			- http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.stimuli.html
			- http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.recordings.html


	- Define objectives (feature + target value/range)

		+ for each (feature, target_value) combination : create ephys.efeature.eFELFeature 
			- see http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.efeatures.html

		+ create a WeightedSumObjective to calculate fitness
			- http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.objectives.html


	- create Evaluator: ephys.evaluators.CellEvaluator
		+ http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.evaluators.html


	- create Optimization: bpop.deapext.optimisations.DEAPOptimisation

		+ see http://bluepyopt.readthedocs.io/en/latest/deapext/bluepyopt.deapext.optimisations.html#module-bluepyopt.deapext.optimisations

		+ IBEADEAPOptimisation is same as DEAPOptimisation (subclass without extras)


"""



import bluepyopt as bpop
import bluepyopt.ephys as ephys

from bpop_ext_gillies import StnFullModel, StnReducedModel
from bpop_ext_opt import (
	NrnScaleRangeParameter, NrnOffsetRangeParameter, 
	PhysioProtocol, NrnSpaceClamp
	)

# Gillies & Willshaw model mechanisms
import gillies_model
gleak_name = gillies_model.gleak_name

################################################################################
# MODEL REGIONS
################################################################################

# seclist_name are names of SectionList declared in the cell model we optimize

somatic_region = ephys.locations.NrnSeclistLocation('somatic', seclist_name='somatic')

dendritic_region = ephys.locations.NrnSeclistLocation('somatic', seclist_name='dendritic')

################################################################################
# MODEL PARAMETERS
################################################################################

# SOMATIC PARAMETERS

soma_gl_param = ephys.parameters.NrnSectionParameter(                                    
						name='gleak_soma',		# assigned name
						param_name=gleak_name,	# NEURON name
						locations=[somatic_region],
						bounds=[7.84112e-7, 7.84112e-3],	# default: 7.84112e-05
						frozen=False)

soma_cm_param = ephys.parameters.NrnSectionParameter(
						name='cm_soma',
						param_name='cm',
						bounds=[0.05, 10.0],	# default: 1.0
						locations=[somatic_region],
						frozen=False)


# DENDRITIC PARAMETERS

# For dendrites, conductances and cm have been scaled by the reduction method.
# We want to keep their spatial profile/distribution, i.e. just scale these.
# Hence: need to define our own parameter that scales these distributions.

# TODO: choose right parameter type (look at instantiate() method)
dend_gl_factor = NrnScaleRangeParameter(
					name='gleak_dend_scale',
					param_name=gleak_name,
					bounds=[0.05, 10.0],
					locations=[dendritic_region],
					frozen=False)

dend_cm_factor = NrnScaleRangeParameter(
					name='cm_dend_scale',
					param_name=gleak_name,
					bounds=[0.05, 10.0],
					locations=[dendritic_region],
					frozen=False)

dend_ra_param = ephys.parameters.NrnSectionParameter(
					name='Ra_dend',
					param_name='Ra',
					bounds=[50, 500.0],			# default: 150.224
					locations=[dendritic_region],
					frozen=False)

# for set of most important active conductance: scale factor
scaled_gbar = ['gna_NaL', 'gk_Ih', 'gk_sKCa', 'gcaT_CaT', 'gcaL_HVA', 'gcaN_HVA']

# Make parameters to scale channel conductances
dend_gbar_params = []
for gbar_name in scaled_gbar:

	gbar_scale_param = NrnScaleRangeParameter(
						name		= gbar_name + '_dend_scale',
						param_name	= gbar_name,
						bounds		= [0.05, 10.0],
						locations	= [dendritic_region],
						frozen		= False)

	dend_gbar_params.append(gbar_scale_param)


# Groups of parameters to be used in optimizations
passive_params = [soma_gl_param, soma_cm_param, dend_gl_factor, dend_cm_factor, 
				dend_ra_param]

active_params = dend_gbar_params

all_params = passive_params + active_params


# Default values for parameters
default_params = {
	'gleak_soma':		7.84112e-05,
	'cm_soma':			1.0,
	'Ra_dend':			150.224,
	'cm_dend_scale':	1.0,
	'gleak_dend_scale':	1.0,
}
for param in dend_gbar_params:
	default_params[param.name] = 1.0

################################################################################
# PROTOCOLS
################################################################################

# Location for current clamp
soma_stim_loc = ephys.locations.NrnSeclistCompLocation(
				name			='soma_stim_loc',
				seclist_name	='somatic',
				sec_index		=0,		# index in SectionList
				comp_x			=0.5)	# x-location in Section

# ==============================================================================
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
				location		= soma_stim_loc,
				total_duration	= del_depol)

stim2_dep = ephys.stimuli.NrnSquarePulse(
				step_amplitude	= I_depol,
				step_delay		= del_depol,
				step_duration	= dur_depol,
				location		= soma_stim_loc,
				total_duration	= del_depol + dur_depol)

stim3_hyp = ephys.stimuli.NrnSquarePulse(
				step_amplitude	= I_hyper,
				step_delay		= del_depol + dur_depol,
				step_duration	= del_depol,
				location		= soma_stim_loc,
				total_duration	= dur_total)

plat_rec1 = ephys.recordings.CompRecording(
				name			= 'plateau.soma.v',
				location		= soma_stim_loc,
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

# ==============================================================================
# REBOUND protocol

dur_hyper = 500

reb_clmp1 = NrnSpaceClamp(
				step_amplitudes	= [0, 0, -75],
				step_durations	= [0, 0, dur_hyper],
				total_duration	= 2000,
				location		= soma_stim_loc)


reb_rec1 = ephys.recordings.CompRecording(
				name			= 'rebound.soma.v',
				location		= soma_stim_loc,
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

# ==============================================================================
# TODO: SYNAPTIC protocol

proto_characteristics_feats = {
	plateau_protocol : plateau_characterizing_feats,
	rebound_protocol : rebound_characterizing_feats,
}

proto_response_intervals = {
	plateau_protocol: (plat_start, plat_stop),
	rebound_protocol: (reb_start, reb_stop)
}

################################################################################
# OPTIMIZATION EXPERIMENTS
################################################################################

def make_features(protocol, feats_weights_params, stim_interval):
	"""
	Make eFEL features.

	@return		a dictionary {feature_names: feature_objects}
	"""
	candidate_feats = {}
	default_trace = {'': protocol.recordings[0].name}

	for feat_name, feat_params in feats_weights_params.iteritems():

		feature = ephys.efeatures.eFELFeature(
						name				='{}.{}'.format(protocol.name, feat_name),
						efel_feature_name	= feat_name,
						recording_names		= feat_params.get('traces', default_trace),
						stim_start			= stim_interval[0],
						stim_end			= stim_interval[1],
						double_settings		= feat_params.get('double', None),
						int_settings		= feat_params.get('int', None),
						)

		candidate_feats[feat_name] = feature

	return candidate_feats


def make_opt_features():
	"""
	Make features that associate each protocol with some relevant metrics.

	@return		dict(protocol : dict(feature_name : tuple(feature, weight)))

					I.e. a dictionary that maps ephys.protocol objects to another
					dictionary, that maps feature names to a feature object and
					its weight for the optimization.

	@note	available features:
				- import efel; efel.getFeatureNames()
				- see http://efel.readthedocs.io/en/latest/eFeatures.html
				- see pdf linked there (ctrl+f: feature name with underscores as spaces)
				- each feature has specific (required_features, required_trace_data, required_parameters)

	"""

	opt_used_protocols = [plateau_protocol, rebound_protocol] # SETPARAM: protocols used for optimization

	proto_feat_dict = {}

	# For each protocol used in optimization: make the Feature objects
	for proto in opt_used_protocols:
		proto_feat_dict[proto] = {} # feature_name -> (feature, weight)

		# Make eFEL features
		candidate_feats = make_features(
							proto,
							proto_characteristics_feats[proto],
							proto_response_intervals[proto])

		# Add them to dict
		for feat_name in candidate_feats.keys():
			feat_weight = proto_characteristics_feats[proto][feat_name]['weight']
			
			# Add to feature dict if nonzero weight
			if feat_weight > 0.0:
				feat_obj = candidate_feats[feat_name]
				proto_feat_dict[proto][feat_name] = feat_obj, feat_weight

	return proto_feat_dict


def get_features_targets():
	"""
	Calculate target values for features used in optimization (using full model).
	"""

	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	full_model = StnFullModel(name='StnGillies')

	# Get features used in optimization
	proto_feats = make_opt_features()

	# Run each protocol and get its responses
	for proto, feats in proto_feats.iteritems():

		# Run protocol with full cell model
		responses = proto.run(
						cell_model=full_model, 
						param_values={},
						sim=nrnsim,
						isolate=False)

		# Use response to calculate target value for each features
		for feat_name, feat_data in feats.iteritems():

			# Calculate feature value from full model response
			efel_feature, weight = feat_data
			target_value = efel_feature.calculate_feature(responses)

			# Now we can set the target value
			efel_feature.exp_mean = target_value
			efel_feature.exp_std = 0.0

	# return features for each protocol with target values filled in
	return proto_feats


def test_full_model(proto):
	"""
	Test stimulation protocol applied to full cell model.
	"""

	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	# Create Gillies & Willshaw (2005) STN model
	full_model = StnFullModel(name='StnGillies')

	# Apply protocol and simulate
	responses = proto.run(
					cell_model=full_model, 
					param_values={},
					sim=nrnsim,
					isolate=False)

	## Plot protocol responses
	from matplotlib import pyplot as plt
	for resp_name, traces in responses.iteritems():
		plt.figure()
		plt.plot(traces['time'], traces['voltage'])
		plt.suptitle(resp_name)
	plt.show(block=False)


def test_reduced_model():
	"""
	Test stimulation protocol applied to reduced cell model.
	"""

	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	# Create reduced model
	red_model = StnReducedModel(
					name		= 'StnFolded',
					fold_method	= 'marasco',
					num_passes	= 7,
					params		= all_params)

	# Test protocol
	responses = plateau_protocol.run(
					cell_model=red_model, 
					param_values=default_params,
					sim=nrnsim, 
					isolate=True) # SETPARAM: set to True for multiprocessing

	## Plot protocol responses
	from matplotlib import pyplot as plt
	for resp_name, traces in responses.iteritems():
		plt.figure()
		plt.plot(traces['time'], traces['voltage'])
		plt.suptitle(resp_name)
	plt.show(block=False)


def optimize_active():
	"""
	Optimization routine for reduced cell model.
	"""
	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	############################################################################
	# Parameters & Cell Model

	# Make parameters
	gleak_orig = 7.84112e-05
	gleak_fit = 12.43169e-5 # fit to match Zin_DC (see praxis_passive.py)
	dend_gl_param = ephys.parameters.NrnSectionParameter(
						name		= 'gleak_dend_param',
						param_name	= gleak_name,
						locations	= [dendritic_region],
						bounds		= [gleak_fit*1e-1, gleak_fit*1e1],
						value		= gleak_fit,
						frozen		= True)

	cm_orig = 1.0
	cm_fit = cm_orig * (gleak_fit / gleak_orig) # preserve membrane time constant
	dend_cm_param = ephys.parameters.NrnSectionParameter(
						name		= 'cm_dend_param',
						param_name	= 'cm',
						locations	= [dendritic_region],
						bounds		= [1.0, 1.0],
						value		= cm_fit,
						frozen		= True)

	# Frozen parameters are passive parameters fit previously in passive model
	dend_passive_params = [dend_gl_param, dend_cm_param]

	# Free parameters are active conductances with large impact on response
	dend_active_params = dend_gbar_params # Free parameters
	dend_all_params = dend_passive_params + dend_active_params

	# Create reduced model
	## TODO: active gbar scalers: set to additive?
	red_model = StnReducedModel(
					name		= 'StnFolded',
					fold_method	= 'marasco',
					num_passes	= 7,
					params		= dend_all_params)

	############################################################################
	# Features & Objectives

	# Get dictionary with Feature objects for each protocol
	protos_feats = get_features_targets()

	# Collect characteristic features for all protocols used in evaluation
	all_opt_features = []
	all_opt_weights = []
	for proto, char_feats in protos_feats.iteritems():
		feats, weights = zip(*char_feats.items()) # get list(keys), list(values)
		all_opt_features.extend(feats)
		all_opt_weights.extend(weights)

	# Make final objective function based on selected set of features
	total_objective = ephys.objectives.WeightedSumObjective(
				name = 'optimize_all',
				features = all_opt_features,
				weights = all_opt_weights)

	############################################################################
	# Evaluators & Optimization

	# Make evaluator to evaluate model using objective calculator
	score_calc = ephys.objectivescalculators.ObjectivesCalculator([total_objective])

	eval_protos = {proto.name: proto for proto in protos_feats.keys()}
	cell_evaluator = ephys.evaluators.CellEvaluator(
						cell_model			= red_model,
						param_names			= ['gnabar_hh', 'gkbar_hh'],
						fitness_protocols	= eval_protos,
						fitness_calculator	= score_calc,
						sim					= nrnsim)

	# Make optimization using the model evaluator
	optimisation = bpop.optimisations.DEAPOptimisation(
						evaluator		= cell_evaluator,
						offspring_size	= 10)



	final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=5)




if __name__ == '__main__':
	proto_feats = get_features_targets()