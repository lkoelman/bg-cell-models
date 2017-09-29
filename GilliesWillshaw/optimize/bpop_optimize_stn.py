"""
Optimization of reduced neuron model using BluePyOpt.

@author	Lucas Koelman

@date	12/09/2017


NOTES

- see SETPARAM comment for parameters that should be set by user

"""

import pickle

import bluepyopt as bpop
import bluepyopt.ephys as ephys

from bpop_cellmodels import StnFullModel, StnReducedModel
from bpop_extensions import NrnScaleRangeParameter, NrnOffsetRangeParameter
from bpop_protocols_stn import (
	plateau_protocol, rebound_protocol, synburst_protocol,
	proto_characteristics_feats, proto_response_intervals, proto_vars
	)

# Gillies & Willshaw model mechanisms
import gillies_model
gleak_name = gillies_model.gleak_name

from evalmodel.cellpopdata import StnModel

# Adjust verbosity of loggers
import logging
silent_loggers = ['marasco']
verbose_loggers = []
for logname in silent_loggers:
	logger = logging.getLogger('marasco')
	if logger: logger.setLevel(logging.WARNING)

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


# SETPARAM: filepath of saved responses
PROTO_RESPONSES_FILE = "/home/luye/cloudstore_m/simdata/fullmodel/STN_Gillies2005_proto_responses.pkl"

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

	opt_used_protocols = [plateau_protocol, rebound_protocol, synburst_protocol] # SETPARAM: protocols used for optimization

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


def save_proto_responses(protocols, cellmodel, filepath):
	"""
	Run protocols using given cell model and save responses to file.
	"""
	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	# Run each protocol and get its responses
	responses = {}
	for proto in protocols:

		# Run protocol with full cell model
		responses[proto.name] = proto.run(
						cell_model=cellmodel, 
						param_values={},
						sim=nrnsim,
						isolate=False)

	# Save to file
	with open(filepath, 'w') as recfile:
		pickle.dump(responses, recfile)

	print("Saved responses to file {}".format(filepath))


def load_proto_responses(filepath):
	"""
	Load protocol responses from pickle file.

	@return		dictionary {protocol: responses}
	"""
	with open(filepath, 'r') as recfile:
		responses = pickle.load(recfile)
		return responses


def get_features_targets(saved_responses=None):
	"""
	Calculate target values for features used in optimization (using full model).

	@param saved_responses		file path to pickled responses dictionary

	@effect		get features used in optimization, then calculates their target
				values using the full STN model.

	@return		dictionary {protocol : {feature_name : (feature, weight) } }

					I.e. a dictionary that maps ephys.protocol objects to another
					dictionary, that maps feature names to a feature object and
					its weight for the optimization.
	"""

	if saved_responses is not None:
		proto_responses = load_proto_responses(saved_responses)
	else:
		nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
		full_model = StnFullModel(name='StnGillies')

	# Get features used in optimization
	proto_feats = make_opt_features()

	# Run each protocol and get its responses
	for proto, feats in proto_feats.iteritems():

		# Get response traces
		if saved_responses is not None:
			responses = proto_responses[proto.name]
		else:
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


def test_protocol(proto, cellmodel, export_locals=True):
	"""
	Test stimulation protocol applied to full cell model.

	@param	cellmodel		either a CellModel object, StnModel enum instance,
							or string "full" / "reduced"

	EXAMPLE USAGE:

	proto = synburst_protocol
	test_protocol(proto, 'full')
	
	"""
	# Get protocol mechanisms that need to be isntantiated
	proto_mechs = proto_vars[proto].get('pp_mechs', []) + \
	              proto_vars[proto].get('range_mechs', [])

	proto_params = proto_vars[proto].get('pp_mech_params', [])

	# Make the model
	if cellmodel in ('full', StnModel.Gillies2005):
		cellmodel = StnFullModel(name='StnGillies')
	
	elif cellmodel in ('reduced', StnModel.Gillies_FoldMarasco):
		cellmodel = StnReducedModel(
						name		= 'StnFolded',
						fold_method	= 'marasco',
						num_passes	= 7,
						mechs		= proto_mechs,
						params		= proto_params)

	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	# Apply protocol and simulate
	responses = proto.run(
					cell_model		= cellmodel, 
					param_values	= {},
					sim				= nrnsim,
					isolate			= False) # allows us to query cell with h.allsec()

	## Plot protocol responses
	from matplotlib import pyplot as plt
	for resp_name, traces in responses.iteritems():
		plt.figure()
		plt.plot(traces['time'], traces['voltage'])
		plt.suptitle(resp_name)
	plt.show(block=False)

	if export_locals:
		globals().update(locals())


def inspect_protocol(proto, cellmodel, export_locals=True):
	"""
	Test stimulation protocol applied to full cell model.

	@param	cellmodel		either a CellModel object, StnModel enum instance,
							or string "full" / "reduced"

	EXAMPLE USAGE:

	proto = synburst_protocol
	test_protocol(proto, 'full')
	
	"""
	# Get protocol mechanisms that need to be isntantiated
	proto_mechs = proto_vars[proto].get('pp_mechs', []) + \
	              proto_vars[proto].get('range_mechs', [])

	proto_params = proto_vars[proto].get('pp_mech_params', [])

	# Make the model
	if cellmodel in ('full', StnModel.Gillies2005):
		cellmodel = StnFullModel(name='StnGillies')
	
	elif cellmodel in ('reduced', StnModel.Gillies_FoldMarasco):
		cellmodel = StnReducedModel(
						name		= 'StnFolded',
						fold_method	= 'marasco',
						num_passes	= 7,
						mechs		= proto_mechs,
						params		= proto_params)

	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	# protocol.run() cleans up after running: therefore instantiate everything ourselves
	cellmodel.freeze({}) # TODO: do you need to freeze model params to some value here?
	cellmodel.instantiate(sim=nrnsim)
	proto.instantiate(sim=nrnsim, icell=cellmodel.icell)


	if export_locals:
		globals().update(locals())


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

	# FROZEN PARAMETERS are passive parameters fit previously in passive model
	dend_passive_params = [dend_gl_param, dend_cm_param]

	# FREE PARAMETERS are active conductances with large impact on response
	dend_active_params = dend_gbar_params # Free parameters
	dend_all_params = dend_passive_params + dend_active_params

	############################################################################
	# Features & Objectives

	# Get dictionary with Feature objects for each protocol
	protos_feats = get_features_targets(saved_responses=PROTO_RESPONSES_FILE)

	# Collect characteristic features for all protocols used in evaluation
	all_opt_features = []
	all_opt_weights = []
	for proto, featdict in protos_feats.iteritems():
		feats, weights = zip(*featdict.values()) # values ist list of (feature, weight)
		all_opt_features.extend(feats)
		all_opt_weights.extend(weights)

	# Make final objective function based on selected set of features
	total_objective = ephys.objectives.WeightedSumObjective(
				name = 'optimize_all',
				features = all_opt_features,
				weights = all_opt_weights)

	############################################################################
	# Evaluators & Optimization

	# Create reduced model
	## TODO: experiment with gbar param type: scale/offset
	red_model = StnReducedModel(
					name		= 'StnFolded',
					fold_method	= 'marasco',
					num_passes	= 7,
					params		= dend_all_params)

	# Make evaluator to evaluate model using objective calculator
	score_calc = ephys.objectivescalculators.ObjectivesCalculator([total_objective])

	eval_protos = {proto.name: proto for proto in protos_feats.keys()}
	eval_params = [param.name for param in dend_active_params]
	
	# 
	cell_evaluator = ephys.evaluators.CellEvaluator(
						cell_model			= red_model,
						param_names			= eval_params, # fitted parameters
						fitness_protocols	= eval_protos,
						fitness_calculator	= score_calc,
						sim					= nrnsim,
						isolate_protocols	= True)

	# Make optimization using the model evaluator
	optimisation = bpop.optimisations.DEAPOptimisation(
						evaluator		= cell_evaluator,
						offspring_size	= 10)



	final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=5)




if __name__ == '__main__':
	# Calculate features for each optimization protocol
	# proto_feats = get_features_targets()

	# Save full model responses
	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
	full_model = StnFullModel(name='StnGillies')
	protocols = proto_characteristics_feats.keys()
	save_proto_responses(protocols, full_model, PROTO_RESPONSES_FILE)