"""
Optimization of reduced neuron model using BluePyOpt.

@author	Lucas Koelman

@date	12/09/2017


NOTES

- see SETPARAM comment for parameters that should be set by user

"""

import pickle

# BluePyOpt modules
import bluepyopt as bpop
import bluepyopt.ephys as ephys

# Our custom BluePyOpt modules
from bpop_cellmodels import StnFullModel, StnReducedModel
import bpop_protocols_stn as StnProtocols
from bpop_protocols_stn import BpopProtocolWrapper
import bpop_features_stn as StnFeatures
import bpop_parameters_stn as StnParameters

# Gillies & Willshaw model mechanisms
import gillies_model
gleak_name = gillies_model.gleak_name

# Physiology parameters
from evalmodel.cellpopdata import StnModel
from evalmodel.proto_common import StimProtocol
CLAMP_PLATEAU = StimProtocol.CLAMP_PLATEAU
CLAMP_REBOUND = StimProtocol.CLAMP_REBOUND
MIN_SYN_BURST = StimProtocol.MIN_SYN_BURST

# Adjust verbosity of loggers
import logging
silent_loggers = ['marasco']
verbose_loggers = []
for logname in silent_loggers:
	logger = logging.getLogger('marasco')
	if logger: logger.setLevel(logging.WARNING)



# SETPARAM: filepath of saved responses
PROTO_RESPONSES_FILE = "/home/luye/cloudstore_m/simdata/fullmodel/STN_Gillies2005_proto_responses.pkl" # backup is in filename.old.pkl


################################################################################
# OPTIMIZATION EXPERIMENTS
################################################################################


def plot_proto_responses(proto_responses):
	"""
	Plot responses stored in a dictionary.
	"""
	from matplotlib import pyplot as plt
	for proto_name, responses in proto_responses.iteritems():
		
		fig, axes = plt.subplots(len(responses))
		try:
			iter(axes)
		except TypeError:
			axes = [axes]

		for index, (resp_name, response) in enumerate(sorted(responses.items())):
			axes[index].plot(response['time'], response['voltage'], label=resp_name)
			axes[index].set_title(resp_name)
		
		fig.tight_layout()
	
	plt.show(block=False)


def save_proto_responses(responses, filepath):
	"""
	Save protocol responses to file.
	"""

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


def run_proto_responses(cell_model, ephys_protocols):
	"""
	Run protocols using given cell model and return responses,
	indexed by protocol.name.
	"""
	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	# Run each protocol and get its responses
	all_responses = {}
	for e_proto in ephys_protocols:

		response = e_proto.run(
						cell_model		= cell_model, 
						param_values	= {},
						sim				= nrnsim,
						isolate			= False)

		all_responses[e_proto.name] = response

	return all_responses


def test_protocol(stim_proto, model_type, export_locals=True):
	"""
	Test stimulation protocol applied to full cell model.

	@param	model_type		either a CellModel object, StnModel enum instance,
							or string "full" / "reduced"

	EXAMPLE:

		test_protocol(StimProtocol.MIN_SYN_BURST, 'full')
	
	"""
	if model_type in ('full', StnModel.Gillies2005):
		model_type = StnModel.Gillies2005
		fullmodel = True
	elif model_type in ('reduced', StnModel.Gillies_FoldMarasco):
		model_type = StnModel.Gillies_FoldMarasco
		fullmodel = False


	# instantiate protocol
	proto = BpopProtocolWrapper.make(stim_proto, model_type)

	# Get protocol mechanisms that need to be isntantiated
	proto_mechs = proto.proto_vars.get('pp_mechs', []) + \
				  proto.proto_vars.get('range_mechs', [])

	proto_params = proto.proto_vars.get('pp_mech_params', [])

	# Make cell model
	if fullmodel:
		cellmodel = StnFullModel(
						name		= 'StnGillies',
						mechs		= proto_mechs,
						params		= proto_params)
	else:
		cellmodel = StnReducedModel(
						name		= 'StnFolded',
						fold_method	= 'marasco',
						num_passes	= 7,
						mechs		= proto_mechs,
						params		= proto_params)

	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	# Apply protocol and simulate
	responses = proto.ephys_protocol.run(
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


def inspect_protocol(stim_proto, model_type, export_locals=True):
	"""
	Test stimulation protocol applied to full cell model.

	@param	model_type		either a CellModel object, StnModel enum instance,
							or string "full" / "reduced"

	EXAMPLE USAGE:

	proto = synburst_protocol
	test_protocol(proto, 'full')
	
	"""
	if model_type in ('full', StnModel.Gillies2005):
		model_type = StnModel.Gillies2005
		fullmodel = True
	elif model_type in ('reduced', StnModel.Gillies_FoldMarasco):
		model_type = StnModel.Gillies_FoldMarasco
		fullmodel = False


	# instantiate protocol
	proto = BpopProtocolWrapper.make(stim_proto, model_type)

	# Get protocol mechanisms that need to be isntantiated
	proto_mechs = proto.proto_vars.get('pp_mechs', []) + \
				  proto.proto_vars.get('range_mechs', [])

	proto_params = proto.proto_vars.get('pp_mech_params', [])

	# Make cell model
	if fullmodel:
		cellmodel = StnFullModel(
						name		= 'StnGillies',
						mechs		= proto_mechs,
						params		= proto_params)
	else:
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
						locations	= [StnParameters.dendritic_region],
						bounds		= [gleak_fit*1e-1, gleak_fit*1e1],
						value		= gleak_orig, # SETPARAM: use fitted gl value
						frozen		= True)

	cm_orig = 1.0
	cm_fit = cm_orig * (gleak_fit / gleak_orig) # preserve membrane time constant
	dend_cm_param = ephys.parameters.NrnSectionParameter(
						name		= 'cm_dend_param',
						param_name	= 'cm',
						locations	= [StnParameters.dendritic_region],
						bounds		= [1.0, 1.0],
						value		= cm_orig, # SETPARAM: use fitted cm value
						frozen		= True)

	# FROZEN PARAMETERS are passive parameters fit previously in passive model
	dend_passive_params = [dend_gl_param, dend_cm_param]

	# FREE PARAMETERS are active conductances with large impact on response
	dend_active_params = StnParameters.dend_gbar_params # Free parameters
	dend_all_params = dend_passive_params + dend_active_params

	# NOTE: we don't need to define NrnModMechanism for these parameters, since they are not inserted by BluePyOpt but by our custom model setup code

	############################################################################
	# Protocols for optimization

	stn_model_type = StnModel.Gillies_FoldMarasco # SETPARAM: model type to optimize

	# Protocols to use for optimization
	opt_stim_protocols = [CLAMP_REBOUND, MIN_SYN_BURST]

	# Make all protocol data
	red_protos = {stim_proto: BpopProtocolWrapper.make(stim_proto, stn_model_type) for stim_proto in opt_stim_protocols}

	# Collect al frozen mechanisms and parameters required for protocols to work
	proto_mechs, proto_params = BpopProtocolWrapper.all_mechs_params(red_protos.values())

	# Create reduced model
	red_model = StnReducedModel(
					name		= 'StnFolded',
					fold_method	= 'marasco',
					num_passes	= 7,
					mechs		= proto_mechs,
					params		= proto_params + dend_all_params)

	############################################################################
	# Features & Objectives

	# Get protocol responses for full model
	if PROTO_RESPONSES_FILE is not None:
		full_responses = load_proto_responses(PROTO_RESPONSES_FILE)
	else:
		full_protos = [BpopProtocolWrapper.make(stim_proto, stn_model_type) for stim_proto in opt_stim_protocols]
		full_mechs, full_params = BpopProtocolWrapper.all_mechs_params(full_protos)
		full_model = StnFullModel(
					name		= 'StnGillies',
					mechs		= full_mechs,
					params		= full_params)
		full_responses = run_proto_responses(full_model, full_protos)

	# Make EFEL feature objects
	stimprotos_feats = StnFeatures.make_opt_features(red_protos.values())

	# Calculate target values from full model responses
	StnFeatures.calc_feature_targets(stimprotos_feats, full_responses)

	# Collect characteristic features for all protocols used in evaluation
	all_opt_features, all_opt_weights = StnFeatures.all_features_weights(stimprotos_feats.values())

	# Make final objective function based on selected set of features
	total_objective = ephys.objectives.WeightedSumObjective(
				name = 'optimize_all',
				features = all_opt_features,
				weights = all_opt_weights)

	# Calculator maps model responses to scores
	score_calc = ephys.objectivescalculators.ObjectivesCalculator([total_objective])

	############################################################################
	# Evaluators & Optimization

	# Make evaluator to evaluate model using objective calculator
	opt_ephys_protos = {k.name: v.ephys_protocol for k,v in red_protos.iteritems()}
	opt_params_names = [param.name for param in dend_active_params]
	
	# TODO: check that: synapse parameters must be on model but unactive during clamp protocols
	cell_evaluator = ephys.evaluators.CellEvaluator(
						cell_model			= red_model,
						param_names			= opt_params_names, # fitted parameters
						fitness_protocols	= opt_ephys_protos,
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
	model_type = StnModel.Gillies2005

	# Make all available protocols
	all_stim_protos = StnProtocols.PROTOCOL_WRAPPERS.keys()
	full_protos = [BpopProtocolWrapper.make(stim_proto, model_type) for stim_proto in all_stim_protos]

	# Make cell model
	full_mechs, full_params = BpopProtocolWrapper.all_mechs_params(full_protos)
	full_model = StnFullModel(
					name		= 'StnGillies',
					mechs		= full_mechs,
					params		= full_params)

	# Run protocols and save responses
	ephys_protos = [proto.ephys_protocol for proto in full_protos]
	full_responses = run_proto_responses(full_model, ephys_protos)
	
	save_proto_responses(full_responses, PROTO_RESPONSES_FILE)

	# Load and plot
	# responses = load_proto_responses(PROTO_RESPONSES_FILE)
	# plot_proto_responses(responses)