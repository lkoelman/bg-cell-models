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

from bpop_ext_gillies import StnFullModel, StnReducedModel, NrnScaleRangeParameter

# Gillies & Willshaw model mechanisms
from gillies_model import gleak_name

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

dend_gbar_params = []
for gbar_name in scaled_gbar:

	gbar_scale_param = NrnScaleRangeParameter(
						name		= gbar_name + '_dend_scale',
						param_name	= gbar_name,
						bounds		= [0.05, 10.0],
						locations	= [dendritic_region],
						frozen		= False)

	dend_gbar_params.append(gbar_scale_param)


all_params = [soma_gl_param, soma_cm_param, dend_gl_factor, dend_cm_factor, 
				dend_ra_param] + dend_gbar_params

# Set default values
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

# PLATEAU protocol (as in article: sequence of three square current pulses)

soma_stim_loc = ephys.locations.NrnSeclistCompLocation(
				name			='soma_stim_loc',
				seclist_name	='somatic',
				sec_index		=0,
				comp_x			=0.5)

I_hyper = -0.17			# hyperpolarize to -70 mV (see fig. 10C)
I_depol = I_hyper + 0.2	# see fig. 10D: 0.2 nA (=stim.amp) over hyperpolarizing current

del_depol = 1000
dur_depol = 50			# see fig. 10D, top right
dur_total = 2000
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

plateau_protocol = ephys.protocols.SweepProtocol('plateau', 
					[stim1_hyp, stim2_dep, stim3_hyp], [plat_rec1])


# SYNAPTIC protocol

################################################################################
# OPTIMIZATION EXPERIMENTS
################################################################################


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

	"""

	proto_feat_dict = {}

	############################################################################
	# Plateau protocol features
	proto = plateau_protocol
	proto_feat_dict[proto] = {} # new map feature_name -> (feature, weight)

	plateau_characterizing_feats = {
		# Timing
		'Spikecount':			1.0,	# (int) The number of peaks during stimulus
		# 'mean_frequency',		0.0,	# (float) the mean frequency of the firing rate
		# 'burst_mean_freq',		# (array) The mean frequency during a burst for each burst
		'adaptation_index':		1.0,	# (float) Normalized average difference of two consecutive ISIs
		'ISI_CV':				1.0,	# (float) coefficient of variation of ISI durations
		# 'ISI_log',					# no documentation
		'AP_duration_change':	1.0,	# (array) Difference of the durations of the second and the first action potential divided by the duration of the first action potential
		'AP_duration_half_width_change': 1.0,# (array) Difference of the FWHM of the second and the first action potential divided by the FWHM of the first action potential
		'AP_rise_time':			1.0,	# (array) Time from action potential onset to the maximum
		'AP_rise_rate':			1.0,	# (array) Voltage change rate during the rising phase of the action potential
		'AP_height':			1.0,	# (array) The voltages at the maxima of the peak
		'AP_amplitude':			1.0,	# (array) The relative height of the action potential
		'spike_half_width':		1.0,	# (array) The FWHM of each peak
		'AHP_time_from_peak':	1.0,	# (array) Time between AP peaks and AHP depths
		'AHP_depth':			1.0,	# (array) relative voltage values at the AHP
		'min_AHP_values':		1.0,	# (array) Voltage values at the AHP
		
	}

	# Make the eFEL features
	for feat_name, feat_weight in plateau_characterizing_feats.iteritems():

		# TODO: this needs to be done on per-feature basis because each feature has (required_features, required_trace_data, required_parameters), as seen in the PDF file describing features. That's whhy currently some features return None.
		feature = ephys.efeatures.eFELFeature(
					name				='{}.{}'.format(proto.name, feat_name),
					efel_feature_name	= feat_name,
					recording_names		= {'': plat_rec1.name},
					stim_start			= plat_start,
					stim_end			= plat_stop,
					exp_mean			= None,	# measure in full model
					exp_std				= None,	# only have one target model: small std
					)

		proto_feat_dict[proto][feat_name] = (feature, feat_weight)


	# TODO: use calculate_feature() on full model, and set exp_mean / exp_std from this in reduced model

	return proto_feat_dict


def calc_feature_targets():
	"""
	Calculate target values for features used in optimization (using full model).
	"""

	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	full_model = StnFullModel(name='StnGillies')

	# Get features used in optimization
	proto_feats = make_opt_features()

	# Run each protocol and get its responses
	for proto, feats in proto_feats.iteritems():

		responses = proto.run(
						cell_model=full_model, 
						param_values={},
						sim=nrnsim,
						isolate=False)

		for feat_name, feat_data in feats.iteritems():
			# Calculate feature value from full model response
			efel_feature, weight = feat_data
			target_value = efel_feature.calculate_feature(responses)

			# Now we can set the target value
			efel_feature.exp_mean = target_value
			efel_feature.exp_std = 0.0

	# return features for each protocol with target values filled in
	return proto_feats


def optimize_main():
	"""
	Optimize a reduced model.
	"""

	red_model = StnReducedModel(
					name		= 'StnFolded',
					fold_method	= 'marasco',
					num_passes	= 7,
					params		= all_params)

	## Test protocol
	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
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

	

	# Make the final objective function based on selected set of good features
	all_opt_features = []
	all_opt_weights = []

	total_objective = ephys.objectives.WeightedSumObjective(
				name = 'optimize_all',
				features = all_opt_features,
				weights = all_opt_weights)


	# Make evaluator to evaluate model using objective calculator

	# Make optimization using the model evaluator

if __name__ == '__main__':
	proto_feats = calc_feature_targets()