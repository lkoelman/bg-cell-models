"""
Automated optimization of a reduced STN cell with respect to 
the full Gillies & Willshaw model.

@author Lucas Koelman
@date	08-02-2017
"""
import re
import collections

# Add common modules to Python path
import sys, os.path
scriptdir, scriptfile = os.path.split(__file__)
modulesbase = os.path.normpath(os.path.join(scriptdir, '..'))
sys.path.append(modulesbase)

# Load NEURON
import neuron
from neuron import h
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library
NRN_MECH_PATH = os.path.normpath(os.path.join(scriptdir, 'nrn_mechs'))
neuron.load_mechanisms(NRN_MECH_PATH)

import numpy as np
from neurotune import optimizers
from neurotune import evaluators
from neurotune import controllers

import reduce_bush_sejnowski as bush
	
# Load NEURON function libraries
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

# Gillies & Willshaw model mechanisms
gillies_mechs_chans = {'STh': ['gpas'], # passive/leak channel
				'Na': ['gna'], 'NaL': ['gna'], # Na channels
				'KDR': ['gk'], 'Kv31': ['gk'], 'sKCa':['gk'], # K channels
				'Ih': ['gk'], # nonspecific channels
				'CaT': ['gcaT'], 'HVA': ['gcaL', 'gcaN'], # Ca channels
				'Cacum': []} # No channels
mechs_chans = gillies_mechs_chans
gleak_name = 'gpas_STh'
glist = [gname+'_'+mech for mech,chans in mechs_chans.iteritems() for gname in chans]

class Protocol:
	""" Experimental protocols """
	SPONTANEOUS = 0
	REBOUND = 2
	PLATEAU = 3

# Saved voltage trace for each protocol
protocol_Vm_paths = {
	Protocol.SPONTANEOUS:"C:\\Users\\lkoelman\\cloudstore_m\\simdata\\fullmodel\\spont_fullmodel_Vm_dt25e-3_0ms_2000ms.csv"
}

# Time interval where voltage traces should be compared for each protocol
protocol_intervals = {
	Protocol.SPONTANEOUS:	(200.0, 1000.0),
	Protocol.REBOUND:		(200.0, 1000.0),
	Protocol.PLATEAU:		(200.0, 1000.0),
}

class Simulation(object):
	""" 
	Helper class for the Controller object to run simulations and generate
	raw simulation data
	"""

	def __init__(self, rec_section, dt=0.025):
		self.sec = rec_section
		self.dt = dt
		self.v_init = v_init
		self.celsius = celsius

	def set_aCSF(self, req):
		""" Set global initial ion concentrations (artificial CSF properties) """
		if req == 3: # Beurrier et al (1999)
			h.nai0_na_ion = 15
			h.nao0_na_ion = 150

			h.ki0_k_ion = 140
			h.ko0_k_ion = 3.6

			h.cai0_ca_ion = 1e-04
			h.cao0_ca_ion = 2.4

			h('cli0_cl_ion = 4') # self-declared Hoc var
			h('clo0_cl_ion = 135') # self-declared Hoc var

		if req == 4: # Bevan & Wilson (1999)
			h.nai0_na_ion = 15
			h.nao0_na_ion = 128.5

			h.ki0_k_ion = 140
			h.ko0_k_ion = 2.5

			h.cai0_ca_ion = 1e-04
			h.cao0_ca_ion = 2.0

			h('cli0_cl_ion = 4')
			h('clo0_cl_ion = 132.5')

		if req == 0: # NEURON's defaults
			h.nai0_na_ion = 10
			h.nao0_na_ion = 140

			h.ki0_k_ion = 54
			h.ko0_k_ion = 2.5

			h.cai0_ca_ion = 5e-05
			h.cao0_ca_ion = 2

			h('cli0_cl_ion = 0')
			h('clo0_cl_ion = 0')

	def set_recording(self, recordStep=0.025):
		""" Set up recording Vectors to record from relevant pointers """
		# Named sections to record from
		secs = {'soma': self.sec}

		# Specify traces
		traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)
		traceSpecs['V_soma'] = {'sec':'soma', 'loc':0.5, 'var':'v'}
		traceSpecs['t_global'] = {'var':'t'}

		# Make trace record Vectors
		self.rec_dt = recordStep
		self.recData = analysis.recordTraces(secs, traceSpecs, self.rec_dt)

	def simulate_protocol(protocol):
		""" Simulate the given experimental protocol """

		if protocol == Protocol.SPONTANEOUS:
			# Spontaneous firing, no stimulation
			h.dt = self.dt
			h.celsius = 37 # different temp from paper (fig 3B: 25degC, fig. 3C: 35degC)
			h.v_init = -60 # paper simulations use default v_init
			self.set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

		elif protocol == Protocol.REBOUND:
			# Plateau potential: short depolarizing pulse at hyperpolarized level
			h.dt = self.dt
			h.celsius = 30 # different temp from paper
			h.v_init = -60 # paper simulations sue default v_init
			set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

			# Stimulation: depolarizing pulse at hyperpolarized level
			stim = h.IClamp(self.sec(0.5))
			self.stim = stim
			I_hyper = -0.17 # hyperpolarize to -70 mV (see fig. 10C)
			I_depol = I_hyper + 0.2 # see fig. 10D: 0.2 nA (=stim.amp) over hyperpolarizing current
			dur_depol = 50 # see fig. 10D, top right
			del_depol = 1000
			self.i_ampvec = h.Vector([I_hyper, I_depol, I_hyper])
			self.i_tvec = h.Vector([0, del_depol, del_depol+dur_depol])
			self.i_ampvec.play(stim, stim._ref_amp, self.i_tvec)

		elif protocol == Protocol.PLATEAU:
			# Rebound burst: response to bried hyperpolarizing pulse
			h.dt = self.dt
			h.celsius = 35 # different temp from paper
			h.v_init = -60 # paper simulations sue default v_init
			set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

			# Stimulation: brief hyperpolarizing pulse, then release
			stim = h.IClamp(self.sec(0.5))
			self.stim = stim
			I_hyper = -0.25 # hyperpolarize to -70 mV (see fig. 10C)
			dur_hyper = 500
			del_hyper = 500
			del_depol = 1000
			stim.amp = I_hyper
			stim.delay = del_hyper
			stim.dur = dur_hyper

		else:
			raise Exception("Unrecognized protocol {0}".format(protocol))

		# Set up recording vectors
		self.set_recording()

		# Run simulation
		h.init() # calls finitialize() and fcurrent()
		t_interval = protocol_intervals[protocol]
		neuron.run(t_interval[1] + 5.0)

		# Return time and voltage traces
		# NOTE: do not extract segment, time axes must match with target trace!
		#       (extraction is done via Evaluator.analysis_start/end_time)
		v_rec = np.array(self.recData['V_soma'])
		t_rec = np.array(self.recData['t_global'])
		return t_rec, v_rec


class STNCellController(object):
	""" 
	The Controller class maps candidate parameter sets to raw data
	by setting up the model for the candidate parameter set and 
	simulating it

	The only interface it must adhere to is providing a run()
	method with the following signature:

	run(candidates: list(list(float)), parameters:list(string)) -> (t_trace, v_trace)
	"""

	def __init__(self, protocol, reduced_model_path):
		""" Make new Controller that builds and runs model starting from
			the given base model.

		@type	protocol			Protocol enum instance
		@param	protocol			Experimental protocol that should be simulated

		@type	reduced_model_path	string
		@param	reduced_model_path	Path to pickle file containing the serialized
									clusters of the reduced model.
		"""
		self.target_protocol = protocol
		self.reduced_model_path = reduced_model_path
		self.clusters = bush.load_clusters(self.reduced_model_path)

	def run(self, candidates, parameters):
		"""
		Run simulation for each candidate

		NOTE: this is the only function where its signature is fixed
			  by an interface/protoype of the neurotune package

		@type	candidates	list(list(float))
		@param	candidates	list of candidate parameter sets (each candidate
							is a list of parameter values)

		@type	parameters	list of parameter names
		@param	parameters	our self-defined list of parameter names: these
							will be used to build the model in some way

		@return				list of (t_trace, v_trace) for each candidate,
							i.e. a list(tuple(list(float), list(float)))
		"""
		traces = []
		for candidate in candidates:
			cand_params = dict(zip(parameters,candidate))
			t,v = self.run_candidate(cand_params)
			traces.append([t,v])

		return traces

	def run_candidate(self, cand_params, show=False):
		"""
		Run simulation for an individual candidate.
		"""

		# Create all Sections
		for sec in h.allsec():
			h.delete_section() # delete existing cells
		eq_secs = bush.rebuild_sections(self.clusters) # stores refs on neuron.h object
		soma_sec = next(sec for sec in eq_secs if sec.name().startswith('soma'))
		dend_secs = [sec for sec in eq_secs if sec is not soma_sec]

		# Pattern matching for gbar factors
		scale_prefix = r'^gbar_sca_'
		scale_pattern = re.compile(scale_prefix)
		gmin_prefix = r'^gbar_min_'
		gmin_pattern = re.compile(gmin_prefix)
		gmax_prefix = r'^gbar_max_'
		gmax_pattern = re.compile(gmax_prefix)

		# Adapt model according to candidate parameters
		for par_name, par_value in cand_params.iteritems():
			scale_match = re.search(scale_pattern, par_name)
			gmin_match = re.search(gmin_pattern, par_name)
			gmax_match = re.search(gmax_pattern, par_name)
			if par_name == 'soma_Ra':
				soma_sec.Ra = par_value

			elif par_name == 'soma_diam_factor':
				for seg in soma_sec:
					seg.diam = seg.diam * par_value

			elif par_name == 'dend_cm_factor':
				for sec in dend_secs:
					for seg in sec:
						seg.cm = seg.cm * par_value

			elif par_name == 'dend_Rm_factor':
				for sec in dend_secs:
					for seg in sec:
						gleak_val = getattr(seg, gleak_name) / par_value
						setattr(seg, gleak_name, gleak_val)

			elif par_name == 'dend_Ra':
				for sec in dend_secs:
					sec.Ra = par_value

			elif par_name == 'dend_diam_factor':
				for sec in dend_secs:
					for seg in sec:
						seg.diam = seg.diam * par_value

			elif scale_match:
				prefix_suffix = re.split(scale_prefix, par_name)
				gname = prefix_suffix[1]
				for sec in dend_secs:
					for seg in sec:
						gval = getattr(seg, gname) * par_value
						setattr(seg, gname, gval)
			else:
				raise Exception("Unrecognized parameter '{}' with value <{}>".format(
								par_name, par_value))

		# Simulate experimental protocol with new cell
		sim = Simulation(soma_sec)
		t_trace, v_trace = sim.simulate_protocol(self.target_protocol)
		return t_trace, v_trace

def optimization_routine():
	""" Main method for the optimization routine """

	# Make a controller to simulate candidates
	reduced_model_path = "C:\\Users\\lkoelman\\cloudstore_m\\simdata\\bush_sejnowski\\stn_reduced_bush.p"
	# Voltage trace of original model for comparison
	target_protocol = Protocol.SPONTANEOUS
	target_Vm_path = protocol_Vm_paths[target_protocol]
	stn_controller = STNCellController(target_protocol, reduced_model_path)

	# Parameters that constitute a candidate ('DNA') and their bounds
	# NOTE: based on fitting routine described in Gillies & Willshaw (2006)
	passive_params_bounds = {
		# soma properties
		'soma_cm_factor': 		(1.0,5.0),
		'soma_Rm_factor': 		(0.5,5.0),
		'soma_Ra':				(100.,300.), # 150 in full model
		'soma_diam_factor':		(0.1,1.0),
		# dendrite properties
		'dend_cm_factor':		(1.0,5.0),
		'dend_Rm_factor':		(0.5,5.0),
		'dend_Ra':				(100.,300.), # 150 in full model
		'dend_diam_factor':		(0.5,2.0),
	}

	active_params_bounds = {
		# soma properties properties
		'soma_gNaL_factor':		(0.5,2.0), # scales constant gNaL distribtuion
		# dendrite properties
		'gbar_min_gk_Ih':		(0.5,2.0), # distal linear dist
		'gbar_max_gk_Ih':		(0.5,2.0),
		'gbar_min_gk_KDR':		(0.5,2.0), # distal step dist
		'gbar_max_gk_KDR':		(0.5,2.0),
		'gbar_min_gk_Kv31':		(0.5,2.0), # proximal linear dist
		'gbar_max_gk_Kv31':		(0.5,2.0),
		'gbar_min_gk_sKCa':		(0.5,2.0), # distal double step dist
		'gbar_max_gk_sKCa':		(0.5,2.0),
		'gbar_min_gcaL_HVA':	(0.5,2.0), # distal linear dist
		'gbar_max_gcaL_HVA':	(0.5,2.0),
		'gbar_min_gcaN_HVA':	(0.5,2.0), # proximal linear dist
		'gbar_max_gcaN_HVA':	(0.5,2.0),
		'gbar_min_gcaT_CaT':	(0.5,2.0), # distal linear dist
		'gbar_max_gcaT_CaT':	(0.5,2.0),
		# dendrite properties: scaling factors
		'gbar_sca_gk_Ih':		(0.5,2.0), # distal linear dist
		'gbar_sca_gk_KDR':		(0.5,2.0), # distal step dist
		'gbar_sca_gk_Kv31':		(0.5,2.0), # proximal linear dist
		'gbar_sca_gk_sKCa':		(0.5,2.0), # distal double step dist
		'gbar_sca_gcaL_HVA':	(0.5,2.0), # distal linear dist
		'gbar_sca_gcaN_HVA':	(0.5,2.0), # proximal linear dist
		'gbar_sca_gcaT_CaT':	(0.5,2.0), # distal linear dist
		'gbar_sca_gna_NaL':		(0.5,2.0), # constant dist
		# 'gbar_sca_gna_Na_factor', # constant dist (negligibly small)
	}

	all_params_bounds = {}
	all_params_bounds.update(passive_params_bounds)
	all_params_bounds.update(active_params_bounds)

	# Parameters to tune spontaneous firing
	spont_params = [
		# soma properties
		'soma_Ra', # 150 in full model
		'soma_diam_factor',
		# dendrite properties
		'dend_cm_factor',
		'dend_Rm_factor',
		'dend_Ra', # 150 in full model
		'dend_diam_factor',
	]
	# Parameters to tune bursts
	burst_params = [
		# soma properties
		'soma_gNaL_factor'
		# dendrite passive properties
		'dend_cm_factor',
		'dend_Rm_factor',
		'dend_Ra',
		'dend_diam_factor',
		# dendrite active properties
		'gbar_sca_gna_NaL',
		'gbar_sca_gk_Ih', # distal linear dist
		'gbar_sca_gk_sKCa', # distal double step dist
		'gbar_sca_gcaL_HVA', # distal linear dist
		'gbar_sca_gcaT_CaT', # distal linear dist
	]
	# Final parameters for current optimization (subset of all parameters)
	final_params = spont_params

	# Parameters for calculation of voltage trace metrics
	trace_analysis_params = {
		'peak_delta':		0, # the value by which a peak or trough has to exceed its neighbours to be considered outside of the noise
		'baseline':			-40., # voltage at which AP width is measured
		'dvdt_threshold':	0, # used in PPTD method described by Van Geit 2007
	}

	# Weight for components of error measure (= targets)
	# NOTE: see metrics in http://pyelectro.readthedocs.io/en/latest/pyelectro.html
	# NOTE: default targets are all keys of in IClampAnalysis.analysis_results dict
	spont_error_weights = {
		# Spike timing/frequency related
		'first_spike_time': 1.0,		# time of first AP
		'max_peak_no': 1.0,				# number of AP peaks
		'min_peak_no': 1.0,				# number of AP throughs
		'spike_frequency_adaptation': 1.0,	# slope of exp fit to initial & final frequency
		'trough_phase_adaptation': 1.0,	# slope of exp fit to phase of first and last through
		'mean_spike_frequency': 1.0,	# mean AP frequency
		'interspike_time_covar': 1.0,	# coefficient of variation of ISIs
		'average_maximum': 1.0,			# average value of AP peaks
		'average_minimum': 1.0,			# average value of AP throughs
		# Spike shape related
		'spike_broadening': 1.0,		# ratio first AP width to avg of following AP widths
		'spike_width_adaptation': 1.0,	# slope of exp fit to first & last AP width
		'peak_decay_exponent': 1.0,		# Decay of AP peaks
		'trough_decay_exponent': 1.0,	# Decay of AP throughs
		'pptd_error':1.0				# Phase-plane trajectory density (see Van Geit (2008))
	}

	final_error_weights = spont_error_weights

	# Make evaluator to map candidate parameter sets to fitness values
	stn_evaluator = evaluators.IClampEvaluator(controller = stn_controller,
												analysis_start_time = protocol_intervals[target_protocol][0],
												analysis_end_time = protocol_intervals[target_protocol][1],
												target_data_path = target_Vm_path,
												parameters = final_params,
												analysis_var = trace_analysis_params,
												weights = final_error_weights,
												targets = None, # because we're using automatic
												automatic = True)

	#make an optimizer
	min_constraints = [all_params_bounds[par][0] for par in final_params]
	max_constraints = [all_params_bounds[par][1] for par in final_params]
	# The optimizer creates an inspyred.ec.EvolutionaryComputation() algorithm
	# and calls its evolve() method (see docs at http://pythonhosted.org/inspyred/reference.html#inspyred.ec.EvolutionaryComputation)
	my_optimizer = optimizers.CustomOptimizerA(max_constraints,
											min_constraints,
											stn_evaluator,
											population_size = 3, # initial number of candidates
											max_evaluations = 100,
											num_selected = 3,
											num_offspring = 3,
											num_elites = 1,
											seeds = None)

	#run the optimizer
	my_optimizer.optimize()

if __name__ == '__main__':
	optimization_routine()
