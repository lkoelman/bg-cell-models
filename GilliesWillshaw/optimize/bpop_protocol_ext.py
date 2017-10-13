"""
Extension of BluePyOpt protocol that allow instantiating stimuli, synapses, etc.
without specifying them in the form of ephys mechanisms & parameters (using bpop.ephys module).

@author Lucas Koelman

@date   7/10/2017

"""

import bluepyopt.ephys as ephys

import logging
logger = logging.getLogger('bpop_ext')

import collections
from common import analysis


class SelfContainedProtocol(ephys.protocols.SweepProtocol):
	"""
	Extension of BluePyOpt protocol that allow instantiating stimuli, synapses, etc. 
	without specifying them in the form of ephys mechanisms & parameters.
	"""

	def __init__(
			self,
			name=None,
			stimuli=None,
			recordings=None,
			cvode_active=None,
			total_duration=None,
			proto_init_funcs=None,
			proto_setup_funcs_pre=None,
			proto_setup_funcs_post=None,
			proto_setup_kwargs_const=None,
			proto_setup_kwargs_getters=None,
			rec_traces_funcs=None,
			plot_traces_funcs=None):
		"""
		Constructor
		
		@param init_func	function(sim, model) that takes Simulator and instantiated 
							CellModel (icell) as arguments in that order

		@param proto_setup_funcs_pre	protocol setup functions that need to be called
										before cell model is instantiated

		@param proto_setup_funcs_post	protocol setup functions that need to be called
										after cell model is instantiated

		@param proto_setup_funcs_kwargs_const	Constant keyword arguments to protocol
												setup functions

		@param proto_setup_funcs_kwargs_getters	Getter functions for keyword arguments to 
												protocol setup functions. These functions
												may create new entries in the passed dict.
		"""

		# init_physio(sim, icell)
		self._proto_init_funcs = [] if proto_init_funcs is None else list(proto_init_funcs)
		# rec_traces(icell, stim_data_dict, trace_spec_data, recorded_hoc_objects)
		self._funcs_rec_traces = [] if rec_traces_funcs is None else list(rec_traces_funcs)
		# plot_traces(trace_rec_data)
		self._funcs_plot_traces = [] if plot_traces_funcs is None else list(plot_traces_funcs)

		# Protocol setup functions applied to full model
		self._proto_setup_funcs_pre = [] if proto_setup_funcs_pre is None else list(proto_setup_funcs_pre)
		self._proto_setup_funcs_post = [] if proto_setup_funcs_post is None else list(proto_setup_funcs_post)

		# Common keyword arguments for all functions
		self._proto_setup_kwargs_const = proto_setup_kwargs_const if proto_setup_kwargs_const is not None else {}
		self._proto_setup_kwargs_getters = proto_setup_kwargs_getters if proto_setup_kwargs_getters is not None else []

		# Data used by protocol setup functions
		self.trace_spec_data = collections.OrderedDict()
		self.trace_rec_data = {}
		self.recorded_hoc_objects = {}
		self.recorded_pp_markers = []
		self.record_contained_traces = False

		########################################################################
		# SweepProtocol parameters

		if stimuli is None:
			stimuli = []

		self._total_duration = total_duration

		super(SelfContainedProtocol, self).__init__(
			name,
			stimuli=stimuli,
			recordings=recordings,
			cvode_active=cvode_active)


	@property
	def total_duration(self):
		"""
		Total duration of protocol (hides SweepProtocol.total_duration)
		"""
		return self._total_duration


	def _run_func(self, cell_model, param_values, sim=None):
		"""
		Run protocols.

		Overrides SweepProtocol._run_func()

		NOTE: call graph for SweepProtocol is as follows:

			protocol.run(cell_model, param_values, sim) -> _run_func(...) :
				
				model.instantiate()
				
				protocol.instantiate() : <=== THIS FUNCTION
					stimulus.instantiate() :
						location.instantiate()
					recording.instantiate()
				
				sim.run()
		"""

		try:

			# Fixes each param.value to individual's 'genes'
			cell_model.freeze(param_values)

			# Pass functions and parameters to cell_model before instantiation
			self.pre_model_instantiate(cell_model=cell_model, sim=sim)
			# Make final cell model
			cell_model.instantiate(sim=sim)
			# Instatiate things that need final cell model
			self.post_model_instantiate(cell_model=cell_model, sim=sim)

			try:
				sim.run(self.total_duration, cvode_active=self.cvode_active)
			
			except (RuntimeError, ephys.simulators.NrnSimulatorException):
				
				logger.debug(
					'SelfContainedProtocol: Running of parameter set {%s} generated '
					'an exception, returning None in responses',
					str(param_values))
				
				responses = {recording.name: None 
								for recording in self.recordings}
			else:
				responses = {recording.name: recording.response
								for recording in self.recordings}

			self.destroy(sim=sim)

			cell_model.destroy(sim=sim)

			cell_model.unfreeze(param_values.keys())

			return responses

		except:
			import sys
			import traceback
			raise Exception(
				"".join(traceback.format_exception(*sys.exc_info())))


	def pre_model_instantiate(self, cell_model=None, sim=None):
		"""
		Function executed before cell model instantiation.
		"""
		# Create keyword arguments for protocol setup functions
		kwargs_default = {
			# 'icell': icell, # icell filled in by cell_model
			'nrnsim': sim.neuron.h,
			'stim_data': {}, # synapses and netcons
			'rng_info': {'stream_indices': {}} # map from low index to current highest index
		}

		# NOTE: proto kwargs must only be instantiated once per model instantiation
		logger.debug("Instantiating protocol setup kwargs...")
		self._this_proto_setup_kwargs = kwargs_default
		self._this_proto_setup_kwargs.update(self._proto_setup_kwargs_const)

		# kwargs getters add keyword arguments on-the-fly
		for kwarg_name, kwarg_getter in self._proto_setup_kwargs_getters.iteritems():
			self._this_proto_setup_kwargs[kwarg_name] = kwarg_getter(self._this_proto_setup_kwargs)

		cell_model.proto_setup_funcs = self._proto_setup_funcs_pre
		cell_model.proto_setup_kwargs = self._this_proto_setup_kwargs


	def post_model_instantiate(self, cell_model=None, sim=None):
		"""
		Function executed after cell model instantiation.
		"""
		# See function below
		self.instantiate(sim=sim, icell=cell_model.icell)


	def instantiate(self, sim=None, icell=None):
		"""
		Instantiate
		"""
		# Make stimuli (inputs)
		for proto_setup_post in self._proto_setup_funcs_post:
			proto_setup_post(**self._this_proto_setup_kwargs)

		# Make custom recordings and store
		if self.record_contained_traces:
			# Custom recording functions
			for rec_func in self._funcs_rec_traces:
				rec_func(icell, self.stim_data, self.trace_spec_data, self.recorded_hoc_objects)

			# Create recording vectors
			recData, markers = analysis.recordTraces(
									self.recorded_hoc_objects, 
									self.trace_spec_data,
									recordStep=0.05)

			self.trace_rec_data = recData
			self.recorded_pp_markers.extend(markers)

		# Initialize physiological conditions
		for init_func in self._proto_init_funcs:
			init_func(**self._this_proto_setup_kwargs)

		# Finally instantiate ephys.stimuli and ephys.recordings
		super(SelfContainedProtocol, self).instantiate(
			sim=sim,
			icell=icell)


	def plot_contained_traces(self):
		"""
		Plot traces recorded as self-contained traces, i.e. traces not passed as
		ephys.recordings objects in constructor.
		"""
		for func in self._funcs_plot_traces:
			func(self.trace_rec_data)

	def get_contained_traces(self):
		"""
		Return contained traces recorded using recording functions.

		@return		collections.OrderedDict<str,h.Vector> : traces dict
		"""
		return self.trace_rec_data


	def destroy(self, sim=None):
		"""
		Destroy protocol
		"""
		logger.debug("Destroying SelfContainedProtocol")
		# Destroy self-contained stim data
		self.trace_spec_data = None
		# self.trace_rec_data = None # Do not destroy: must be available after run()
		self.recorded_hoc_objects = None
		self.recorded_pp_markers = None

		self._this_proto_setup_kwargs = None

		# Make sure stimuli are not active in next protocol if cell model reused
		for stim in self.stimuli:
			if hasattr(stim, 'iclamp'):
				stim.iclamp.amp = 0
				stim.iclamp.dur = 0
			elif hasattr(stim, 'seclamp'):
				for i in range(3):
					setattr(stim.seclamp, 'amp%d' % (i+1), 0)
					setattr(stim.seclamp, 'dur%d' % (i+1), 0)

		# Calls destroy() on each stimulus
		super(SelfContainedProtocol, self).destroy(sim=sim)