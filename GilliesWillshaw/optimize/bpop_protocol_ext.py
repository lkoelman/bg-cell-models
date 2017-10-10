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
			init_physio_funcs=None,
			make_stim_funcs=None,
			stimfuncs_kwargs=None,
			rec_traces_funcs=None,
			plot_traces_funcs=None):
		"""
		Constructor
		
		@param init_func	function(sim, model) that takes Simulator and instantiated 
							CellModel (icell) as arguments in that order
		"""

		# init_physio(sim, icell)
		self._funcs_init_physio = [] if init_physio_funcs is None else list(init_physio_funcs)
		# make_stims(sim, icell, stim_data_dict)
		self._funcs_make_stims = [] if make_stim_funcs is None else list(make_stim_funcs)
		# rec_traces(icell, stim_data_dict, trace_spec_data, recorded_hoc_objects)
		self._funcs_rec_traces = [] if rec_traces_funcs is None else list(rec_traces_funcs)
		# plot_traces(trace_rec_data)
		self._funcs_plot_traces = [] if plot_traces_funcs is None else list(plot_traces_funcs)

		# Common keyword arguments for all functions
		self._stimfuncs_kwargs = stimfuncs_kwargs

		# Data used by protocol setup functions
		self.stim_data = {}
		self.rng_info = {'stream_indices': {}} # map from low index to current highest index
		
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
		"""

		try:
			cell_model.freeze(param_values)
			cell_model.instantiate(sim=sim)

			self.instantiate(sim=sim, icell=cell_model.icell)

			try:
				sim.run(self.total_duration, cvode_active=self.cvode_active)
			
			except (RuntimeError, ephys.simulators.NrnSimulatorException):
				
				logger.debug(
					'SweepProtocol: Running of parameter set {%s} generated '
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


	def instantiate(self, sim=None, icell=None):
		"""
		Instantiate

		NOTE: call graph for SweepProtocol is as follows:

			protocol.run(cell_model, param_values, sim) -> _run_func(...) :
				
				model.instantiate()
				
				protocol.instantiate() : <=== THIS FUNCTION
					stimulus.instantiate() :
						location.instantiate()
					recording.instantiate()
				
				sim.run()

		NOTE: operations in this function:

			make_stims(sim, icell, stim_data_dict, rng_info)
			rec_traces(icell, stim_data_dict, trace_spec_data, recorded_hoc_objects)
			init_physio(sim, icell)
		
		"""
		# Make stimuli (inputs)
		default_kwargs = {
			'nrnsim': sim.neuron.h,
			'icell': icell,
			'stim_data': self.stim_data,
			'rng_info': self.rng_info,
		}
		all_kwargs = dict(default_kwargs)

		if self._stimfuncs_kwargs is not None:
			all_kwargs.update(self._stimfuncs_kwargs)

		for stim_func in self._funcs_make_stims:
			stim_func(**all_kwargs)

		# Make custom recordings and store
		if self.record_contained_traces:
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
		for init_func in self._funcs_init_physio:
			init_func(sim, icell)

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


	def destroy(self, sim=None):
		"""
		Destroy protocol
		"""

		# Destroy self-contained stim data
		self.stim_data = None
		self.rng_info = None
		self.trace_spec_data = None
		self.trace_rec_data = None
		self.recorded_hoc_objects = None
		self.recorded_pp_markers = None

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