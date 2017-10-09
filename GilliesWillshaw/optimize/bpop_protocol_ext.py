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


class ContainedProtocol(ephys.protocols.SweepProtocol):
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
			init_physio_funcs=None,
			make_stims_funcs=None,
			rec_traces_funcs=None,
			plot_traces_funcs=None):
		"""
		Constructor
		
		Args:
			init_func:  function(sim, model) that takes Simulator and instantiated 
						CellModel (icell) as arguments in that order
		"""

		# init_physio(sim, icell)
		self._funcs_init_physio = [] if init_physio_funcs is None else list(init_physio_funcs)
		# make_stims(sim, icell, stim_data_dict)
		self._funcs_make_stims = [] if make_stims_funcs is None else list(make_stims_funcs)
		# rec_traces(icell, stim_data_dict, trace_spec_data, recorded_hoc_objects)
		self._funcs_rec_traces = [] if rec_traces_funcs is None else list(rec_traces_funcs)
		# plot_traces(trace_rec_data)
		self._funcs_plot_traces = [] if plot_traces_funcs is None else list(plot_traces_funcs)

		self.stim_data = {}
		self.trace_spec_data = collections.OrderedDict()
		self.trace_rec_data = {}
		self.recorded_hoc_objects = {}
		self.recorded_pp_markers = []
		self.record_contained_traces = False


		super(ContainedProtocol, self).__init__(
			name,
			stimuli=stimuli,
			recordings=recordings,
			cvode_active=cvode_active)


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

			make_stims(sim, icell, stim_data_dict)
			rec_traces(icell, stim_data_dict, trace_spec_data, recorded_hoc_objects)
			init_physio(sim, icell)
		
		TODO: make StnModelEvaluator-specific functionwrap a general function that conforms to the same protocol as here (i.e. pass it a dictionary). Start only with the protocols you need, e.g. proto_bacgkround.
		"""
		# Make inputs and store
		for func in self._funcs_make_stims:
			func(sim, icell, self.stim_data)

		# Make custom recordings and store
		if self.record_contained_traces:
			for func in self._funcs_rec_traces:
				func(icell, self.stim_data, self.trace_spec_data, self.recorded_hoc_objects)

			# Create recording vectors
			_, markers = analysis.recordTraces(
									self.recorded_hoc_objects, 
									self.trace_spec_data,
									recordStep=0.05,
									recData=self.trace_rec_data)
			self.recorded_pp_markers.extend(markers)

		# Initialize physiological conditions
		for func in self._funcs_init_physio:
			func(sim, icell)

		# Finally instantiate ephys.stimuli and ephys.recordings
		super(ContainedProtocol, self).instantiate(
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

		# Make sure stimuli are not active in next protocol if cell model reused
		# NOTE: should better be done in Stimulus objects themselves for encapsulation, but BluePyOpt built-in Stimuli don't do this
		for stim in self.stimuli:
			if hasattr(stim, 'iclamp'):
				stim.iclamp.amp = 0
				stim.iclamp.dur = 0
			elif hasattr(stim, 'seclamp'):
				for i in range(3):
					setattr(stim.seclamp, 'amp%d' % (i+1), 0)
					setattr(stim.seclamp, 'dur%d' % (i+1), 0)

		# Calls destroy() on each stimulus
		super(ContainedProtocol, self).destroy(sim=sim)