"""
Functions for setting up STN experimental protocols.
"""

from enum import Enum, unique
import re

import numpy as np
import neuron
h = neuron.h

from common import analysis, logutils
logger = logutils.getBasicLogger(name='stn_protos')

@unique
class StimProtocol(Enum):
	"""
	Synaptic stimulus sets or electrode stimulation protocols
	to administer to STN cell
	"""
	SPONTANEOUS = 0				# spontaneous firing (no inputs)
	CLAMP_PLATEAU = 1			# plateau potential (Gillies 2006, Fig. 10C-D)
	CLAMP_REBOUND = 2			# rebound burst (Gillies 2006, Fig. 3-4)
	SYN_BACKGROUND_HIGH = 3		# synaptic bombardment, high background activity
	SYN_BACKGROUND_LOW = 4		# synaptic bombardment, low background activity
	SYN_PARK_PATTERNED = 5		# pathological input, strong patterned cortical input with strong GPi input in antiphase
	SINGLE_SYN_GABA = 6
	SINGLE_SYN_GLU = 7
	MIN_SYN_BURST = 8			# burst using minimal combination of GLU + GABA synapses
	PASSIVE_SYN = 10			# propagation of EPSP/IPSP in passive cell


ClampProtocols = (StimProtocol.CLAMP_REBOUND, StimProtocol.CLAMP_PLATEAU)

SynapticProtocols = tuple(proto for proto in list(StimProtocol) if (
							(proto not in ClampProtocols) and
							(proto != StimProtocol.SPONTANEOUS)))

@unique
class EvaluationStep(Enum):
	"""
	Steps in evaluation of cell model
	"""
	INIT_SIMULATION = 0
	MAKE_INPUTS = 1
	RECORD_TRACES = 2
	PLOT_TRACES = 3

# Protocol-specific functions registered for each evaluation step
EVALUATION_FUNCS = dict(((proto, {}) for proto in list(StimProtocol)))


def register_step(step, protocol):
	"""
	Decorator factory to register a function implementing an evaluation step for given protocol.

	@note   since it takes arguments, it is a decorator factory rather than a decorator
			and should return the actual decorator function
	"""
	
	def decorate_step(step_func):
		# don't make wrapper function, only register it
		step_func.protocol = protocol
		step_func.evaluation_step = step
		EVALUATION_FUNCS[protocol][step] = step_func
		return step_func

	return decorate_step


def pick_random_segments(sections, n_segs, elig_func, rng=None, refs=True):
	"""
	Pick random segments with spatially uniform distribution.

	@param	refs	True if sections argument are SectionRef instead of Section
	"""
	# Get random number generator
	if rng is None:
		rng = np.random

	# Gather segments that are eligible.
	if refs:
		elig_segs = [seg for ref in sections for seg in ref.sec if elig_func(seg)]
	else:
		elig_segs = [seg for sec in sections for seg in sec if elig_func(seg)]
	
	logger.debug("Found {} eligible candidate segments".format(len(elig_segs)))

	# Sample segments
	#   Note that nseg/L is not necessarily uniform so that randomly picking
	#   segments will not lead to a uniform spatial distribution of synapses.
	target_segs = [] # target segments, including their x-location
	Ltotal = sum((seg.sec.L/seg.sec.nseg for seg in elig_segs)) # summed length of all found segments
	for i in xrange(n_segs):
		sample = rng.random_sample() # in [0,1)
		# Pick segment at random fraction of combined length of Sections
		Ltraversed = 0.0
		for seg in elig_segs:
			Lseg = seg.sec.L/seg.sec.nseg
			if Ltraversed <= (sample*Ltotal) < Ltraversed+Lseg:
				# Find x on Section by interpolation
				percent_seg = (sample*Ltotal - Ltraversed)/Lseg
				xwidth = 1.0/seg.sec.nseg
				x0_seg = seg.x - 0.5*xwidth
				x_on_sec = x0_seg + percent_seg*xwidth
				target_segs.append(seg.sec(x_on_sec))
			Ltraversed += Lseg

	logger.debug("Picked {} target segments.".format(len(target_segs)))

	return target_segs

def extend_dictitem(d, key, val, append=True):
	"""
	Append value to the item in d[key]
	"""
	item = d.setdefault(key, [])
	if append:
		item.append(val) # append to list
	else:
		item.extend(val) # if val is list, join lists


def index_or_name(trace_data):
	"""
	Ordering function: gets trace name or index.
	"""
	name = trace_data[0]
	m = re.search(r'\[([\d]+)\]', name) # get index [i]
	key = int(m.groups()[0]) if m else name
	return key


def plot_all_spikes(trace_vectors, **kwargs):
	"""
	Plot all recorded spikes.

	@pre		all traces containing spike times have been tagged
				with prefix 'AP_'. If not, provide a custom filter
				function in param 'trace_filter'

	@param trace_filter		filter function for matching spike traces.

	@param kwargs			can be used to pass any arguments of analysis.plotRaster()
	"""

	# Get duration
	if 'timeRange' not in kwargs:
		tvec = analysis.to_numpy(trace_vectors['t_global'])
		trace_dur = tvec[tvec.size-1] # last time point
		kwargs['timeRange'] = (0, trace_dur) # set interval if not given

	# Get spikes
	trace_filter = kwargs.pop('trace_filter', None)
	if trace_filter is None:
		trace_filter = lambda t: t.startswith('AP_')
	orderfun = index_or_name
	
	spikeData = analysis.match_traces(trace_vectors, trace_filter, orderfun) # OrderedDict

	
	# Plot spikes in rastergram
	fig, ax = analysis.plotRaster(spikeData, **kwargs)
	return fig, ax


def report_spikes(trace_vectors, **kwargs):
	"""
	Plot all recorded spikes.

	@pre		all traces containing spike times have been tagged
				with prefix 'AP_'. If not, provide a custom filter
				function in param 'trace_filter'

	@param trace_filter		filter function for matching spike traces.

	@param kwargs			can be used to pass any arguments of analysis.plotRaster()
	"""

	# Get spikes
	trace_filter = kwargs.pop('trace_filter', None)
	if trace_filter is None:
		trace_filter = lambda t: t.startswith('AP_')
	orderfun = index_or_name
	
	spikeData = analysis.match_traces(trace_vectors, trace_filter, orderfun) # OrderedDict

	# Count spikes and sum times
	spiketrain_lengths_timesums = [(len(train), sum(train)) for train in spikeData.values()]
	train_lengths, train_sums = zip(*spiketrain_lengths_timesums)
	
	# Report them
	length_report = "Recorded spike train lengths are: {}".format(train_lengths)
	timesum_report = "Sum of spike times for each train is: {}".format(train_sums)
	logger.debug(length_report)
	logger.debug(timesum_report)

	return None, None # same num out as plot functions


def plot_all_Vm(trace_vectors, **kwargs):
	"""
	Plot all membrane voltages.
	"""
	fig_per = kwargs.get('fig_per', None)
	recordStep = kwargs.get('recordStep', None)

	# Plot data
	filterfun = lambda t: t.startswith('V_')
	orderfun = index_or_name
	recV = analysis.match_traces(trace_vectors, filterfun, orderfun)

	if fig_per == 'cell' and len(recV) > 5:
		rot = 0
	else:
		rot = 90

	figs = analysis.plotTraces(recV, recordStep, yRange=(-80,40), 
								traceSharex=True, oneFigPer=fig_per, 
								labelRotation=rot)
	return figs


def plot_GABA_traces(trace_vectors, **kwargs):
	"""
	Plot GABA synapse traces.
	"""
	recordStep = kwargs.get('recordStep', None)

	# Plot data
	syn_traces = analysis.match_traces(trace_vectors, lambda t: re.search(r'GABAsyn', t))
	n, KD = h.n_GABAsyn, h.KD_GABAsyn # parameters of kinetic scheme
	hillfac = lambda x: x**n/(x**n + KD)
	analysis.plotTraces(syn_traces, recordStep, 
				traceSharex	= True,
				title		= 'Synaptic variables',
				traceXforms	= {'Hill_syn': hillfac},
				oneFigPer	= kwargs.get('fig_per', None))


def plot_GLU_traces(trace_vectors, **kwargs):
	"""
	Plot GABA synapse traces.
	"""
	recordStep = kwargs.get('recordStep', None)

	# Plot synaptic variables
	syn_traces = analysis.match_traces(trace_vectors, lambda t: re.search(r'GLUsyn', t))
	analysis.plotTraces(syn_traces, recordStep,
				traceSharex	= True,
				title		= 'Synaptic variables',
				oneFigPer	= kwargs.get('fig_per', None))