"""
Functions for setting up STN experimental protocols.
"""

from enum import Enum, unique
import numpy as np

logger = None

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


def pick_random_segments(secrefs, n_segs, elig_func, rng=None):
	"""
	Pick random segments with spatially uniform distribution.
	"""
	# Get random number generator
	if rng is None:
		rng = np.random

	# Gather segments that are eligible.
	elig_segs = [seg for ref in secrefs for seg in ref.sec if elig_func(seg)]
	logger.debug("Found {} eligible target segments".format(len(elig_segs)))

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