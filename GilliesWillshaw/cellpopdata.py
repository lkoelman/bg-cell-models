"""
Parameters for STN connections

@author		Lucas Koelman
@date		12/07/2017
"""

from enum import Enum
import types
import numpy as np

class PhysioState(Enum):
	"""
	Physiological state of the cell
	"""
	NORMAL = 0
	PARKINSONIAN = 1
	PARK_DBS = 2


class Populations(Enum):
	"""
	Physiological state of the cell
	"""
	STN = 0
	CTX = 1
	GPE = 2


class Neurotransmitters(Enum):
	"""
	Neurotransmitters used in synaptic connections
	"""
	AMPA = 0
	NMDA = 1
	GABAA = 2
	GABAB = 3


class ParameterSource(Enum):
	Final = 0
	CommonOccurrence = 1	# Widely used or commonly accepted values
	Chu2015 = 2
	Baufreton2009 = 3
	Fan2012 = 4
	Atherton2013 = 5
	Kumaravelu2016 = 6
	

def getConnParams(target_pop, physio_state, use_sources, rng=None):
	"""
	Get parameters for afferent connections onto given population,
	in given physiological state.
	"""

	pop = Populations
	nt = Neurotransmitters
	ps = ParameterSource

	if rng is None:
		rng = np.random

	if target_pop == pop.STN:

		# Initialize parameters dict
		cp = {
			pop.CTX : {
				nt.AMPA: {},
				nt.NMDA: {},
			},
			pop.GPE : {
				nt.GABAA: {},
				nt.GABAB: {},
			},
		}

		#######################################################################
		# CTX -> STN parameters
		Erev = 0.
		Ermp = -70.

		# Default parameters
		cp[pop.CTX][nt.AMPA][ps.Final] = {
			'delay': 1.0,
			'rate': 50.0,
			'noise': 0.33,
		}

		# ---------------------------------------------------------------------
		# Parameters from Chu (2015)
		cp[pop.CTX][nt.AMPA][ps.Chu2015] = {
			'Ipeak': -275., 	# peak synaptic current (pA)
			'gbar': -275.*1e-3 / (Ermp - Erev), # gbar calculation
			'tau1': 1.0,
			'tau2': 4.0,
			'Erev': Erev,
		}
		cp[pop.CTX][nt.NMDA][ps.Chu2015] = {
			'Ipeak': -270., 	# peak synaptic current (pA)
			'gbar': -270.*1e-3 / (Ermp - Erev), # gbar calculation
			'tau1': 1.0,
			'tau2': 50.0,
			'Erev': Erev,
		}
		if physio_state == PhysioState.PARKINSONIAN:
			cp[pop.CTX][nt.AMPA][ps.Chu2015].update({
				'Ipeak': -390.,
				'gbar': -390.*1e-3 / (Ermp - Erev),	
			})
		

		#######################################################################
		# GPe -> STN parameters
		Erev = -85.
		Ermp = -70.

		# ---------------------------------------------------------------------
		# Parameters from Chu (2015)
		cp[pop.GPE][nt.GABAA][ps.Chu2015] = {
			'Ipeak': 350., 	# peak synaptic current (pA)
			'gbar': 350.*1e-3 / (Ermp - Erev), # gbar calculation
			'tau1': 2.6,
			'tau2': 5.0,
			'Erev': Erev,
		}
		if physio_state == PhysioState.PARKINSONIAN:
			cp[pop.GPE][nt.GABAA][ps.Chu2015].update({
				'Ipeak': 450.,
				'gbar': 450.*1e-3 / (Ermp - Erev),
				'tau1': 3.15,
				'tau2': 6.5,
			})


		# ---------------------------------------------------------------------
		# Parameters from Baufreton (2009)
		cp[pop.GPE][nt.GABAA][ps.Baufreton2009] = {
			'Ipeak': {
					'min': 375.,
					'max': 680.,
					'units': 'pA',
				},
			'gbar': 530.*1e-3 / (Ermp - Erev), # gbar calculation
			'tau1': 2.6,
			'tau2': 5.0,
			'Erev': Erev,
		}

		# ---------------------------------------------------------------------
		# Parameters from Fan (2012)
		cp[pop.GPE][nt.GABAA][ps.Fan2012] = {
			'gbar': {
				'mean': 7.03e-3,
				'deviation': 3.10e-3,
				'units': 'uS'
				},
			'tau1': 1.0,
			'tau2': 6.0,
			'Erev': Erev,
		}
		if physio_state == PhysioState.PARKINSONIAN:
			cp[pop.GPE][nt.GABAA][ps.Fan2012].update({
				'gbar': {
					'mean': 11.17e-3,
					'deviation': 5.41e-3,
					'units': 'uS'
					},
				'tau1': 1.0,
				'tau2': 8.0,
			})

		# Use preferred sources to update final parameters
		for p in list(Populations):
			if p not in cp:
				continue

			for n in list(Neurotransmitters):
				if n not in cp[p]:
					continue

				for src in reversed(use_sources):
					if not src in cp[p][n]:
						continue

					# Successively overwrite with params of each source
					src_params = cp[p][n][src]
					cp[p][n][ps.Final].update(src_params)

		return cp


def make_synapse(post_seg, pre_obj, cd, rng=None):
	"""
	Insert synapse POINT_PROCESS in given section.

	@param cd		dict containing electrophysiological data
					relating to the synapse/connection

	@effect			creates an Exp2Syn with an incoming NetCon with
					weight equal to maximum synaptic conductance
	"""
	if rng is None
		rng = np.random

	# Make synapse
	syn = h.Exp2Syn(post_seg)
	syn_params = {
		'e': 'Erev',
		'tau1': 'tau1',
		'tau2': 'tau2'
	}

	# Set synapse parameters
	for syn_attr, conn_par in syn_params.iteritems():

		par_data = cd[conn_par]
		par_val = None

		if isinstance(par_data, (float, int)):
			par_val = par_data

		elif isinstance(par_data, types.FunctionType):
			par_val = par_data()

		elif isinstance(par_data, dict):

			if ('min' in par_data and 'max' in par_data):
				lower = par_data['min']
				upper = par_data['max']
				par_val = lower + rng.rand()*(upper-lower)

			elif ('mean' in par_data and 'deviation' in par_data):
				lower = par_data['mean'] - par_data['deviation']
				upper = par_data['mean'] + par_data['deviation']
				par_val = lower + rng.rand()*(upper-lower)

			elif ('mean' in par_data and 'stddev' in par_data):
				par_val = rng.normal(par_data['mean'], par_data['stddev'])

			else:
				raise ValueError('Could not infer distribution from parameters in {}'.format(par_data))

		# Set the attribute
		setattr(syn, syn_attr, par_val)

	# TODO: let user specify waveform in [0,1], then make NetStim with given noise, rate, waveform
	if src_is_event_generating_point_process:
		nc = h.NetCon(pre_obj, syn)

	elif src_is_waveform:
		stimsource = h.NetStim() # Create a NetStim
		stimsource.interval = stim_rate**-1*1e3 # Interval between spikes
		stimsource.number = 1e9 # max number of spikes
		stimsource.noise = 0.33 # Fractional noise in timing

		nc = h.NetCon(stimsource, syn)
		waveform_Vector.play(nc._ref_weight[0], stimtimevec)

	else:
		# physiological cells need threshold detection to generate events
		nc = hoc.NetCon(pre_cell(0.5)._ref_v, receiver, sec=pre_cell)
		nc.threshold =  p_thresh

	nc.delay = p_delay
	nc.weight[wid] = p_sbar

	return nc, syn