"""
Automated optimization of a reduced STN cell with respect to 
the full Gillies & Willshaw model.

@author Lucas Koelman
@date	08-02-2017
"""

import neuron
from neuron import h

import re

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

class Simulation(object):


class STNCellController(object):
	""" 
	This class maps candidate parameter sets to raw data
	by setting up the model and running a simulation
	"""

	def __init__(self, reduced_model_path):
		""" Make new Controller that builds and runs model starting from
			the given base model.
		"""
		self.reduced_model_path = reduced_model_path
		self.clusters = None

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
            t,v = self.run_individual(cand_params)
            traces.append([t,v])

        return traces

    def run_candidate(self, cand_params, show=False):
    	"""
    	Run simulation for an individual candidate.
    	"""
    	# Load the reduced model
    	if self.clusters is None:
    		self.clusters = bush.load_clusters(self.reduced_model_path)
    	eq_secs = bush.rebuild_sections(self.clusters)

    	# Adapt model according to candidate parameters
    	for par_name, par_value in cand_params.iteritems():
    		if par_name == 'gNaL_factor':
    			# Adjust gbar for NaL current
				for sec in eq_secs:
					for seg in sec:
						seg.gna_NaL = seg.gna_NaL * par_value

			elif par_name == 'RC_factor':
				# Adjust RC constant (divide equally over Cm & Rm)
				Rm_factor = math.sqrt(par_value)
				Cm_factor = math.sqrt(par_value)
				for sec in eq_secs:
					for seg in sec:
						seg.gpas_STh = seg.gpas_STh / Rm_factor # multiply Rm is divide gpas
						seg.cm = seg.cm * Cm_factor

			elif par_name in ['area_scale_soma', 'area_scale_trunk',
								'area_scale_smooth', 'area_scale_spiny']:
				# Get sections that are soma/trunk/smooth/spiny
				match = re.search(r'[a-zA-Z]+$', par_name) 
				clu_prefix = match.group()
				secfilter = lambda sec: sec.name().startswith(clu_prefix)
				secs = [sec for sec in eq_secs if secfilter(sec)]

				# Scale Cm and all gbar in the section by given factor
				for sec in secs:
					for seg in sec:
						seg.cm = seg.cm * par_value
						for gname in glist:
							setattr(seg, gname, getattr(seg, gname)*par_value)

			else:
				raise Exception("Unrecognized parameter '{}' with value <{}>".format(
								par_name, par_value))

		# Run the three simulations (spontaneous firing, rebound burst, plateau potential)