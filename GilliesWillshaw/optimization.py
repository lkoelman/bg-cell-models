"""
Automated optimization of a reduced STN cell with respect to 
the full Gillies & Willshaw model.

@author Lucas Koelman
@date	08-02-2017
"""
import re
import collections

import numpy as np
import neuron
from neuron import h

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

class Simulation(object):
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
		traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}

		# Make trace record Vectors
		recordStep = 0.05
		self.recData = analysis.recordTraces(secs, traceSpecs, recordStep)

		# Make time record Vector
		self.rec_t = h.Vector()
        self.rec_t.record(h._ref_t)

	def get_results():
		""" Get time and voltage trace recorded during last simulation """
		v_rec = np.array(self.recData['V_soma'])
		t_rec = np.array(self.rec_t)
		return t_rec, v_rec

	def simulate_protocol(protocol):
		""" Simulate the given experimental protocol """
		if protocol == Protocol.SPONTANEOUS:
			# Spontaneous firing, no stimulation
			h.dt = self.dt
			h.celsius = 37 # different temp from paper (fig 3B: 25degC, fig. 3C: 35degC)
			h.v_init = -60 # paper simulations use default v_init
			self.set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)
			
			self.set_recording()
			h.init() # calls finitialize() and fcurrent()
			neuron.run(2000)

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

			# Run simulation
			self.set_recording()
			h.init() # calls finitialize() and fcurrent()
			neuron.run(2000)

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

			# Run simulation
			self.set_recording()
			h.init() # calls finitialize() and fcurrent()
			neuron.run(2000)
		else:
			raise Exception("Unrecognized protocol {0}".format(protocol))


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
            t,v = self.run_candidate(cand_params)
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
		somasec = next(sec for sec in eq_secs if sec.name().startswith('soma'))
		sim = Simulation(somasec)
		target_protocol = Protocol.REBOUND
		t_trace, v_trace = sim.simulate_protocol(target_protocol)

		# protocols = [Protocol.SPONTANEOUS, Protocol.REBOUND, Protocol.PLATEAU]
		# for proto in protocols:
		# 	sim.simulate_protocol(proto)
		# 	t_trace, v_trace = sim.get_results()
		return t_trace, v_trace

def optimization_routine():
	pass

if __name__ == '__main__':
	optimization_routine()
