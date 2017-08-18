"""
Functions to set up STN experimental protocol with low and high
background activity


################################################################################
CONNECTIVITY observations
################################################################################


Bevan2017HandbookBG

	- The major GABAergic input to the STN arises from ipsilateral Nkx2-1- and LHx6-expressing GPe neurons,the majority of which also express parvalbumin.

	- Approximately 70-80% of GPe neurons are estimated to innervate the STN  with each STN neuron receiving input from a small fraction of GPe-STN neurons (Baufreton et al., 2009). 
	
	- Each GPe-STN axon forms a sparse, spatially distributed ter-minal field that synapses on the proximal somatodendritic regions of a small number of widely dispersed STN neurons  
	
	- The axon initial segment of STN neurons is devoid of both GABAergic inputs and GABARs (Atherton et al., 2008).



	- Unitary GABAAR-mediated GPe-STN conduc-tances are large (~5-15 nS) because each axon forms multiple synaptic inputs with an individual postsynap-tic neuron
	
	- However, sustained presynaptic GPe-STN activity is associated with profound short-term synaptic depression due to depletion of release-ready vesicles, which limits the reliability and ampli-tude of unitary synaptic transmission (Atherton et al.,2013). 
	
	- Given the high rates of activity of GPe-STN neurons in vivo (Abdi et al., 2015; Dodson et al., 2015), short-term synaptic depression is likely to greatly restrict the influence of individual GPe neurons on the STN. 
	
	- The anatomical organization and short-term plasticity properties of unitary GPe-STN inputs therefore greatly reduce the probably of detecting correlated GPe-STN activity.



	- Cortical inputs terminate on small diameter dendrites and dendritic spines of STN neurons (Bevan et al., 1995) and act through postsynaptic AMPARs and NMDARs with classical drug sensitivities,kinetics, and voltage dependencies (Chu et al., 2015).



	- The STN also receives a major serotonergic input from the dorsal raphe (Lavoie and Parent, 1990; Parent et al., 2010; Wallman et al., 2011). 
	
	- Serotonergic synapses are distributed uniformly throughout the STN and pre- dominantly form asymmetric synapses with the den- dritic shafts and spines of STN neurons (Parent et al., 2010; Wallman et al., 2011). 
	
	- Serotonin typically excites STN neurons through activation of Gq-coupled 5HT2CRs and Gs-coupled 5HT4Rs but also inhibits a subset of STN neurons through activation of Gi-cou- pled 5HT1ARs (Shen et al., 2007; Stanford et al., 2005; Xiang et al., 2005). 
	
	- Excitation is associated with a decrease in voltage-independent K+ conductance and an increase in nonselective cation conductance, whereas inhibition is mediated through an increase in K+ conductance (Shen et al., 2007; Stanford et al., 2005; Xiang et al., 2005). 
	
	- Serotonin also reduces the initial probability of both glutamatergic and GABAergic transmission through activation of presynaptic Gi-cou- pled 5HT1BRs (Shen and Johnson, 2008).



	- The parafascicular thalamic nucleus  also projects ipsilaterally and topographically to the STN (Bevan et al., 1995; Groenewegen and Berendse, 1990; Lanciego et al., 2004; Sadikot et al.,1992). 

	- In rodents, the medial, central, and lateral parts of the parafascicular thalamic nucleus project to corre-sponding zones of the STN. Parafascicular thalamicneurons that project to STN also send collaterals tothe striatum and layer VI of the cortex (Deschenes et al., 1996). 

	- Thalamic inputs also terminate on the dendritic shafts and spines of STN neurons. However,thalamic inputs target larger diameter and thus moreproximal dendrites than cortical inputs (Bevan et al.,1995). 

	- The available evidence suggests that para-fascicular inputs also excite STN neurons throughactivation of postsynaptic AMPARs and NMDARs (Mouroux and Feger, 1993).



	- Both noncholinergic and cholinergic PPN neuronsproject bilaterally to the STN but with an ipsilateral preference (Bevan and Bolam, 1995; Kita and Kita,2011). 

	- Because noncholinergic and cholinergic PPN axon terminals express high levels of glutamate theyare presumed to use it as a neurotransmitter (Bevanand Bolam, 1995; Clarke et al., 1997). 

	- PPN inputs terminate on the dendritic shafts and spines of STN neurons (Bevan and Bolam, 1995).



Galvan2004DifferentialDistributionGABAaGABAb

	- we found that the GABAA receptors are distributed evenly along synaptic specializations in the STN. 

	- Also, the density of GABAA alpha1 subunit labeling at symmetric synapses was similar on proximal and distal dendrites, as originally found in the cerebellum (Somogyi et al., 1996). 

	- Since it has been proposed that GABAergic postsynaptic responses correlate directly with the number of GABA receptors (Nusser et al., 1997), our observations suggest that the strength of pallidal GABAergic synapses remains relatively constant along the dendritic tree of STN neurons.



BaufretonBevan2008
	
	- It is estimated that STN neurons each receive approximately 300 synaptic inputs, each with a mean conductance of approximately 0.8 nS (this study).



Baufreton 2009

	- Light and electron microscopic analyses revealed that single GP axons give rise to sparsely distributed terminal clusters, many of which correspond to multiple synapses with individual STN neurons.
	
	- Application of the minimal stimulation technique in brain slices confirmed that STN neurons receive multisynaptic unitary inputs and that these inputs largely arise from different sets of GABAergic axons



################################################################################
ACTIVITY observations
################################################################################

Mallet2016Neuron89

	- GPe Proto cells fired regularly at high rates (47.3 +/- 6.1 Hz) while GPe Arky cells (projecting to Str) were more irregularly active with lower awake firing rates (8.9 +/- 1.9 Hz)

	- Both GPe Arky and Proto cells show similar population-level entrainment to LFP beta oscillations with few individual cells strongly entrained

	- A subset of GPe Proto cells show clear pauses in activity from high baseline, consistent with high-frequency discharging



Nambu2014Frontiers8

	- Average firing rate of GPe neurons decreased significantly from 65.2+/-25.8 to 41.2+/-22.5 upon MPTP treatment


"""

# Python stdlib
import re

# NEURON
import neuron
h = neuron.h

# Physiological parameters
import cellpopdata as cpd
from cellpopdata import PhysioState, Populations as Pop, NTReceptors as NTR, ParameterSource as Cit

# Stimulation protocols
from proto_common import *

# Global parameters

# PROBLEM:
#	- we have estimates for the total number of synapses
#	- we have measurements of the post-synaptic response to axonal stimulation
#	- However, this response is most likely due to multiple synapses (MULTI SYNAPSE RULE: axons make multi-synaptic contacts)
#	- so when we calibrate a synapse mechanism to match this response, our synapse mechanism emulates the effect of multiple real synapses

n_syn_stn_tot = 300		# [SETPARAM] 300 total synapses on STN
frac_ctx_syn = 3.0/4.0	# [SETPARAM] fraction of CTX/GPE of STN afferent synapses

gsyn_single = 0.8e-3	# [SETPARAM] average unitary conductance [uS])
CALC_MSR_FROM_GBAR = False

MSR_NUM_SYN = {			# [SETPARAM] average number of contacts per multi-synaptic contact
	Pop.CTX: 5,
	Pop.GPE: 5,
}

FRAC_SYN = {
	Pop.CTX: frac_ctx_syn,
	Pop.GPE: 1.0 - frac_ctx_syn,
}

def init_sim(self, protocol):
	"""
	Initialize simulator to simulate background protocol
	"""

	# Only adjust duration
	self._init_sim(dur=5000)


def make_inputs(self, connector=None):
	"""
	Make a realistic number of CTX and GPe synapses that fire
	a background firing pattern onto STN.
	"""

	if connector is None:
		cc = cpd.CellConnector(self.physio_state, self.rng)
	else:
		cc = connector

	###########################################################################
	# CTX inputs

	# Filter to select distal, smaller-diam dendritic segments
	is_ctx_target = lambda seg: seg.diam <= 1.0 # see Gillies cell diagram

	# Parameters for making connection
	syn_mech_NTRs = ('GLUsyn', [NTR.AMPA, NTR.NMDA])
	refs_con = [Cit.Chu2015]
	refs_fire = [Cit.Custom, Cit.Bergman2015RetiCh3]

	# Get connection & firing parameters
	con_par = cc.getConParams(Pop.CTX, Pop.STN, refs_con)
	fire_par = cc.getFireParams(Pop.CTX, self.physio_state, refs_fire, 
					custom_params={'rate_mean': 20.0})

	# Make CTX GLUtamergic inputs
	make_background_inputs(self, Pop.CTX, is_ctx_target, syn_mech_NTRs, fire_par, con_par, cc)

	###########################################################################
	# GPe inputs

	# Filter to select proximal, larger-diam dendritic segments
	is_gpe_target = lambda seg: seg.diam > 1.0 # see Gillies cell diagram

	# Parameters for making connection
	syn_mech_NTRs = ('GABAsyn', [NTR.GABAA, NTR.GABAB])
	refs_con = [Cit.Chu2015, Cit.Fan2012, Cit.Atherton2013]
	refs_fire = [Cit.Bergman2015RetiCh3]

	# Get connection & firing parameters
	con_par = cc.getConParams(Pop.GPE, Pop.STN, refs_con)
	fire_par = cc.getFireParams(Pop.GPE, self.physio_state, refs_fire)

	# Make GPe GABAergic inputs
	make_background_inputs(self, Pop.GPE, is_gpe_target, syn_mech_NTRs, fire_par, con_par, cc)


def rec_traces(self, protocol, traceSpecs):
	"""
	Record all traces for this protocol.
	"""
	# record synaptic traces
	rec_GABA_traces(self, protocol, traceSpecs)
	rec_GLU_traces(self, protocol, traceSpecs)

	# record membrane voltages
	rec_Vm(self, protocol, traceSpecs)

	# Record input spikes
	rec_spikes(self, protocol, traceSpecs)


def plot_traces(self, model, protocol):
	"""
	Plot all traces for this protocol
	"""

	# Plot Vm in select number of segments
	self._plot_all_Vm(model, protocol, fig_per='cell')

	# Plot rastergrams
	gpe_filter = lambda trace: re.search('AP_'+Pop.GPE.name, trace)
	self._plot_all_spikes(model, protocol, trace_filter=gpe_filter, color='r')

	ctx_filter = lambda trace: re.search('AP_'+Pop.CTX.name, trace)
	self._plot_all_spikes(model, protocol, trace_filter=ctx_filter, color='g')


def make_background_inputs(self, POP_PRE, is_target_seg, syn_mech_NTRs, fire_par, con_par, connector):
	"""
	Make synapses with background spiking from given population.

	@param refs_fire		References for firing parameters

	@param refs_con			References for connectivity parameters
	"""

	model = self.target_model
	cc = connector

	# Get max synaptic conductance
	syn_mech, syn_NTRs = syn_mech_NTRs
	gsyn_multi = max((con_par[NTR].get('gbar', 0) for NTR in syn_NTRs))

	# Calculate number of synapses
	n_syn_single = int(FRAC_SYN[POP_PRE] * n_syn_stn_tot) # number of unitary synapses for this population
	if CALC_MSR_FROM_GBAR:
		gsyn_tot = n_syn_single * gsyn_single # total parallel condutance desired [uS]
		n_syn_multi = int(gsyn_tot / gsyn_multi)
	else:
		n_syn_multi = int(n_syn_single / MSR_NUM_SYN[POP_PRE])
	
	n_syn = n_syn_multi
	logger.debug("Number of {}->STN MSR synapses = {}".format(POP_PRE.name, n_syn))

	# Get target segments: distribute synapses over dendritic trees
	dendrites = self.model_data[model]['sec_refs']['dendrites']
	dend_secrefs = sum(dendrites, [])
	target_segs = pick_random_segments(dend_secrefs, n_syn, is_target_seg, rng=self.rng)

	# Data for configuring inputs
	gid = self.model_data[model]['gid']
	n_syn_existing = self.get_num_syns(model)

	tstart = 300
	stim_rate = fire_par['rate_mean']
	pause_rate = fire_par.get('pause_rate_mean', 0)
	pause_dur = fire_par.get('pause_dur_mean', 0)
	discharge_dur = fire_par.get('discharge_dur_mean', 0)
	# TODO: set intra-burst rate (higher than mean rate), set burst dur, calculate 'number' from these two, then set controlling stim rate to pause rate

	# Make synapses
	new_inputs = {}
	for i_seg, target_seg in enumerate(target_segs):

		# Index of the new synapse
		i_syn = n_syn_existing + i_seg

		# Make RNG for spikes
		stimrand = h.Random() # see CNS2014 Dura-Bernal example or EPFL cell synapses.hoc file
		# MCellRan4: each stream should be statistically independent as long as the highindex values differ by more than the eventual length of the stream. See http://www.neuron.yale.edu/neuron/static/py_doc/programming/math/random.html?highlight=MCellRan4
		dur_max_ms, dt_ms = 10000.0, 0.025
		num_indep_repicks = dur_max_ms / dt_ms
		low_index, high_index = gid+250+self.base_seed, int(i_syn*num_indep_repicks + 100)
		logger.debug("Seeding RNG with index {}".format(high_index))
		stimrand.MCellRan4(high_index, low_index) # high_index can also be set using .seq()
		stimrand.negexp(1) # if num arrivals is poisson distributed, ISIs are negexp-distributed
		
		# make NetStim spike generator
		stimsource = h.NetStim()
		stimsource.interval = stim_rate**-1*1e3 # Interval between spikes
		stimsource.number = 1e9 # inexhaustible for our simulation
		stimsource.noise = 1.0
		stimsource.noiseFromRandom(stimrand) # Set it to use this random number generator
		stimsource.start = tstart

		if pause_rate > 0:
			assert (discharge_dur < 1/pause_rate), "Discharge duration must be smaller than inter-pause interval"
			logger.debug("Creating pausing NetStim with pause_rate={} and pause_dur={}".format(pause_rate, pause_dur))

			# Make spike generator exhaustible
			stimsource.number = discharge_dur*stim_rate # expected number of spikes in discharge duration

			# Make RNG for spike control
			ctlrand = h.Random()
			low_index, high_index = gid+100+self.base_seed, int(i_syn*num_indep_repicks + 100)
			ctlrand.MCellRan4(high_index, low_index) # high_index can also be set using .seq()
			ctlrand.negexp(1)

			# control spike generator spiking pattern
			stimctl = h.NetStim()
			# stimctl.interval = pause_rate**-1*1e3
			stimctl.interval = pause_dur*1e3 # replenish only works when spikes exhaused
			stimctl.number = 1e9
			stimctl.noise = 1.0
			stimctl.noiseFromRandom(ctlrand)
			stimctl.start = tstart

			# Connect to spike generator
			# off_nc = h.NetCon(stimctl, stimsource)
			# off_nc.weight[0] = -1 # turn off spikegen
			# off_nc.delay = 0
			
			ctl_nc = h.NetCon(stimctl, stimsource)
			ctl_nc.weight[0] = 1 # turn on spikegen (resets available spikes)
			# ctl_nc.delay = pause_dur*1e3

			extend_dictitem(new_inputs, 'com_NetCons', ctl_nc)
			# extend_dictitem(new_inputs, 'com_NetCons', off_nc)
			extend_dictitem(new_inputs, 'NetStims', stimctl)
			extend_dictitem(new_inputs, 'RNGs', ctlrand)

		# Make synapse and NetCon
		syn, nc, wvecs = cc.make_synapse((POP_PRE, Pop.STN), (stimsource, target_seg), 
							syn_mech, syn_NTRs, con_par_data=con_par)

		# Save inputs
		extend_dictitem(new_inputs, 'syn_NetCons', nc)
		extend_dictitem(new_inputs, 'synapses', syn)
		extend_dictitem(new_inputs, 'NetStims', stimsource)
		extend_dictitem(new_inputs, 'RNGs', stimrand)
		extend_dictitem(new_inputs, 'stimweightvec', wvecs)

	self.add_inputs(POP_PRE.name.lower(), model, **new_inputs)


def rec_GABA_traces(self, protocol, traceSpecs):
	"""
	Record traces at GABA synapses

	@param n_syn		number of synaptic traces to record
	"""

	n_syn = 3 # number of recorded synapses

	rec_segs = self.model_data[self.target_model]['rec_segs'][protocol]
	model = self.target_model
	
	# Add synapse and segment containing it
	nc_list = self.model_data[model]['inputs']['gpe']['syn_NetCons']
	for i_syn, nc in enumerate(nc_list):
		if i_syn > n_syn-1:
			break

		syn_tag = 'GABAsyn%i' % i_syn
		seg_tag = 'GABAseg%i' % i_syn

		# Record from synapse POINT_PROCESS and postsynaptic segment
		rec_segs[syn_tag] = nc.syn()
		rec_segs[seg_tag] = nc.syn().get_segment()

		# Record synaptic variables
		traceSpecs['gA_GABAsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'g_GABAA'}
		traceSpecs['gB_GABAsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'g_GABAB'}


def rec_GLU_traces(self, protocol, traceSpecs):
	"""
	Record traces at GLU synapses
	"""

	n_syn = 3 # number of recorded synapses

	rec_segs = self.model_data[self.target_model]['rec_segs'][protocol]
	model = self.target_model
	
	# Add synapse and segment containing it
	nc_list = self.model_data[model]['inputs']['ctx']['syn_NetCons']
	for i_syn, nc in enumerate(nc_list):
		if i_syn > n_syn-1:
			break

		syn_tag = 'GLUsyn%i' % i_syn
		seg_tag = 'GLUseg%i' % i_syn

		# Record from synapse POINT_PROCESS and postsynaptic segment
		rec_segs[syn_tag] = nc.syn()
		rec_segs[seg_tag] = nc.syn().get_segment()

		# Record synaptic variables
		traceSpecs['gA_GLUsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'g_AMPA'}
		traceSpecs['gN_GLUsyn%i' % i_syn] = {'pointp':syn_tag, 'var':'g_NMDA'}


def rec_Vm(self, protocol, traceSpecs):
	"""
	Record membrane voltages in all recorded segments
	"""
	rec_segs = self.model_data[self.target_model]['rec_segs'][protocol]
	
	for seclabel, seg in rec_segs.iteritems():
		if isinstance(seg, neuron.nrn.Segment):
			traceSpecs['V_'+seclabel] = {'sec':seclabel, 'loc':seg.x, 'var':'v'}


def rec_spikes(self, protocol, traceSpecs):
	"""
	Record input spikes delivered to synapses.
	"""
	model = self.target_model
	rec_pops = [Pop.GPE, Pop.CTX]

	for pre_pop in rec_pops:
		nc_list = self.model_data[model]['inputs'][pre_pop.name.lower()]['syn_NetCons']
		for i_syn, nc in enumerate(nc_list):

			# Add NetCon to list of recorded objects
			syn_tag = pre_pop.name + 'syn' + str(i_syn)
			self._add_recorded_obj(syn_tag, nc, protocol)

			# Specify trace
			traceSpecs['AP_'+syn_tag] = {'netcon':syn_tag}
			