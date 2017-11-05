"""
Python code to run and plot Gillies & Willshaw (2006) model

@author Lucas Koelman
@date	21-10-2016

NOTE: this script relies on commenting out the 'graphics=1' line in sample.hoc
"""

import neuron
h = neuron.h
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

import matplotlib.pyplot as plt

from common.treeutils import ExtSecRef, getsecref
from reducemodel import redutils

# Load NEURON mechanisms
import os.path
scriptdir, scriptfile = os.path.split(__file__)
NRN_MECH_PATH = os.path.normpath(os.path.join(scriptdir, 'nrn_mechs'))
neuron.load_mechanisms(NRN_MECH_PATH)

# Global variables
gillies_gdict = {
	'STh':	['gpas'], 								# passive/leak channel
	'Na':	['gna'], 'NaL':['gna'],					# Na channels
	'KDR':	['gk'], 'Kv31':['gk'], 'sKCa':['gk'],	# K channels
	'Ih':	['gk'], 								# nonspecific channels
	'CaT':	['gcaT'], 'HVA':['gcaL', 'gcaN'], 		# Ca channels
	'Cacum': [],									# No channels
}
gillies_mechs = list(gillies_gdict.keys()) # all mechanisms
gillies_glist = [gname+'_'+mech for mech,chans in gillies_gdict.iteritems() for gname in chans]
gleak_name = 'gpas_STh'
active_gbar_names = [gname for gname in gillies_glist if gname != gleak_name]


def stn_cell_gillies():
	"""
	Initialize Gillies & Willshaw cell model
	"""
	if not hasattr(h, 'SThcell'):
		h.xopen("createcell.hoc")
	else:
		print("Gillies STN cell already exists. Cannot create more than one instance.")
	
	soma = h.SThcell[0].soma
	dends = h.SThcell[0].dend0, h.SThcell[0].dend1
	stims = h.stim1, h.stim2, h.stim3

	return soma, dends, stims

def get_stn_refs():
	"""
	Make SectionRef for each section and assign identifiers
	"""
	if not hasattr(h, 'SThcell'):
		stn_cell_gillies()
	
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendLrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0] # 0 is left tree
	dendRrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1] # 1 is right tree
	allsecrefs = [somaref] + dendLrefs + dendRrefs
	
	for noderef in allsecrefs:
		
		# Assign indices in /sth-data/treeX-nom.dat
		if noderef in dendLrefs:
			noderef.tree_index = 0
			noderef.table_index = dendLrefs.index(noderef) + 1
		
		elif noderef in dendRrefs:
			noderef.tree_index = 1
			noderef.table_index = dendRrefs.index(noderef) + 1
		
		elif noderef is somaref:
			noderef.tree_index = -1
			noderef.table_index = 0

		# Assign a unique GID based on table and tree index
		noderef.gid = min(0,noderef.tree_index)*100 + noderef.table_index

	return somaref, dendLrefs, dendRrefs


def get_soma_refs(all_refs):
	"""
	Return SectionRef to soma sections
	"""
	return [ref for ref in all_refs if ref.sec.same(h.SThcell[0].soma)]


def get_each_dend_refs(all_refs):
	"""
	Get one list of SectionRef for each dendrite.
	"""
	dend0 = [getsecref(sec, all_refs) for sec in h.SThcell[0].dend0]
	dend1 = [getsecref(sec, all_refs) for sec in h.SThcell[0].dend0]
	return dend0, dend1


def get_all_dend_refs(all_refs):
	"""
	Return list of SectionRef to unique dendritic sections.
	"""
	dend0, dend1 = get_each_dend_refs(all_refs)
	return dend0 + dend1


def make_passive(sec_refs, save_gbar=True):
	"""
	Make given sections passive by setting active conductances to zero.

	@param	sec_refs	list(SectionRef) of sections to make passive

	@param	save_gbar	if True, current conductance values will be savec on the
						SectionRef object in a dict named 'initial_params'
	"""
	for ref in sec_refs:

		# Store original conductances
		redutils.store_seg_props(ref, gillies_gdict, attr_name='initial_params')

		# Set active conductances to zero
		for seg in ref.sec:
			for gbar in active_gbar_names:
				setattr(seg, gbar, 0.0)

	# NOTE: reset as follows:
	# for ref in sec_refs:
	#     # Restore parameter dict stored on SectionRef
	#     redutils.set_range_props(ref, ref.initial_params)


def reset_channel_gbar():
	"""
	Reset all channel conductances to initial state.

	NOTE: initialization copied from sample.hoc/createcell.hoc
	"""
	# Soma
	h.SThcells[0].soma.gna_Na = h.default_gNa_soma
	h.SThcells[0].soma.gna_NaL = h.default_gNaL_soma

	# Dendrites
	h.cset(0,"gk_KDR","")
	h.cset(0,"gk_Kv31","")
	h.cset(0,"gk_Ih","")
	h.cset(0,"gk_sKCa","")
	h.cset(0,"gcaT_CaT","")
	h.cset(0,"gcaN_HVA","")
	h.cset(0,"gcaL_HVA","")


def setionstyles_gillies(sec):
	"""
	Set ion styles to work correctly with membrane mechanisms
	"""
	sec.push()
	h.ion_style("na_ion",1,2,1,0,1)
	h.ion_style("k_ion",1,2,1,0,1)
	h.ion_style("ca_ion",3,2,1,1,1)
	h.pop_section()


def set_aCSF(req):
	"""
	Set global initial ion concentrations (artificial CSF properties)

	This is a Python version of the Hoc function set_aCSF()

	@param req		int: identifier

					0 = NEURON defaults

					3 = Beurrier et al (1999)

					4 = Bevan & Wilson (1999)

					5 = equilibrium concentrations at 35 degrees celsius

	NOTE: only cai is actually changed during the simulation
	"""

	if req == 3: # Beurrier et al (1999)
		h.nai0_na_ion = 15
		h.nao0_na_ion = 150

		h.ki0_k_ion = 140
		h.ko0_k_ion = 3.6

		h.cai0_ca_ion = 1e-04
		h.cao0_ca_ion = 2.4

		h('cli0_cl_ion = 4') # self-declared Hoc var
		h('clo0_cl_ion = 135') # self-declared Hoc var

	elif req == 4: # Bevan & Wilson (1999)
		h.nai0_na_ion = 15
		h.nao0_na_ion = 128.5

		h.ki0_k_ion = 140
		h.ko0_k_ion = 2.5

		h.cai0_ca_ion = 1e-04
		h.cao0_ca_ion = 2.0

		h('cli0_cl_ion = 4')
		h('clo0_cl_ion = 132.5')

	elif req == 0: # NEURON's defaults
		h.nai0_na_ion = 10
		h.nao0_na_ion = 140

		h.ki0_k_ion = 54
		h.ko0_k_ion = 2.5

		h.cai0_ca_ion = 5e-05
		h.cao0_ca_ion = 2

		h('cli0_cl_ion = 0')
		h('clo0_cl_ion = 0')

	elif req == 5: # equilibrium concentrations (average) at 35 degree celsius

		h.nai0_na_ion = 15
		h.nao0_na_ion = 128.5

		h.ki0_k_ion = 140
		h.ko0_k_ion = 2.5

		h.cai0_ca_ion = 0.04534908688919702 # min:0.019593383952621085 / max: 0.072908152365581
		h.cao0_ca_ion = 2.0

		h('cli0_cl_ion = 4')
		h('clo0_cl_ion = 132.5')


def applyApamin(soma, dends):
	"""
	Apply apamin (reduce sKCa conductance)

	@param soma		soma Section

	@param dends	list of dendritic Sections objects
	
	NOTE: in paper they say reduce by 90 percent but in code
	they set everything to 0 except in soma where they divide
	by factor 10
	"""
	soma(0.5).__setattr__('gk_sKCa', 0.0000068)
	for sec in dends:
		for iseg in range(1, sec.nseg+1):
			xnode = (2.*iseg-1.)/(2.*sec.nseg) # arclength of current node (segment midpoint)
			sec(xnode).__setattr__('gk_sKCa', 0.0)


def stn_init_physiology():
	"""
	Initialize STN cell in biologically plausible physiological state.
	"""
	h.celsius = 35
	h.v_init = -68.0
	h.set_aCSF(4)
	h.init()


################################################################################
# Simulations
################################################################################

def runsim_paper(plotting='NEURON'):
	"""
	Run original simulation provided with model files
	"""

	if plotting=='NEURON':
		h('graphics = 1') # turn on plotting using NEURON
	else:
		h('graphics = 0') # turn off plotting using NEURON
	
	# Execute hoc file containing simulation
	h.xopen('sample.hoc')

	if plotting=='mpl' or plotting=='matplotlib':

		# Regular firing plots
		fig = plt.figure()
		plt.suptitle("Regular firing")

		plt.subplot(3,1,1)
		plt.plot(h.recapt.as_numpy(), h.recapv.as_numpy())
		plt.xlabel("Action Potential (30 degC)")

		plt.subplot(3,1,2)
		plt.plot(h.recsp1t.as_numpy(), h.recsp1v.as_numpy())
		plt.xlabel("Rest firing at 25 degC")

		plt.subplot(3,1,3)
		plt.plot(h.recsp2t.as_numpy(), h.recsp2v.as_numpy())
		plt.xlabel("Rest firing at 37 degC")

		# Burst firing plots
		plt.figure()
		plt.suptitle("Bursting")

		plt.subplot(4,1,1)
		plt.plot(h.recrbt.as_numpy(), h.recrbv.as_numpy())
		plt.xlabel("Rebound burst (at 35 degC)")

		plt.subplot(4,1,2)
		plt.plot(h.recsrt.as_numpy(), h.recsrv.as_numpy())
		plt.xlabel("Slow rhythmic bursting (Apamin, 37 degC)")

		plt.subplot(4,1,3)
		plt.plot(h.recfrt.as_numpy(), h.recfrv.as_numpy())
		plt.xlabel("Fast rhythmic bursting (Apamin, 37 degC, CaL-10%)")

		plt.subplot(4,1,4)
		plt.plot(h.recmrt.as_numpy(), h.recmrv.as_numpy())
		plt.xlabel("Mixed rhythmic bursting (Apamin, 37 degC, CaL+10%)")

		# if htmlplotting:
		# 	plugins.clear(fig)
		# 	plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.MousePosition())
		# 	mpld3.show()
		plt.show()


def runtest_actionpotential():
	"""
	Run AP experiment from original Hoc file
	"""

	print("*** Action potential form\n")\

	# Set up recording
	recapt = h.Vector()
	recapv = h.Vector()
	recapt.record(h._ref_t)
	recapv.record(h.SThcell[0].soma(0.5)._ref_v)

	# Simulate
	h.celsius = 30
	h.set_aCSF(3) # this sets initial concentrations via global vars
	h.tstop = 500
	h.dt = 0.025
	h.init()
	h.run()

	fig = plt.figure()
	plt.plot(recapt.as_numpy(), recapv.as_numpy())
	plt.xlabel("Action Potential (30 degC)")
	plt.show(block=False)


def runtest_restfiring():
	"""
	Run rest firing experiment from original Hoc file
	"""

	print("*** Resting firing rate (at 25 & 37 degC) \n")\

	# Set up recording
	recsp1t = h.Vector()
	recsp1v = h.Vector()
	recsp1t.record(h._ref_t)
	recsp1v.record(h.SThcell[0].soma(0.5)._ref_v)

	# Simulate
	h.celsius = 25
	h.set_aCSF(4)
	h.tstop = 2100
	h.dt=0.025
	h.init()
	h.run()

	# Set up recording
	recsp2t = h.Vector()
	recsp2v = h.Vector()
	recsp2t.record(h._ref_t)
	recsp2v.record(h.SThcell[0].soma(0.5)._ref_v)

	# Simulate
	h.celsius = 37
	h.tstop = 2100
	h.dt=0.025
	h.init()
	h.run()

	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(recsp1t.as_numpy(), recsp1v.as_numpy())
	plt.xlabel("Rest firing at 25 degC")
	plt.subplot(2,1,2)
	plt.plot(recsp2t.as_numpy(), recsp2v.as_numpy())
	plt.xlabel("Rest firing at 37 degC")
	plt.show(block=False)


def runtest_reboundburst():
	"""
	Run rebound burst experiment from original Hoc file
	"""

	print("*** Rebound burst (at 35 degC) \n")

	# Set up recording
	recrbt = h.Vector()
	recrbv = h.Vector()
	recrbt.record(h._ref_t)
	recrbv.record(h.SThcell[0].soma(0.5)._ref_v)

	recicat = h.Vector()
	recicat.record(h.SThcell[0].soma(0.5).CaT._ref_iCaT)
	reccai = h.Vector()
	reccai.record(h.SThcell[0].soma(0.5)._ref_cai)

	# Simulate
	h.celsius = 35

	h.stim1.delay = 0
	h.stim1.dur = 1000
	h.stim1.amp = 0.0

	h.stim2.delay = 1000
	h.stim2.dur = 500
	h.stim2.amp = -0.25

	h.stim3.delay = 1500
	h.stim3.dur = 1000
	h.stim3.amp = 0.0

	h.set_aCSF(4)
	h.tstop = 2500
	h.dt=0.025
	h.init()
	h.run()

	# Plot
	plt.figure()
	plt.plot(recrbt.as_numpy(), recrbv.as_numpy())
	plt.xlabel("Rebound (35 degC)")

	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(recrbt.as_numpy(), recicat.as_numpy(), label='ICaT')
	plt.legend()
	plt.subplot(2,1,2)
	plt.plot(recrbt.as_numpy(), reccai.as_numpy(), label="[Ca]_i")
	plt.legend()

	plt.show(block=False)


def runtest_slowbursting():
	"""
	Run slow rhytmic bursting experiment from original Hoc file
	"""

	print("*** Slow rhythmic bursting (at 37 degC) \n")

	# Set up recording
	recsrt = h.Vector()
	recsrv = h.Vector()
	recsrt.record(h._ref_t)
	recsrv.record(h.SThcell[0].soma(0.5)._ref_v)

	# Simulate
	h.celsius = 37

	h.stim1.delay = 0
	h.stim1.dur = 40000
	h.stim1.amp = -0.25

	h.stim2.delay = 0
	h.stim2.dur = 0
	h.stim2.amp = 0.0

	h.stim3.delay = 0
	h.stim3.dur = 0
	h.stim3.amp = 0.0 

	h.set_aCSF(4)
	h.tstop = 8000
	h.applyApamin()
	h.dt=0.025
	h.init()
	h.run()
	h.washApamin()

	plt.figure()
	plt.plot(recsrt.as_numpy(), recsrv.as_numpy())
	plt.xlabel("Slow rhythmic bursting (Apamin, 37 degC)")
	plt.show(block=False)


def runtest_fastbursting():
	"""
	Run fast rhythmic bursting experiment from original Hoc file
	"""

	print("*** Fast rhythmic bursting (at 37 degC) \n")

	# Set up recording
	recfrt = h.Vector()
	recfrv = h.Vector()
	recfrt.record(h._ref_t)
	recfrv.record(h.SThcell[0].soma(0.5)._ref_v)

	# Simulate
	h.celsius = 37

	h.stim1.delay = 0
	h.stim1.dur = 40000
	h.stim1.amp = -0.35

	h.set_aCSF(4)
	h.tstop = 4000
	h.cset(0,"gcaL_HVA","-dl0.9") # 10% decrease in dendritic linear CaL (see Figure 8A)
	h.applyApamin()
	h.dt=0.025
	h.init()
	h.run()
	h.washApamin()

	plt.figure()
	plt.plot(recfrt.as_numpy(), recfrv.as_numpy())
	plt.xlabel("Fast rhythmic bursting (Apamin, 37 degC, CaL-10%)")
	plt.show(block=False)


def runtest_mixedbursting():
	"""
	Run mixed bursting experiment from original Hoc file
	"""

	print("*** Mixed rhythmic bursting (at 37 degC) \n")

	# Set up recording
	recmrt = h.Vector()
	recmrv = h.Vector()
	recmrt.record(h._ref_t)
	recmrv.record(h.SThcell[0].soma(0.5)._ref_v)

	# Simulate
	h.celsius = 37

	h.stim1.delay = 0
	h.stim1.dur = 40000
	h.stim1.amp = -0.32

	h.set_aCSF(4)
	h.tstop = 8000
	h.cset(0,"gcaL_HVA","-dl1.1") # 10% increase in dendritic linear CaL (see Figure 8A,B)
	h.applyApamin()
	h.dt=0.025
	h.init()
	h.run()
	h.washApamin()

	plt.figure()
	plt.plot(recmrt.as_numpy(), recmrv.as_numpy())
	plt.xlabel("Mixed rhythmic bursting (Apamin, 37 degC, CaL+10%)")
	plt.show(block=False)


def runalltests():
	"""
	Run all experiments from Hoc file
	"""
	stn_cell_gillies()
	runtest_actionpotential()
	runtest_restfiring()
	runtest_reboundburst()
	runtest_slowbursting()
	runtest_fastbursting()
	runtest_mixedbursting()


def runtest_multithreaded(testfun, nthread):
	"""
	Run a test procol in multithreaded mode
	"""
	# make cell
	stn_cell_gillies()

	# enable multithreaded execution
	h.cvode_active(0)
	h.load_file('parcom.hoc')
	pct = h.ParallelComputeTool[0]
	pct.nthread(4)
	pct.multisplit(1)
	pct.busywait(1)

	# Do test
	t0 = h.startsw()
	testfun()
	t1 = h.startsw(); h.stopsw()
	print("Elapsed time: {} ms".format(t1-t0))


def runtest_singlethreaded(testfun, use_tables=True):
	"""
	Run a test protocol in single-threaded mode
	"""
	# make cell
	stn_cell_gillies()

	# Disable tables
	if not use_tables:
		for at in dir(h):
			if at.startswith('usetable_'):
				setattr(h, at, 0)
				print("Disabled TABLE {}".format(at))

	# Do test
	t0 = h.startsw()
	testfun()
	t1 = h.startsw(); h.stopsw()
	print("Elapsed time: {} ms".format(t1-t0))

if __name__ == '__main__':
	stn_cell_gillies()
	# runtest_reboundburst()

	# runtest_singlethreaded(runtest_reboundburst)
	# runtest_multithreaded(runtest_reboundburst, 6)