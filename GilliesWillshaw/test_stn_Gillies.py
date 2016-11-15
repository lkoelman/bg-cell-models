"""
Run experiments using the Gillies & Willshaw (2006) STN neuron model

The experiments are designed to discover which currents are responsible
for the different features of STN cell dynamics

@author Lucas Koelman
@date	28-10-2016
@note	must be run from script directory or .hoc files not found

"""

import numpy as np
import matplotlib.pyplot as plt

import neuron
from neuron import h
nrn = neuron
hoc = h

import sys
import os.path
scriptdir, scriptfile = os.path.split(__file__)
modulesbase = os.path.normpath(os.path.join(scriptdir, '..'))
sys.path.append(modulesbase)
from common import analysis
import collections

import reducemodel

# Load NEURON mechanisms
# add this line to nrn/lib/python/neuron/__init__.py/load_mechanisms()
# from sys import platform as osplatform
# if osplatform == 'win32':
# 	lib_path = os.path.join(path, 'nrnmech.dll')
NRN_MECH_PATH = os.path.normpath(os.path.join(scriptdir, 'nrn_mechs'))
neuron.load_mechanisms(NRN_MECH_PATH)
	

# Load NEURON function libraries
hoc.load_file("stdlib.hoc") # Load the standard library
hoc.load_file("stdrun.hoc") # Load the standard run library

# Global variables
soma = None
gillies_mechs = ['STh', 'Na', 'NaL', 'KDR', 'Kv31', 'Ih', 'Cacum', 'sKCa', 'CaT', 'HVA'] # all mechanisms
gillies_gdict = {'STh': ['gpas'], # passive/leak channel
				'Na': ['gna'], 'NaL':['gna'], # Na channels
				'KDR':['gk'], 'Kv31':['gk'], 'sKCa':['gk'], # K channels
				'Ih':['gk'], # nonspecific channels
				'CaT':['gcaT'], 'HVA':['gcaL', 'gcaN']} # Ca channels
gillies_glist = []
for mechname, mechgs in gillies_gdict.iteritems():
	for gname in mechgs: gillies_glist.append(gname+'_'+mechname)

def lambda_f(f, diam, Ra, cm):
	""" Compute electrotonic length (taken from stdlib.hoc) """
	return 1e5*np.sqrt(diam/(4*np.pi*f*Ra*cm))

def stn_cell_gillies():
	""" Initialize cell/setup for running individual tests on cell """
	# Create cell and three IClamp in soma
	# make set-up
	global soma
	if soma is None:
		h.xopen("createcell.hoc")
		soma = h.SThcell[0].soma
	dends = h.SThcell[0].dend0, h.SThcell[0].dend1
	stims = h.stim1, h.stim2, h.stim3
	return soma, dends, stims

def stn_cell_threesec(resurgent=False):
	""" Initialie reduced Gilliew & Willshaw model consisting of
		three sections (one section for each dendritic tree)

	OBSERVATIONS:
	- same behavior as original model
	"""
	# Properties from SThprotocell.hoc
	all_Ra = 150.224
	all_cm = 1.0
	soma_L = 18.8
	soma_diam = 18.3112

	# Create soma
	soma = h.Section()
	soma.nseg = 1
	soma.Ra = all_Ra
	soma.diam = soma_diam
	soma.L = soma_L
	soma.cm = all_cm
	for mech in gillies_mechs: soma.insert(mech)
	setconductances(soma, -1)
	setionstyles_gillies(soma)
	
	# Right dendritic tree (Fig. 1, small one)
	dend1 = h.Section()
	dend1.connect(soma(0))
	dend1.diam = 1.9480 # diam of topmost parent section
	dend1.L = 549.86 # equivalent length using Rall model
	dend1.Ra = all_Ra
	dend1.cm = all_cm
	opt_nseg1 = int(np.ceil(dend1.L/(0.1*lambda_f(100., dend1.diam, dend1.Ra, dend1.cm))))
	dend1.nseg = opt_nseg1
	for mech in gillies_mechs: dend1.insert(mech) # insert mechanisms
	setconductances(dend1, 1) # set channel conductances
	setionstyles_gillies(dend1) # set ion styles

	# Left dendritic tree (Fig. 1, big one)
	dend0 = h.Section()
	dend0.connect(soma(1))
	dend0.diam = 3.0973 # diam of topmost parent section
	dend0.L = 703.34 # equivalent length using Rall model
	dend0.Ra = all_Ra
	dend0.cm = all_cm
	opt_nseg0 = int(np.ceil(dend0.L/(0.1*lambda_f(100., dend0.diam, dend0.Ra, dend0.cm))))
	dend0.nseg = opt_nseg0
	for mech in gillies_mechs: dend0.insert(mech) # insert mechanisms
	setconductances(dend0, 0) # set channel conductances
	setionstyles_gillies(dend0) # set ion styles

	# Create stimulator objects
	stim1 = h.IClamp(soma(0.5))
	stim2 = h.IClamp(soma(0.5))
	stim3 = h.IClamp(soma(0.5))

	return soma, (dend0, dend1), (stim1, stim2, stim3)

def stn_cell_threecomp():
	""" Initialize reduced Gilliew & Willshaw model consisting of
		three compartments (three sections with one segment each
		in NEURON)

	OBSERVATIONS
	- cannot get same behavour as in original model or reduced model
	  with large amount of compartments. Processes/interaction of membrane
	  mechanisms in distal dendrite is not sufficiently isolated from
	  processes in soma.
	"""
	# Properties from SThprotocell.hoc
	all_Ra = 150.224
	all_cm = 1.0
	soma_L = 18.8
	soma_diam = 18.3112

	# Create soma
	soma = h.Section()
	soma.nseg = 1
	soma.Ra = all_Ra
	soma.diam = soma_diam
	soma.L = soma_L
	soma.cm = all_cm
	for mech in gillies_mechs: soma.insert(mech)
	setconductances(soma, -1)
	setionstyles_gillies(soma)
	
	# Right dendritic tree (Fig. 1, small one)
	d_proximal, d_distal = 1.9480, 0.4870 # diam of most proximal/parent sec, most distal sec
	dend1 = h.Section()
	dend1.connect(soma(0))
	dend1.diam = d_proximal
	dend1.L = 549.86 # equivalent length using Rall model
	dend1.Ra = all_Ra
	dend1.cm = all_cm
	dend1.nseg = 1
	for mech in gillies_mechs: dend1.insert(mech) # insert mechanisms
	setconductances(dend1, 1, fixbranch=8, fixloc=0.98) # set channel conductances
	setionstyles_gillies(dend1) # set ion styles

	# Left dendritic tree (Fig. 1, big one)
	d_proximal, d_distal = 3.0973, 0.4870 # diam of most proximal/parent sec, most distal sec
	dend0 = h.Section()
	dend0.connect(soma(1))
	dend0.diam = d_proximal
	dend0.L = 703.34 # equivalent length using Rall model
	dend0.Ra = all_Ra
	dend0.cm = all_cm
	dend0.nseg = 1
	for mech in gillies_mechs: dend0.insert(mech) # insert mechanisms
	setconductances(dend0, 0, fixbranch=10, fixloc=0.98) # set channel conductances
	setionstyles_gillies(dend0) # set ion styles

	# Create stimulator objects
	stim1 = h.IClamp(soma(0.5))
	stim2 = h.IClamp(soma(0.5))
	stim3 = h.IClamp(soma(0.5))

	return soma, (dend0, dend1), (stim1, stim2, stim3)

def stn_cell_resurgent():
    """ Initialize reduced Gilliew & Willshaw model consisting of
        three sections (one section for each dendritic tree)

    """
    # Properties from SThprotocell.hoc
    all_Ra = 150.224
    all_cm = 1.0
    soma_L = 18.8
    soma_diam = 18.3112

    # List of mechanisms to insert
    stn_mechs = list(gillies_mechs)
    stn_mechs.append('Narsg') # add Raman & bean (2001) resurgent channel model

    # List of channel conductances
    stn_glist = list(gillies_glist)
    stn_glist.append('gbar_Narsg')

    # Create soma
    soma = h.Section()
    soma.nseg = 1
    soma.Ra = all_Ra
    soma.diam = soma_diam
    soma.L = soma_L
    soma.cm = all_cm
    for mech in stn_mechs: soma.insert(mech)
    setconductances(soma, -1, glist=stn_glist)
    setionstyles_gillies(soma)
    
    # Right dendritic tree (Fig. 1, small one)
    dend1 = h.Section()
    dend1.connect(soma(0))
    dend1.diam = 1.9480 # diam of topmost parent section
    dend1.L = 549.86 # equivalent length using Rall model
    dend1.Ra = all_Ra
    dend1.cm = all_cm
    opt_nseg1 = int(np.ceil(dend1.L/(0.1*lambda_f(100., dend1.diam, dend1.Ra, dend1.cm))))
    dend1.nseg = opt_nseg1
    for mech in stn_mechs: dend1.insert(mech) # insert mechanisms
    setconductances(dend1, 1, glist=stn_glist) # set channel conductances
    setionstyles_gillies(dend1) # set ion styles

    # Left dendritic tree (Fig. 1, big one)
    dend0 = h.Section()
    dend0.connect(soma(1))
    dend0.diam = 3.0973 # diam of topmost parent section
    dend0.L = 703.34 # equivalent length using Rall model
    dend0.Ra = all_Ra
    dend0.cm = all_cm
    opt_nseg0 = int(np.ceil(dend0.L/(0.1*lambda_f(100., dend0.diam, dend0.Ra, dend0.cm))))
    dend0.nseg = opt_nseg0
    for mech in stn_mechs: dend0.insert(mech) # insert mechanisms
    setconductances(dend0, 0, glist=stn_glist) # set channel conductances
    setionstyles_gillies(dend0) # set ion styles

    # Create stimulator objects
    stim1 = h.IClamp(soma(0.5))
    stim2 = h.IClamp(soma(0.5))
    stim3 = h.IClamp(soma(0.5))

    return soma, (dend0, dend1), (stim1, stim2, stim3)

################################################################################
# Functions for reduced model
################################################################################

def setconductances(sec, dendidx, fixbranch=None, fixloc=None, glist=None):
	""" Set conductances at the node/midpoint of each segment
		by interpolating values along longest path
		(e.g. along branch 1-2-5 in dend1)

	@param fixbranch	if you want to map the section to a fixed
						branch, provide its number/index
	@param fixloc		if you want to map all segments/nodes
						to a fixed location on the mapped branch,
						provide a location (0<=x<=1)
	@param glist		list of mechanisms to set conductances for
	"""

	# Load channel conductances from file
	allgmats = reducemodel.loadgstructs()
	if glist is None:
		glist = list(gillies_glist)

	# Na & NaL are not from file
	h("default_gNa_soma = 1.483419823e-02")
	h("default_gNa_dend = 1.0e-7")
	h("default_gNaL_soma = 1.108670852e-05")
	h("default_gNaL_dend = 0.81e-5")

	# branch indices along longest path
	if dendidx == 1:
		geostruct = reducemodel.loadgeotopostruct(dendidx)
		longestpath = np.array([1,2,5])
		pathL = np.array([geostruct[i-1]['L'] for i in longestpath])
	elif dendidx == 0:
		geostruct = reducemodel.loadgeotopostruct(dendidx)
		longestpath = np.array([1,2,4,7])
		pathL = np.array([geostruct[i-1]['L'] for i in longestpath])
	else: # -1: soma
		# dimensions not in treeX-nom.dat file
		longestpath = np.array([1]) # soma is dendidx=-1, branchidx=0 in file
		pathL = np.array([18.8])

	# Distributed conductances: interpolate each conductance along longest path
	for iseg in range(1, sec.nseg+1):
		xnode = (2.*iseg-1.)/(2.*sec.nseg) # arclength of current node (segment midpoint)
		lnode = xnode*sum(pathL) # equivalent length along longest path

		# first determine on which branch we are and how far on it
		if (fixbranch is not None) and (fixloc is not None):
			nodebranch = fixbranch
			xonbranch = fixloc
		else:
			nodebranch = np.NaN # invalid branch
			xonbranch = 0
			for i, branchidx in enumerate(longestpath): # map node to branch and location
				if lnode <= pathL[0:i+1].sum(): # location maps to this branch
					begL = pathL[0:i].sum()
					endL = pathL[0:i+1].sum()
					nodebranch = branchidx
					xonbranch = (lnode-begL)/(endL-begL) # how far along this branch are we
					break
			if np.isnan(nodebranch):
				raise Exception('could not map to branch')

		# now interpolate all conductances from file
		for gname, gmat in allgmats.iteritems():
			if gname not in glist:
				print('Skipping conductance: '+gname)
				continue
			print('Setting conductance: '+gname)
			branchrows = (gmat['dendidx']==dendidx) & (gmat['branchidx']==nodebranch-1)
			gnode = np.interp(xonbranch, gmat[branchrows]['x'], gmat[branchrows]['g'])
			sec(xnode).__setattr__(gname, gnode)

		# Conductances with constant value (vals: see tools.hoc/washTTX)
		gNa = 1.483419823e-02 if dendidx==-1 else 1.0e-7 # see h.default_gNa_soma/dend in .hoc file
		gNaL = 1.108670852e-05 if dendidx==-1 else 0.81e-5 # see h.default_gNaL_soma/dend in .hoc file
		gNarsg = 0.016 # same as in .mod file and Akeman papeer
		g_fixed = {'gna_Na':gNa, 'gna_NaL':gNaL, 'gbar_Narsg':gNarsg} # NOTE: Narsg is NOT in Gillies model
		for gname, gval in g_fixed.iteritems():
			if gname not in glist: continue
			sec(xnode).__setattr__(gname, gval)

def setionstyles_gillies(sec):
	""" Set ion styles to work correctly with membrane mechanisms """
	sec.push()
	h.ion_style("na_ion",1,2,1,0,1)
	h.ion_style("k_ion",1,2,1,0,1)
	h.ion_style("ca_ion",3,2,1,1,1)
	h.pop_section()

def set_aCSF(req):
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

def applyApamin(soma, dends):
	""" Apply apamin (reduce sKCa conductance)

		NOTE: in paper they say reduce by 90 percent but in code
		they set everything to 0 except in soma where they divide
		by factor 10
	"""
	soma(0.5).__setattr__('gk_sKCa', 0.0000068)
	for sec in dends:
		for iseg in range(1, sec.nseg+1):
			xnode = (2.*iseg-1.)/(2.*sec.nseg) # arclength of current node (segment midpoint)
			sec(xnode).__setattr__('gk_sKCa', 0.0)

################################################################################
# Experiments
################################################################################

def test_spontaneous(resurgent=False, fullmodel=True):
	""" Run rest firing experiment from original Hoc file 

	PAPER

	TEST RESULTS

	CURRENTS
	- during ISI:
		- INaP (INaL) is significant (-0.02) in ISI and looks like main mechanism responsible
		  for spontaneous depolarization/firing
		- IKCa (repolarizing) is also significant (+0.004) during ISI
			- sKCa is responsible for most of the AHP, as it should
		- ICaT slowly activates in dendrites but is small (0.0008)
			- might help spontaneous depolarization
	- during AP
		- the HVA currents ICaL and ICaN contribute to depolarization
		- peak IKv3 is twice as high as IKDR (contributes more to repolarization)
	

	"""

	# make set-up
	if fullmodel:
		soma, dends, stims = stn_cell_gillies()
	else:
		soma, dends, stims = stn_cell_threesec()
		

	# Set simulation parameters
	dur = 2000
	h.dt = 0.025
	h.celsius = 37 # different temp from paper
	h.v_init = -60 # paper simulations use default v_init
	set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

	# Record
	secs = {'soma': soma}
	traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)
	traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}
	# Na currents
	if resurgent:
		traceSpecs['I_Na'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'ina'}
	else:
		traceSpecs['I_Na'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'ina'}
	traceSpecs['I_NaL'] = {'sec':'soma','loc':0.5,'mech':'NaL','var':'inaL'}
	# K currents
	traceSpecs['I_KDR'] = {'sec':'soma','loc':0.5,'mech':'KDR','var':'ik'}
	traceSpecs['I_Kv3'] = {'sec':'soma','loc':0.5,'mech':'Kv31','var':'ik'}
	traceSpecs['I_KCa'] = {'sec':'soma','loc':0.5,'mech':'sKCa','var':'isKCa'}
	traceSpecs['I_h'] = {'sec':'soma','loc':0.5,'mech':'Ih','var':'ih'}
	# Ca currents
	traceSpecs['I_CaL'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iLCa'}
	traceSpecs['I_CaN'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iNCa'}
	traceSpecs['I_CaT'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'iCaT'}
	recordStep = 0.05
	recData = analysis.recordTraces(secs, traceSpecs, recordStep)

	# Simulate
	h.tstop = dur
	h.init() # calls finitialize() and fcurrent()
	h.run()

	# Analyze
	analysis.plotTraces(recData, recordStep)
	recI = collections.OrderedDict()
	for key in reversed(recData): recI[key] = recData[key]
	recI.pop('V_soma')
	analysis.cumulPlotTraces(recI, recordStep, cumulate=False)

def test_plateau(fulltree=True, fullseg=True):
	""" Test plateau potential evoked by applying depolarizing pulse 
		at hyperpolarized level of membrane potential

	GILLIES CURRENTS

	OTSUKA CURRENTS
	- KA peak decreases during burst (height of peaks during AP), as KA decreases the firing frequency also decreases
		- hence KA seems to responsible for rapid repolarization and maintenance of high-frequency firing
	- KCa peak increases over about 25 ms (height of peaks during AP), and decreases during last 100 ms of burst
	- CaT is the first depolarizing current that rises after realease from hyperpolarization and seems to be
	  responsible for initiation of the rebound burst
		- CaT bootstraps burst (bootstraps pos feedback of CaL entry)
		- it runs out of fuel during burst and thus may contribute to ending the burst
			- this is contradicted by burst at regular Vm: there drop in ICaL clearly ends burst
	- CaL reaces steady maximum peak after approx. 70 ms into the burst, after CaT is already past its peak
		- hypothesis that it seems responsible for prolonging th burst seems plausible
		- burst seems to go on as long as CaT+CaL remains approx. constant, and burst ends as long as CaT too low

	"""
	# make set-up
	if fulltree:
		soma, dends, stims = stn_cell_gillies()
		# Load section indicated with arrow in fig. 5C
		# If you look at tree1-nom.dat it should be the seventh entry 
		# (highest L and nseg with no child sections of which there are two instances)
		dendsec = h.SThcell[0].dend1[7]
		dendloc = 0.8 # approximate location along dendrite in fig. 5C
	elif fullseg:
		soma, dends, stims = stn_cell_threesec()
		dendsec = dends[1]
		dendloc = 0.9
	else:
		soma, dends, stims = stn_cell_threecomp()
		dendsec = dends[1]
		dendloc = 0.9
	stim1, stim2, stim3 = stims[0], stims[1], stims[2]

	# Set simulation parameters
	dur = 2000
	h.dt = 0.025
	h.celsius = 30 # different temp from paper
	h.v_init = -60 # paper simulations sue default v_init
	set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

	# Set up stimulation (5 mA/cm2 for 80 ms)
	cellarea = np.pi*soma.diam*soma.L # (micron^2)
	I_hyper = -0.17 # hyperpolarize to -70 mV (see fig. 10C)
	I_depol = I_hyper + 0.2 # see fig. 10D: 0.2 nA (=stim.amp) over hyperpolarizing current
	dur_depol = 50 # see fig. 10D, top right
	del_depol = 1000

	stim1.delay = 0
	stim1.dur = del_depol
	stim1.amp = I_hyper

	stim2.delay = del_depol
	stim2.dur = dur_depol
	stim2.amp = I_depol

	stim3.delay = del_depol + dur_depol
	stim3.dur = dur - (del_depol + dur_depol)
	stim3.amp = I_hyper

	# Record
	secs = {'soma': soma, 'dend': dendsec}
	traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)
	traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}
	# Na currents
	traceSpecs['I_Na'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'ina'}
	traceSpecs['I_NaL'] = {'sec':'soma','loc':0.5,'mech':'NaL','var':'inaL'}
	# K currents
	traceSpecs['I_KDR'] = {'sec':'soma','loc':0.5,'mech':'KDR','var':'ik'}
	traceSpecs['I_Kv3'] = {'sec':'soma','loc':0.5,'mech':'Kv31','var':'ik'}
	traceSpecs['I_KCa'] = {'sec':'soma','loc':0.5,'mech':'sKCa','var':'isKCa'}
	traceSpecs['I_h'] = {'sec':'soma','loc':0.5,'mech':'Ih','var':'ih'}
	# K currents (dendrite)
	traceSpecs['dI_KCa'] = {'sec':'dend','loc':dendloc,'mech':'sKCa','var':'isKCa'}
	# Ca currents (soma)
	traceSpecs['I_CaL'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iLCa'}
	traceSpecs['I_CaN'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iNCa'}
	traceSpecs['I_CaT'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'iCaT'}
	# Ca currents (dendrite)
	traceSpecs['dI_CaL'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iLCa'}
	traceSpecs['dI_CaN'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iNCa'}
	traceSpecs['dI_CaT'] = {'sec':'dend','loc':dendloc,'mech':'CaT','var':'iCaT'}
	# Start recording
	recordStep = 0.05
	recData = analysis.recordTraces(secs, traceSpecs, recordStep)

	# Simulate
	h.tstop = dur
	h.init() # calls finitialize() and fcurrent()
	h.run()

	# Analyze
	burst_time = [del_depol-50, del_depol+200]
	# Soma currents
	recSoma = collections.OrderedDict()
	for k, v in recData.iteritems():
		if not k.startswith('d'): recSoma[k] = recData[k]
	analysis.plotTraces(recSoma, recordStep, timeRange=burst_time)
	# Soma currents (relative)
	recSoma.pop('V_soma')
	analysis.cumulPlotTraces(recSoma, recordStep, cumulate=False, timeRange=burst_time)
	# Dendrite currents
	recDend = collections.OrderedDict()
	for k, v in recData.iteritems():
		if k.startswith('d'): recDend[k] = recData[k]
	analysis.cumulPlotTraces(recDend, recordStep, cumulate=False, timeRange=burst_time)

def test_reboundburst(fulltree=True, fullseg=True):
	""" Run rebound burst experiment from original Hoc file

	GILLIES CURRENTS
	- same 'CaT bootstraps CaL' mechanism: 
		- small peak in CaT at beginning of burst triggers sharp rise in ICaL with 9x higher peak
		- successive CaL peaks decline in magnitude during burst\
	- Ih/HCN sharply declines in peak magnitude over burst to insignificance (approx negexp)
		- burst ends when it died out
		- there is no similar current in Otsuka model

	OTSUKA COMPARISON
		- In Gillies model CaT bootstrap is a small single peak at start of burst, 
		  while in Otsuka model CaT is exp declining peaks
		- In Gillies model ICaL is a declining ramp of peaks (approx linear), while 
		  in Otsuka it is slowly activated and inactivated (bugle of peaks)
		- Ih/HCN and IKCa which determine shape/recovery of AP are different, resulting
		  in a different evolution of AP shape within a burst

	"""

	# make set-up
	if fulltree:
		soma, dends, stims = stn_cell_gillies()
		# Load section indicated with arrow in fig. 5C
		# If you look at tree1-nom.dat it should be the seventh entry 
		# (highest L and nseg with no child sections of which there are two instances)
		dendsec = h.SThcell[0].dend1[7]
		dendloc = 0.8 # approximate location along dendrite in fig. 5C
	elif fullseg:
		soma, dends, stims = stn_cell_threesec()
		dendsec = dends[1]
		dendloc = 0.9
	else:
		soma, dends, stims = stn_cell_threecomp()
		dendsec = dends[1]
		dendloc = 0.9
	stim1, stim2, stim3 = stims[0], stims[1], stims[2]

	# Set simulation parameters
	dur = 2000
	h.dt = 0.025
	h.celsius = 35 # different temp from paper
	h.v_init = -60 # paper simulations sue default v_init
	set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

	# Set up stimulation
	stim1.delay = 0
	stim1.dur = 500
	stim1.amp = 0.0

	stim2.delay = 500
	stim2.dur = 500
	stim2.amp = -0.25

	stim3.delay = 1000
	stim3.dur = 1000
	stim3.amp = 0.0

	# Record
	secs = {'soma': soma, 'dend': dendsec}
	traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)
	traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}
	# Na currents
	traceSpecs['I_Na'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'ina'}
	traceSpecs['I_NaL'] = {'sec':'soma','loc':0.5,'mech':'NaL','var':'inaL'}
	# K currents
	traceSpecs['I_KDR'] = {'sec':'soma','loc':0.5,'mech':'KDR','var':'ik'}
	traceSpecs['I_Kv3'] = {'sec':'soma','loc':0.5,'mech':'Kv31','var':'ik'}
	traceSpecs['I_KCa'] = {'sec':'soma','loc':0.5,'mech':'sKCa','var':'isKCa'}
	traceSpecs['I_h'] = {'sec':'soma','loc':0.5,'mech':'Ih','var':'ih'}
	# K currents (dendrite)
	traceSpecs['dI_KCa'] = {'sec':'dend','loc':dendloc,'mech':'sKCa','var':'isKCa'}
	# Ca currents (soma)
	traceSpecs['I_CaL'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iLCa'}
	traceSpecs['I_CaN'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iNCa'}
	traceSpecs['I_CaT'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'iCaT'}
	# Ca currents (dendrite)
	traceSpecs['dI_CaL'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iLCa'}
	traceSpecs['dI_CaN'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iNCa'}
	traceSpecs['dI_CaT'] = {'sec':'dend','loc':dendloc,'mech':'CaT','var':'iCaT'}
	recordStep = 0.05
	recData = analysis.recordTraces(secs, traceSpecs, recordStep)

	# Simulate
	h.tstop = dur
	h.init() # calls finitialize() and fcurrent()
	h.run()

	# Analyze
	burst_time = [980, 1120]
	# Soma currents
	recSoma = collections.OrderedDict()
	for k, v in recData.iteritems():
		if not k.startswith('d'): recSoma[k] = recData[k]
	analysis.plotTraces(recSoma, recordStep)
	# Soma currents (relative)
	recSoma.pop('V_soma')
	analysis.cumulPlotTraces(recSoma, recordStep, cumulate=False)
	# Dendrite currents
	recDend = collections.OrderedDict()
	for k, v in recData.iteritems():
		if k.startswith('d'): recDend[k] = recData[k]
	analysis.cumulPlotTraces(recDend, recordStep, cumulate=False)
	# Overlay voltage signal
	# plt.plot(np.arange(0, dur, recordStep), recData['V_soma'].as_numpy()*1e-3, color='r')
	# plt.show(block=False)
	# Plot dendrite currentds

def test_slowbursting(fulltree=True, fullseg=True):
	""" Test slow rhythmic bursting mode under conditions of constant 
		hyperpolarizing current injection and lower sKCa conductance

	PROTOCOL
	- lower sKCa conductance by 90percent to promote bursting
	- inject constant hyperpolarizing current

	PAPER OBSERVATIONS
	- CaL itself is responsible for burst initiation
			- CaL bootstraps itself in the dendrites
	- intra-burst: inter-AP Vm gradually hyperpolarizes due to current injection
	- inter-burst: in the dendrites, CaL and CaT (very small due to relatively high Vm) gradually depolarize the membrane
		- slow depolarization continues until majority of CaL channels activated
	"""

	# make set-up
	if fulltree:
		soma, dends, stims = stn_cell_gillies()
		# Load section indicated with arrow in fig. 5C
		# If you look at tree1-nom.dat it should be the seventh entry 
		# (highest L and nseg with no child sections of which there are two instances)
		dendsec = h.SThcell[0].dend1[7]
		dendloc = 0.8 # approximate location along dendrite in fig. 5C
	elif fullseg:
		soma, dends, stims = stn_cell_threesec()
		dendsec = dends[1]
		dendloc = 0.9
	else:
		soma, dends, stims = stn_cell_threecomp()
		dendsec = dends[1]
		dendloc = 0.9
	stim1, stim2, stim3 = stims[0], stims[1], stims[2]

	# Set simulation parameters
	dur = 2000
	h.dt = 0.025
	h.celsius = 37 # for slow bursting experiment
	h.v_init = -60 # paper simulations sue default v_init
	set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

	# Set up stimulation
	stim1.delay = 0
	stim1.dur = dur
	stim1.amp = -0.25

	stim2.delay = 0
	stim2.dur = 0
	stim2.amp = 0.0

	stim3.delay = 0
	stim3.dur = 0
	stim3.amp = 0.0 

	# Record
	secs = {'soma': soma, 'dend': dendsec}
	traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)
	traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}
	# Na currents
	traceSpecs['I_Na'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'ina'}
	traceSpecs['I_NaL'] = {'sec':'soma','loc':0.5,'mech':'NaL','var':'inaL'}
	# K currents
	traceSpecs['I_KDR'] = {'sec':'soma','loc':0.5,'mech':'KDR','var':'ik'}
	traceSpecs['I_Kv3'] = {'sec':'soma','loc':0.5,'mech':'Kv31','var':'ik'}
	traceSpecs['I_KCa'] = {'sec':'soma','loc':0.5,'mech':'sKCa','var':'isKCa'}
	traceSpecs['I_h'] = {'sec':'soma','loc':0.5,'mech':'Ih','var':'ih'}
	# K currents (dendrite)
	traceSpecs['dI_KCa'] = {'sec':'dend','loc':dendloc,'mech':'sKCa','var':'isKCa'}
	# Ca currents (soma)
	traceSpecs['I_CaL'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iLCa'}
	traceSpecs['I_CaN'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iNCa'}
	traceSpecs['I_CaT'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'iCaT'}
	# Ca currents (dendrite)
	traceSpecs['dI_CaL'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iLCa'}
	traceSpecs['dI_CaN'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iNCa'}
	traceSpecs['dI_CaT'] = {'sec':'dend','loc':dendloc,'mech':'CaT','var':'iCaT'}
	# Start recording
	recordStep = 0.05
	recData = analysis.recordTraces(secs, traceSpecs, recordStep)

	# Simulate
	h.tstop = dur
	if fullmodel:
		h.applyApamin() # lower sKCa conductance by 90 percent
	else:
		applyApamin(soma, dends)
	h.init()
	h.run()
	if fullmodel:
		h.washApamin() # restore sKCa conductance to original level

	# Analyze
	burst_time = [] # enter burst time
	# Soma currents
	recSoma = collections.OrderedDict()
	for k, v in recData.iteritems():
		if not k.startswith('d'): recSoma[k] = recData[k]
	analysis.plotTraces(recSoma, recordStep)
	# Soma currents (relative)
	recSoma.pop('V_soma')
	analysis.cumulPlotTraces(recSoma, recordStep, cumulate=False)
	# Dendrite currents (relative)
	recDend = collections.OrderedDict()
	for k, v in recData.iteritems():
		if k.startswith('d'): recDend[k] = recData[k]
	analysis.cumulPlotTraces(recDend, recordStep, cumulate=False)

if __name__ == '__main__':
	test_spontaneous(resurgent=False)
	# test_reboundburst()
	# test_plateau(False, False)
	# test_slowbursting()
	# soma, dends = stn_cell_threesec()