"""
Run experiments using the Gillies & Willshaw (2006) STN neuron model

The experiments are designed to discover which currents are responsible
for the different features of STN cell dynamics

@author Lucas Koelman
@date	28-10-2016
@note	must be run from script directory or .hoc files not found

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import neuron
from neuron import h

import sys
import os.path
scriptdir, scriptfile = os.path.split(__file__)
modulesbase = os.path.normpath(os.path.join(scriptdir, '..'))
sys.path.append(modulesbase)
import collections
import re

from common import analysis
import reducemodel
from reducemodel import lambda_AC
import reduce_marasco as marasco

# Load NEURON mechanisms
# add this line to nrn/lib/python/neuron/__init__.py/load_mechanisms()
# from sys import platform as osplatform
# if osplatform == 'win32':
# 	lib_path = os.path.join(path, 'nrnmech.dll')
NRN_MECH_PATH = os.path.normpath(os.path.join(scriptdir, 'nrn_mechs'))
neuron.load_mechanisms(NRN_MECH_PATH)
	

# Load NEURON function libraries
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

# Global variables
soma = None
gillies_mechs = ['STh', 'Na', 'NaL', 'KDR', 'Kv31', 'Ih', 'Cacum', 'sKCa', 'CaT', 'HVA'] # all mechanisms
gillies_gdict = {'STh': ['gpas'], # passive/leak channel
				'Na': ['gna'], 'NaL':['gna'], # Na channels
				'KDR':['gk'], 'Kv31':['gk'], 'sKCa':['gk'], # K channels
				'Ih':['gk'], # nonspecific channels
				'CaT':['gcaT'], 'HVA':['gcaL', 'gcaN']} # Ca channels
gillies_glist = [gname+'_'+mech for mech,chans in gillies_gdict.iteritems() for gname in chans]


def get_gillies_secs():
	""" Get soma and dendritic trees when full model has been created. 
	@return		tuple soma, (dend0, dend1)
	"""
	secs = list(h.allsec())
	soma = secs[0]
	dend0 = secs[1:24]
	dend1 = secs[24:-1]
	return soma, (dend0, dend1)

def stn_cell_gillies(resurgent=False):
	""" Initialize full Gillies & Willshaw cell model with two dendritic trees
	@param resurgent	if True, resurgent Na current mechanism
						from Do & Bean (2001) is added to the cell
	"""
	# Create cell and three IClamp in soma
	global soma
	if soma is None:
		h.xopen("createcell.hoc")
		soma = h.SThcell[0].soma
	dends = h.SThcell[0].dend0, h.SThcell[0].dend1
	stims = h.stim1, h.stim2, h.stim3

	# Add resurgent current if requested
	if resurgent:
		default_gNarsg_soma = 0.016 # same as in .mod file and Akeman papeer
		default_gNarsg_dend = 1.0e-7 # default value for Na in Gillies model
		soma.insert('Narsg')
		soma.gbar_Narsg = default_gNarsg_soma
		alldendsecs = list(dends[0]) + list(dends[1])
		for sec in alldendsecs:
			sec.insert('Narsg')
			sec.gbar_Narsg = default_gNarsg_dend

	return soma, dends, stims

def stn_cell_Rall(resurgent=False, oneseg=False):
	""" Initialise reduced Gilliew & Willshaw model with each dendritic tree
	reduced to a single equivalent cylinder/section according to Rall's
	reduction method and the number of segments/points determined by
	the electrotonic length.

	@param oneseg	if True, a single segment is usef for each section,
					otherwise the number of segments is determined by 
					dL < 0.1*lambda(100)

	OBSERVATIONS:
	- (nseg from lambda)
		- same behavior as original model except spontaneous firing

	- (nseg=1)
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

	# List of mechanisms/conductances
	stn_mechs = list(gillies_mechs) # List of mechanisms to insert
	stn_glist = list(gillies_glist) # List of channel conductances to modify
	if resurgent: # add Raman & bean (2001) resurgent channel model
		stn_mechs.append('Narsg')
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
	# Configure ion channels (distribution,...)
	for mech in stn_mechs: dend1.insert(mech)
	if oneseg:
		dend1.nseg = 1
		setconductances(dend1, 1, fixbranch=8, fixloc=0.98)
	else:
		opt_nseg1 = int(np.ceil(dend1.L/(0.1*lambda_AC(dend1,100))))
		dend1.nseg = opt_nseg1
		setconductances(dend1, 1, glist=stn_glist)
	setionstyles_gillies(dend1) # set ion styles

	# Left dendritic tree (Fig. 1, big one)
	dend0 = h.Section()
	dend0.connect(soma(1))
	dend0.diam = 3.0973 # diam of topmost parent section
	dend0.L = 703.34 # equivalent length using Rall model
	dend0.Ra = all_Ra
	dend0.cm = all_cm
	# Configure ion channels (distribution,...)
	for mech in stn_mechs: dend0.insert(mech) # insert mechanisms
	if oneseg:
		dend0.nseg = 1
		setconductances(dend0, 0, fixbranch=10, fixloc=0.98)
	else:
		opt_nseg0 = int(np.ceil(dend0.L/(0.1*lambda_AC(dend0,100))))
		dend0.nseg = opt_nseg0
		setconductances(dend0, 0, glist=stn_glist)
	setionstyles_gillies(dend0) # set ion styles

	# Create stimulator objects
	stim1 = h.IClamp(soma(0.5))
	stim2 = h.IClamp(soma(0.5))
	stim3 = h.IClamp(soma(0.5))

	return soma, (dend0, dend1), (stim1, stim2, stim3)

def stn_cell_spiny_smooth_sec():
	""" 
	Soma with a single tree reduced to two compartments: one equivalent
	trunk/smooth sections + one equivalent spiny sections.

	The single tree is equivalent to three instances
	of the small (fig. 1, right) dendritic tree connected to the soma.
	The number og segments is determined empirically.

	[0..soma..1]-[0..smooth..1]-[0..spiny..1]

	TODO: use surface equivalence to scale conductances/cm and conserve axial resistance?
	"""
	# Properties from SThprotocell.hoc
	all_Ra = 150.224
	all_cm = 1.0
	soma_L = 18.8
	soma_diam = 18.3112

	# List of mechanisms/conductances
	stn_mechs = list(gillies_mechs) # List of mechanisms to insert
	stn_glist = list(gillies_glist) # List of channel conductances to modify

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
	
	# Primary neurite, equiv. to smooth dendrites
	smooth = h.Section()
	smooth.connect(soma, 1, 0) # connect parent@0.0 to child@1.0
	smooth.diam = (1.9480+1.2272)/2.0 * 1.0 # 3 times average diameter of branch 1 & 2
	smooth.L = 80 # sum of diameters branch 1 & 2
	smooth.Ra = all_Ra
	smooth.cm = all_cm
	opt_nseg1 = int(np.ceil(smooth.L/(0.1*lambda_AC(smooth,100))))
	smooth.nseg = 9
	for mech in stn_mechs: smooth.insert(mech) # insert mechanisms
	interpconductances(smooth, 1, path=[1,2], glist=stn_glist) # set channel conductances
	setionstyles_gillies(smooth) # set ion styles
	print("Created section with {0} segments".format(smooth.nseg))

	# Secondary neurite, equiv. to spiny dendrites
	spiny = h.Section()
	spiny.connect(smooth, 1, 0) # connect parent@0.0 to child@1.0
	spiny.diam = 0.7695 * 1.0 # 3 times diameter of long branch (branch 5)
	spiny.L = 289 # lenth of long branch
	spiny.Ra = all_Ra
	spiny.cm = all_cm
	opt_nseg0 = int(np.ceil(spiny.L/(0.1*lambda_AC(spiny,100))))
	spiny.nseg = 9
	for mech in stn_mechs: spiny.insert(mech) # insert mechanisms
	interpconductances(spiny, 1, path=[5], glist=stn_glist) # set channel conductances
	setionstyles_gillies(spiny) # set ion styles
	print("Created section with {0} segments".format(spiny.nseg))

	# Create stimulator objects
	stim1 = h.IClamp(soma(0.5))
	stim2 = h.IClamp(soma(0.5))
	stim3 = h.IClamp(soma(0.5))

	return soma, (smooth, spiny), (stim1, stim2, stim3)

def stn_cell(cellmodel):
	""" 
	Create stn cell with given identifier 

	ID		model description
	-------------------------
	1		Original Gillies & Willshaw STN cell model

	2		Gilles & Willshaw model reduced using Rall's model
			to one equivalent section per dendritic tree. The
			number of segments is determined from the electrotonic
			length to yield sufficient accuracy

	3		Same as 2, except only one segment per secion is used

	4		Gillies & Willshaw STN cell model reduced using
			Marasco's reduction method with a custom clustering
			criterion based on diameter

	5		Reduced Gillies & Willshaw STN model using Marasco's
			reduction method except subtrees/axial resistance is not
			averaged but compounded and input resistance is conserved.
	"""
	stims = None
	if cellmodel==1: # Full model
		soma, dends, stims = stn_cell_gillies()
		# Load section indicated with arrow in fig. 5C
		# If you look at tree1-nom.dat it should be the seventh entry 
		# (highest L and nseg with no child sections of which there are two instances)
		dendsec = h.SThcell[0].dend1[7]
		dendloc = 0.8 # approximate location along dendrite in fig. 5C
		allsecs = [soma] + list(dends[0]) + list(dends[1])
	elif cellmodel==2: # Rall - optimal nseg
		soma, dends, stims = stn_cell_Rall()
		dendsec = dends[1]
		dendloc = 0.9
		allsecs = [soma] + list(dends)
	elif cellmodel==3: # Rall - one segment
		soma, dends, stims = stn_cell_Rall(oneseg=True)
		dendsec = dends[1]
		dendloc = 0.9
		allsecs = [soma] + list(dends)
	elif cellmodel==4 or cellmodel==5: # Marasco - custom clustering
		marasco_method = True # whether trees will be averaged (True, as in paper) or Rin conserved (False)
		if cellmodel==5:
			marasco_method = False
		clusters, eq_secs, eq_refs = marasco.reduce_gillies(
			customclustering=True, average_trees=marasco_method)
		soma, dendLsecs, dendRsecs = eq_secs
		dendsec = dendRsecs[-1] # last/most distal section of small dendrite
		dendloc = 0.9
		allsecs = [soma] + dendLsecs + dendRsecs

	# Insert stimulation electrodes
	if stims is None:
		stim1 = h.IClamp(soma(0.5))
		stim2 = h.IClamp(soma(0.5))
		stim3 = h.IClamp(soma(0.5))
		stims = stim1, stim2, stim3
	
	return soma, [(dendsec, dendloc)], stims, allsecs

################################################################################
# Functions for reduced model
################################################################################
def interpconductances(sec, dendidx, path, glist=None):
	""" Interpolate conductances along given path of branches 
	@param sec			section to set conductances for
	@param dendidx		index of the dendritic tree
	@param path			indices of branches along dendritic tree
	@param glist		list of conductances to set (including mechanism suffix)
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
	geostruct = reducemodel.loadgeotopostruct(dendidx)
	pathL = np.array([geostruct[i-1]['L'] for i in path]) # length of each branch along path

	# Distributed conductances: interpolate each conductance along longest path
	for iseg in range(1, sec.nseg+1):
		xnode = (2.*iseg-1.)/(2.*sec.nseg) # arclength of current node (segment midpoint)
		lnode = xnode*sum(pathL) # equivalent length along longest path

		# first determine on which branch we are and how far on it
		nodebranch = np.NaN # invalid branch
		xonbranch = 0
		for i, branchidx in enumerate(path): # map node to branch and location
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


def setconductances(sec, dendidx, fixbranch=None, fixloc=None, glist=None):
	""" Set conductances at the node/midpoint of each segment
		by interpolating values along longest path
		(e.g. along branch 1-2-5 in dend1)

	@param fixbranch	if you want to map the section to a fixed
						branch, provide its number/index
	@param fixloc		if you want to map all segments/nodes
						to a fixed location on the mapped branch,
						provide a location (0<=x<=1)
	@param glist		list of conductances to set (including mechanism suffix)
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
# Recording & Analysis functions
################################################################################

def rec_currents_activations(traceSpecs):
	""" Specify trace recordings for all ionic currents
		and activation variables 

	@param traceSpecs	collections.OrderedDict of trace specifications

	@effect				for each ionic current, insert a trace specification
						for the current, open fractions, activation, 
						and inactivation variables
	"""
	# Na currents
	traceSpecs['I_NaT'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'ina'} # transient sodium
	traceSpecs['I_NaP'] = {'sec':'soma','loc':0.5,'mech':'NaL','var':'inaL'} # persistent sodium
	# K currents
	traceSpecs['I_KDR'] = {'sec':'soma','loc':0.5,'mech':'KDR','var':'ik'}
	traceSpecs['I_Kv3'] = {'sec':'soma','loc':0.5,'mech':'Kv31','var':'ik'}
	traceSpecs['I_KCa'] = {'sec':'soma','loc':0.5,'mech':'sKCa','var':'isKCa'}
	# Ca currents
	traceSpecs['I_CaL'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iLCa'}
	traceSpecs['I_CaN'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iNCa'}
	traceSpecs['I_CaT'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'iCaT'}
	# Nonspecific currents
	traceSpecs['I_HCN'] = {'sec':'soma','loc':0.5,'mech':'Ih','var':'ih'}

	# Na Channel open fractions
	traceSpecs['O_NaT'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'o'}
	# NOTE: leak currents such as I_NaL and gpas_STh are always open
	# K channel open fractions
	traceSpecs['O_KDR'] = {'sec':'soma','loc':0.5,'mech':'KDR','var':'n'}
	traceSpecs['O_Kv3'] = {'sec':'soma','loc':0.5,'mech':'Kv31','var':'p'}
	traceSpecs['O_KCa'] = {'sec':'soma','loc':0.5,'mech':'sKCa','var':'w'}
	# Ca channel open fractions
	traceSpecs['O_CaL'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'o_L'}
	traceSpecs['O_CaN'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'o_N'}
	traceSpecs['O_CaT'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'o'}
	# Nonspecific channel open fractions
	traceSpecs['O_HCN'] = {'sec':'soma','loc':0.5,'mech':'Ih','var':'f'}

	# Na channel activated fractions
	traceSpecs['A_NaT'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'m'}
	# K channels activated fractions - same as open fractions (single state variable)
	traceSpecs['A_KDR'] = {'sec':'soma','loc':0.5,'mech':'KDR','var':'n'}
	traceSpecs['A_Kv3'] = {'sec':'soma','loc':0.5,'mech':'Kv31','var':'p'}
	traceSpecs['A_KCa'] = {'sec':'soma','loc':0.5,'mech':'sKCa','var':'w'}
	# Ca channels activated fractions
	traceSpecs['A_CaL'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'q'} # shared activation var for L/N
	traceSpecs['A_CaN'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'q'} # shared activation var for L/N
	traceSpecs['A_CaT'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'r'}
	# Nonspecific channels activated fractions - same as open fractions (single state variable)
	traceSpecs['A_HCN'] = {'sec':'soma','loc':0.5,'mech':'Ih','var':'f'}

	# Na channel inactivated fractions
	traceSpecs['B_NaT'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'h'}
	# K channels inactivated fractions - not present (single state variable)
	# Ca channels inactivated fractions
	traceSpecs['B_CaL'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'h'} # Ca-depentdent inactivation
	traceSpecs['B_CaN'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'u'} # V-dependent inactivation
	traceSpecs['B_CaT_f'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'s'} # fast inactivation
	traceSpecs['B_CaT_s'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'d'} # slow inactivation
	# Nonspecific channels activated fractions - not present (single state variable)

def plot_currents_activations(recData, recordStep, timeRange=None):
	""" Plot currents and (in)activation variable for each ionic
		current in the same axis. Ionic currents are grouped per ion
		in one figure, and the x-axes are synchronized for zooming
		and panning.

	@param recData		traces recorded using a trace specification provided
						by rec_currents_activations()

	@return				tuple figs, cursors where figs is a list of figures
						that were created and cursors a list of cursors
	"""
	figs = []
	cursors = []

	# Plot activations-currents on same axis per current
	ions_chans = [('NaT', 'NaP', 'HCN'), ('KDR', 'Kv3', 'KCa'), ('CaL', 'CaN', 'CaT')]
	for ionchans in ions_chans: # one figure for each ion
		fig, axrows = plt.subplots(len(ionchans), 1, sharex=True) # one plot for each channel
		for i, chan in enumerate(ionchans):
			# Which traces need to be plotted
			pat = re.compile(r'^\w_' + chan)
			chanFilter = lambda x: re.search(pat, x)
			twinFilter = lambda	x: x.startswith('I_')
			# Plot traces that match pattern
			analysis.cumulPlotTraces(recData, recordStep, showFig=False, 
								fig=None, ax1=axrows[i], yRange=(-0.1,1.1), timeRange=timeRange,
								includeFilter=chanFilter, twinFilter=twinFilter)
			# Add figure interaction
			cursor = matplotlib.widgets.MultiCursor(fig.canvas, fig.axes, 
						color='r', lw=1, horizOn=False, vertOn=True)
		figs.append(fig)
		cursors.append(cursor)

	return figs, cursors

################################################################################
# Experiments
################################################################################

def test_spontaneous(soma, dends_locs, stims, resurgent=False):
	""" Run rest firing experiment from original Hoc file

	@param soma			soma section
	@param dends_locs	list of tuples (sec, loc) containing a section
						and x coordinate to place recording electrode
	@param stims		list of electrodes (IClamp)

	PROTOCOL
	- spontaneous firing: no stimulation

	OBSERVATIONS (GILLIES MODEL)

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
	# Set simulation parameters
	dur = 2000
	h.dt = 0.025
	h.celsius = 37 # different temp from paper (fig 3B: 25degC, fig. 3C: 35degC)
	h.v_init = -60 # paper simulations use default v_init
	set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

	# Recording: trace specification
	secs = {'soma': soma}
	traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)

	# Membrane voltages
	traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}
	for i, (dend,loc) in enumerate(dends_locs):
		dendname = 'dend%i' % i
		secs[dendname] = dend
		traceSpecs['V_'+dendname] = {'sec':dendname,'loc':loc,'var':'v'}

	# Record ionic currents, open fractions, (in)activation variables
	rec_currents_activations(traceSpecs)

	# Set up recording vectors
	recordStep = 0.05
	recData = analysis.recordTraces(secs, traceSpecs, recordStep)

	# Simulate
	h.tstop = dur
	h.init() # calls finitialize() and fcurrent()
	h.run()

	# Plot membrane voltages
	recV = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.startswith('V_')]) # preserves order
	figs_vm = analysis.plotTraces(recV, recordStep, yRange=(-80,40), traceSharex=True)
	vm_fig = figs_vm[0]
	vm_ax = figs_vm[0].axes[0]

	# Plot ionic currents, (in)activation variables
	figs, cursors = plot_currents_activations(recData, recordStep)

	# Plot ionic currents in separate axes
	# recI = collections.OrderedDict()
	# for k, v in recData.iteritems():
	# 	if k.startswith('I_'): recI[k] = v
	# traceYLims = {'I_Na': (-1., 0.1), 'I_NaL': (-2.5e-3, -5e-4),
	# 			'I_KDR': (0, 0.12), 'I_Kv3': (0, 0.20), 'I_KCa': (0, 0.014),
	# 			'I_h': (-5e-4, 2.5e-3), 'I_CaL': (-3.5e-2, 0), 'I_CaN': (-3e-2, 0),
	# 			'I_CaT': (-6e-2, 6e-2)}
	# figs_I = analysis.plotTraces(recI, recordStep, yRange=traceYLims, 
	# 							traceSharex=vm_ax, showFig=False)

	plt.show(block=False)
	return recData, figs, cursors

def test_plateau(soma, dends_locs, stims):
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
	# Get electrodes and sections to record from
	dendsec = dends_locs[0][0]
	dendloc = dends_locs[0][1]
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
	burst_time = [del_depol-50, del_depol+200] # empirical

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

	# Membrane voltages
	traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}
	for i, (dend,loc) in enumerate(dends_locs):
		dendname = 'dend%i' % i
		secs[dendname] = dend
		traceSpecs['V_'+dendname] = {'sec':dendname,'loc':loc,'var':'v'}

	# Record ionic currents, open fractions, (in)activation variables
	rec_currents_activations(traceSpecs)

	# K currents (dendrite)
	traceSpecs['I_KCa_d'] = {'sec':'dend','loc':dendloc,'mech':'sKCa','var':'isKCa'}
	# Ca currents (dendrite)
	traceSpecs['I_CaL_d'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iLCa'}
	traceSpecs['I_CaN_d'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iNCa'}
	traceSpecs['I_CaT_d'] = {'sec':'dend','loc':dendloc,'mech':'CaT','var':'iCaT'}

	# Start recording
	recordStep = 0.05
	recData = analysis.recordTraces(secs, traceSpecs, recordStep)

	# Simulate
	h.tstop = dur
	h.init() # calls finitialize() and fcurrent()
	h.run()

	# Plot membrane voltages
	recV = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.startswith('V_')]) # preserves order
	figs_vm = analysis.plotTraces(recV, recordStep, yRange=(-80,40), traceSharex=True)
	vm_fig = figs_vm[0]
	vm_ax = figs_vm[0].axes[0]

	# Plot ionic currents, (in)activation variables
	figs, cursors = plot_currents_activations(recData, recordStep)

	# # Soma currents
	# recSoma = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if not k.endswith('_d')])
	# Set fixed ranges for comparison
	# traceYLims = {'V_soma': (-80, 40), 'I_Na': (-0.7, 0.1), 'I_NaL': (-2e-3, 0),
	# 			'I_KDR': (0, 0.1), 'I_Kv3': (-1e-2, 0.1), 'I_KCa': (-1e-3, 8e-3),
	# 			'I_h': (-2e-3, 6e-3), 'I_CaL': (-1.5e-2, 1e-3), 'I_CaN': (-2e-2, 1e-3),
	# 			'I_CaT': (-6e-2, 6e-2)}
	# analysis.plotTraces(recSoma, recordStep, timeRange=burst_time, yRange=traceYLims)

	# # Soma currents (relative)
	# recSoma.pop('V_soma')
	# analysis.cumulPlotTraces(recSoma, recordStep, cumulate=False, timeRange=burst_time)

	# Dendrite currents during burst
	recDend = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.endswith('_d')])
	analysis.cumulPlotTraces(recDend, recordStep, timeRange=burst_time)
	
	return recData, figs, cursors

def test_reboundburst(soma, dends_locs, stims):
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
	# Get electrodes and sections to record from
	dendsec = dends_locs[0][0]
	dendloc = dends_locs[0][1]
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
	stim2.amp = -0.15 # -0.25 in full model

	stim3.delay = 1000
	stim3.dur = 1000
	stim3.amp = 0.0

	# Record
	secs = {'soma': soma, 'dend': dendsec}
	traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)

	# Membrane voltages
	traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}
	for i, (dend,loc) in enumerate(dends_locs):
		dendname = 'dend%i' % i
		secs[dendname] = dend
		traceSpecs['V_'+dendname] = {'sec':dendname,'loc':loc,'var':'v'}

	# Record ionic currents, open fractions, (in)activation variables
	rec_currents_activations(traceSpecs)

	# K currents (dendrite)
	traceSpecs['I_KCa_d'] = {'sec':'dend','loc':dendloc,'mech':'sKCa','var':'isKCa'}
	# Ca currents (dendrite)
	traceSpecs['I_CaL_d'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iLCa'}
	traceSpecs['I_CaN_d'] = {'sec':'dend','loc':dendloc,'mech':'HVA','var':'iNCa'}
	traceSpecs['I_CaT_d'] = {'sec':'dend','loc':dendloc,'mech':'CaT','var':'iCaT'}

	# Start recording
	recordStep = 0.05
	recData = analysis.recordTraces(secs, traceSpecs, recordStep)

	# Simulate
	h.tstop = dur
	h.init() # calls finitialize() and fcurrent()
	h.run()

	# Plot membrane voltages
	recV = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.startswith('V_')]) # preserves order
	figs_vm = analysis.plotTraces(recV, recordStep, yRange=(-80,40), traceSharex=True)
	vm_fig = figs_vm[0]
	vm_ax = figs_vm[0].axes[0]

	# Plot ionic currents, (in)activation variables
	figs, cursors = plot_currents_activations(recData, recordStep)

	# Dendrite currents during burst
	burst_time = [980, 1120]
	recDend = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.endswith('_d')])
	analysis.cumulPlotTraces(recDend, recordStep, timeRange=burst_time)

	return recData, figs, cursors

def test_slowbursting(soma, dends_locs, stims):
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
	# Get electrodes and sections to record from
	dendsec = dends_locs[0][0]
	dendloc = dends_locs[0][1]
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
	return recData

def test_burstresurgent(soma, dends_locs, stims):
	""" Run rebound burst experiment from original Hoc file

	EXPECTED BEHAVIOUR
	- INa_rsg slowly inactivated during long firing period
	- INa_rsg deinactivated during hyperpolarization

	TODO: 
	- test replacement of Na mechanism with Narsg modified like in Akemann to match the
	  timing of the two components of Na current
	- test shorter, more realistic hyperpolarization period (corresponding to volley of IPSPs)
	  to see if there the difference is greated between situation with and without Narsg
	"""
	# Get electrodes and sections to record from
	dendsec = dends_locs[0][0]
	dendloc = dends_locs[0][1]
	stim1, stim2, stim3 = stims[0], stims[1], stims[2]

	# Set simulation parameters
	dur = 3500
	h.dt = 0.025
	h.celsius = 35 # different temp from paper
	h.v_init = -60 # paper simulations sue default v_init
	set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

	# Set up stimulation
	stim1.delay = 0
	stim1.dur = 2000
	stim1.amp = 0.025 # evoke fast spiking -> slow inactivation

	stim2.delay = 2000
	stim2.dur = 500
	stim2.amp = -0.25 # hyperpolarizing pulse -> recovery from inactivation (deinactivation)

	stim3.delay = 2500
	stim3.dur = 1000
	stim3.amp = 0.0

	# Record
	secs = {'soma': soma, 'dend': dendsec}
	traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)
	traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}

	# Na currents
	traceSpecs['I_Na'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'ina'}
	traceSpecs['I_NaL'] = {'sec':'soma','loc':0.5,'mech':'NaL','var':'inaL'}
	# Na resurgent current related
	if resurgent:
		traceSpecs['I_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'ina'}
		# traceSpecs['sI_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'Itot'} # inactivated fraction
		# traceSpecs['sC_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'Ctot'} # closed fraction
		# traceSpecs['sB_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'B'} # blocked fraction
		# traceSpecs['sO_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'O'} # open fraction
	natrans=True
	if natrans:
		traceSpecs['sO_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'O'} # open state
		traceSpecs['sI_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'I6'} # inactivated
		traceSpecs['sC_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'Ctot'} # closed state
		traceSpecs['sCI_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'Itot'} # closed+inactivated

	# K currents
	traceSpecs['I_KDR'] = {'sec':'soma','loc':0.5,'mech':'KDR','var':'ik'}
	traceSpecs['I_Kv3'] = {'sec':'soma','loc':0.5,'mech':'Kv31','var':'ik'}
	traceSpecs['I_KCa'] = {'sec':'soma','loc':0.5,'mech':'sKCa','var':'isKCa'}
	traceSpecs['I_h'] = {'sec':'soma','loc':0.5,'mech':'Ih','var':'ih'}
	# Ca currents (soma)
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
	burst_time = [980, 1120]

	# Soma voltage
	recV = {'V_soma':recData['V_soma']}
	analysis.plotTraces(recV, recordStep)

	# Soma currents (relative)
	recI = collections.OrderedDict()
	for k, v in recData.iteritems():
		if k.startswith('I'): recI[k] = recData[k]
	analysis.cumulPlotTraces(recI, recordStep, cumulate=False)

	# Na channel states
	recStates = collections.OrderedDict()
	for k, v in recData.iteritems():
		if k.startswith('s'): recStates[k] = recData[k]
	analysis.cumulPlotTraces(recStates, recordStep, cumulate=False)

	# Overlay voltage signal
	# plt.plot(np.arange(0, dur, recordStep), recData['V_soma'].as_numpy()*1e-3, color='r')
	# plt.show(block=False)

if __name__ == '__main__':
	# Make cell
	soma, dends_locs, stims, allsecs = stn_cell(cellmodel=4)

	# Cell adjustments
	soma.Ra = 2*soma.Ra # correct incorrect calculation for Ra soma cluster

	# Adjust ionic conductances
	for sec in h.allsec():
		for seg in sec:
			seg.gpas_STh = 0.75 * seg.gpas_STh
			seg.cm = 3.0 * seg.cm
			seg.gna_NaL = 0.6 * seg.gna_NaL

	# Attach duplicate of one tree
	# from marasco_ported import dupe_subtree
	# copy_mechs = {'STh': ['gpas']} # use var gillies_gdict for full copy
	# trunk_copy = dupe_subtree(h.trunk_0, copy_mechs	, [])
	# trunk_copy.connect(soma, h.trunk_0.parentseg().x, 0)

	# Adjust passive electric ppties/scaling factors
	# surffact = 1. # f=1 => T=37 ms ; f=2 => T=64 ms
	# soma.cm = 2.0 * soma.cm
	# soma.gpas_STh = 6.0 * soma.gpas_STh
	# for gname in marasco.glist():
	# 	for seg in soma:
	# 		seg.__setattr__(gname, getattr(seg, gname)*surffact)

	# Test spontaneous firing
	# [x] simulated full model
	# [x] Simulated using average axial resistance (Marasco method)
	# [x] Simulated using conservation of Rin (no averaging of trees)
	# recData = test_spontaneous(soma, dends_locs, stims)
	
	# Test rebound burst simulator protocol
	# [x] simulated full model
	# [x] Simulated using average axial resistance (Marasco method)
	# [x] Simulated using conservation of Rin (no averaging of trees)
	recData = test_reboundburst(soma, dends_locs, stims)

	# Test generation of plateau potential
	# [x] simulated full model
	# [x] Simulated using average axial resistance (Marasco method)
	# [x] Simulated using conservation of Rin (no averaging of trees)
	# recData = test_plateau(soma, dends_locs, stims)

	# test_slowbursting()
	