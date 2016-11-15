"""
Reduce Gillies & Willshaw (2006) STN neuron model to two-compartment model


@author Lucas Koelman
@date	03-11-2016
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

class Tree:
	""" A binary tree node """
	def __init__(self, payload, left=None, right=None):
		self.payload = payload
		self.left  = left
		self.right = right

	def __str__(self):
		return str(self.payload)

def treestruct():
	""" Return the two dendritic tree structures from the paper
		
		The 'index' refers to the branch/section indices in the
		.dat files that come with the paper. To get a reference to
		the corresponding section, use index-1 in the Vectors stored
		in the SThcell object.

		e.g. in tree1: branch5 = h.SThcell[0].dend1[4]
	"""
	# Tree structure (map structure to indices in tree1-nom.dat)
	## Fig. 1, right dendritic tree
	dend1tree = Tree({'index':1}, 
					Tree({'index':2}, 
						Tree({'index':4}, Tree({'index':6}), Tree({'index':7})), 
						Tree({'index':5})), 
					Tree({'index':3}, 
						Tree({'index':8}), 
						Tree({'index':9}, Tree({'index':10}), Tree({'index':11}))))

	## Fig. 1, left dendritic tree
	dend0upper = Tree({'index':2}, 
					Tree({'index':4}, 
						Tree({'index':6}, Tree({'index':8}), Tree({'index':9})), 
						Tree({'index':7})), 
					Tree({'index':5}, 
						Tree({'index':10}), 
						Tree({'index':11}, Tree({'index':12}), Tree({'index':13}))))

	dend0lower = Tree({'index':3}, 
					Tree({'index':14}, 
						Tree({'index':16}, Tree({'index':18}), Tree({'index':19})), 
						Tree({'index':17})), 
					Tree({'index':15}, 
						Tree({'index':20}), 
						Tree({'index':21}, Tree({'index':22}), Tree({'index':23}))))

	dend0tree = Tree({'index':1}, dend0upper, dend0lower)
	return dend0tree, dend1tree

def loadgeotopostruct(dendidx):
	""" Load geometry/topology matrix from file """
	dendfile = os.path.normpath(os.path.join(scriptdir, 'sth-data', 'tree{0}-nom.dat'.format(dendidx)))
	gtstruct = np.loadtxt(dendfile, dtype={'names': ('branchidx', 'leftidx', 'rightidx', 'diam', 'L', 'nseg'),
										'formats': ('i4', 'i4', 'i4', 'f4', 'f4', 'i4')})
	return np.unique(gtstruct) # unique and sorted rows (indexable by branch unlike file!)

def loadgstruct(gext):
	""" Return structured array with conductance values for given
		channel.
		@param gext	channel name/file extension in sth-data folder
	"""
	gfile = os.path.normpath(os.path.join(scriptdir, 'sth-data', 'cell-'+gext))
	gstruct = np.loadtxt(gfile, dtype={'names': ('dendidx', 'branchidx', 'x', 'g'),
									   'formats': ('i4', 'i4', 'f4', 'f4')})
	return np.unique(gstruct) # unique and sorted rows

def loadgstructs():
	""" Load all structured arrays with conductance values """
	allgnames = ["gk_KDR", "gk_Kv31", "gk_Ih", "gk_sKCa", "gcaT_CaT", "gcaN_HVA", "gcaL_HVA"]
	gmats = {gname: loadgstruct(gname) for gname in allgnames}
	return gmats

def treechannelstruct():
	""" Return tree structure containing channel distributions """

	# Load conductance matrices
	gstructs = loadgstructs()

	# Initialize tree structure
	dend0tree, dend1tree = treestruct()

	def filltreegstruct(tree, dendidx):
		if tree is None: return

		# Fill this branch/section
		branchidx = tree.payload['index']
		for gname, gmat in gstructs:
			branchrows = (gmat['dendidx']==dendidx) & (gmat['branchidx']==branchidx-1)
			tree.payload['x'+gname] = gmat[branchrows]['x']
			tree.payload[gname] = gmat[branchrows]['g']

		# Fill left and right children
		filltreegstruct(tree.left, dendidx)
		filltreegstruct(tree.right, dendidx)

	# Fill the two dendritic trees
	filltreegstruct(dend0tree, 0)
	filltreegstruct(dend1tree, 1)
	return dend0tree, dend1tree

def lambda_f(f, diam, Ra, cm):
	""" Compute electrotonic length (taken from stdlib.hoc) """
	return 1e5*np.sqrt(diam/(4*np.pi*f*Ra*cm))

def check32rule(d1, d2, d3):
	print('d1^(3/2) = {0}'.format(d1**(3./2.)))
	print('d2^(3/2) + d3^(3/2) = {0}'.format(d2**(3./2.)+d3**(3./2.)))

def getbranchproperties(branchidx, dendsecs):
	""" Return branch properties for the given branch index """
	secidx = branchidx-1

	# section properties
	L = dendsecs[secidx].L
	Ra = dendsecs[secidx].Ra

	# segment properties
	diam = dendsecs[secidx](0.5).diam
	cm = dendsecs[secidx](0.5).cm

	# Electrotonic length
	lambda100 = 1e5*np.sqrt(diam/(4*np.pi*100*Ra*cm))
	l_elec = L/lambda100

	return L, Ra, diam, cm, lambda100, l_elec

def printproperties(tree, dendsecs):
	""" Recursive depth first traversal of tree """
	if tree is None: return 0

	# Get branch properties
	branchidx = tree.payload['index']
	L, Ra, diam, cm, lambda100, l_elec = getbranchproperties(branchidx, dendsecs)

	# Print branch properties
	print("Branch index {0}: L={1}; Ra={2}; diam(0.5)={3}; cm(0.5)={4}; lambda100={5}; l_elec={6}".format(
		branchidx, L, Ra, diam, cm, lambda100, l_elec))
	printproperties(tree.left, dendsecs)
	printproperties(tree.right, dendsecs)

def reducebranch(tree, dendsecs):
	""" Reduce thre tree according to Rall algorithm
	@return		the equivalent electrotonic length of the tree 
				(i.e. that should be added to parent tree of this branch)
	@return		diameter^(3/2) of the given tree branch

	The following properties must be met:
		1. Rm and Ra must be the same in parent + children
		2. Same boundary conditions on all branches
		3. Child branches have same electrotonic length
		4. 3/2 diameter rule satisfied betweeen parents - children
	Properties 1 & 2 are satisfied according to model specification, 
	so only 3 & 4 must be checked.

	NOTE: if tree is the parent tree, the equivalent length is the 
	      returned value times its lambda100
	"""
	if tree is None: return 0, 0

	# Get own electrotonic length and diameter
	branchidx = tree.payload['index']
	L, Ra, diam, cm, lambda100, l_elec = getbranchproperties(branchidx, dendsecs)
	selfd32 = diam**(3./2.)

	# Get equivalent electrotonic length of children
	left_elec, leftd32 = reducebranch(tree.left, dendsecs)
	right_elec, rightd32 = reducebranch(tree.right, dendsecs)

	# Check if they are (almost) the same
	if (left_elec==0 and right_elec==0) or (left_elec/right_elec >= 0.95 and left_elec/right_elec <= 1.05):
		print('OK: electrotonic length of child branches is within 5%% of eachother')
	else:
		print('WARNING: electrotonic length of branches to be merged is not close enough!')
	d32ratio = (leftd32+rightd32)/selfd32
	if (leftd32==0 and rightd32==0) or (d32ratio >= 0.95 and d32ratio <= 1.05):
		print('OK: 3/2 rule satisfied within 5%% accuracy')
	else:
		print('WARNING: 3/2 rule not satisfied within 5%% accuracy" ratio={0}'.format(d32ratio))

	return ((left_elec+right_elec)/2.0 + l_elec), selfd32

def reduce_dends():
	""" Measure electrical/geometrical ppties and reduce dendritic
		trees using Rall algorithm """

	# Create cell and three IClamp in soma
	h.xopen("createcell.hoc")
	somasec = h.SThcell[0].soma
	dend0secs = h.SThcell[0].dend0
	dend1secs = h.SThcell[0].dend1
	dend0tree, dend1tree = treestruct()
	

	# Check from properties if reducable (checked by hand: YES)
	print('== Checking right dendritic tree (smaller one) ===')
	printproperties(dend1tree, dend1secs)

	# equivalent length
	print('== Reducing right dendritic tree (smaller one) ===')
	L, Ra, diam, cm, lambda_root, l_elec = getbranchproperties(1, dend1secs)
	l_equiv, _ = reducebranch(dend1tree, dend1secs)
	L_equiv = l_equiv * lambda_root
	print('Equivalent length is {0}'.format(L_equiv)) # 549.86

	# LEFT TREE
	print('== Checking left dendritic tree (larger one) ===')
	printproperties(dend0tree, dend0secs)

	print('== Reducing left dendritic tree (larger one) ===')
	L, Ra, diam, cm, lambda_root, l_elec = getbranchproperties(1, dend0secs)
	l_equiv, _ = reducebranch(dend0tree, dend0secs)
	L_equiv = l_equiv * lambda_root
	print('Equivalent length is {0}'.format(L_equiv)) # 703.34

def reduce_onedend():
	""" Reduce everything to one dendrite 

	ALGORITHM:
		- use algorithm Sterrat p. 85-86 to compute equivalent imput resistance
		  of two trees in parallel
		- the equivalent Rin has two free parameters: d and L. Use d of the 
		  largest tree and calculate L from this and the equivalent resistance value.
	"""
	# use min. amount of sections to reproduce behaviour
	pass


def plotchanneldist(gext):
	""" Plot channel distributions from cell_g<xyz> files 
	@param gext		conductance specifier/file extenstion, e.g. 'gcaL_HVA'
	"""
	# Load conductance distribution from supplied file
	scriptdir, scriptfile = os.path.split(__file__)
	gfile = os.path.normpath(os.path.join(scriptdir, 'sth-data', 'cell-'+gext))
	gmat = np.loadtxt(gfile)

	dend0tree, dend1tree = treestruct()

	def plotconductances(tree, treeidx, includebranches=None):
		if tree is None: return

		# Get branch properties
		branchidx = tree.payload['index']

		# Plot own distribution
		if includebranches is None or branchidx in includebranches:
			plt.figure()
			branchrows = (gmat[:,0]==treeidx) & (gmat[:,1]==branchidx-1)
			x = gmat[branchrows,2]
			g = gmat[branchrows,3]
			plt.plot(x,g,'o')
			plt.xlabel('x (normalized length)')
			plt.ylabel(gext)
			plt.suptitle('Condcutance distribution in branch {}'.format(branchidx))
			plt.show(block=False)

			# print it
			print('## branch {} conductances ##'.format(branchidx))
			print('\t'.join((str(l) for l in x)))
			print('\t'.join((str(gb) for gb in g)))

		plotconductances(tree.left, treeidx, includebranches)
		plotconductances(tree.right, treeidx, includebranches)

	# Plot it
	plotconductances(dend1tree, 1, includebranches=[1,2,5])


if __name__ == '__main__':
	# plotchanneldist('gcaL_HVA')
	# dend0tree, dend1tree = treechannelstruct()
	gtstruct = loadgeotopostruct(0)