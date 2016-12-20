"""
Reduce Gillies & Willshaw (2006) STN neuron model to two-compartment model


@author Lucas Koelman
@date	03-11-2016
@note	must be run from script directory or .hoc files not found

"""

import numpy as np
import matplotlib.pyplot as plt
import math

import neuron
from neuron import h

# Load NEURON function libraries
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

def lambda_DC(sec, gleak):
	""" Compute electrotonic length in units of micron [um]"""
	# Convert membrane resistance to same units as Ra
	# R_m = 1./(gleak*math.pi*sec.diam*1e-4) # r_m = R_m [Ohm*cm^2] /(pi*d) [Ohm*cm]
	R_m = 1./gleak # units [Ohm*cm^2]
	return 1e2 * math.sqrt(sec.diam*R_m/(4*sec.Ra)) # units: ([um]*[Ohm*cm^2]/(Ohm*cm))^1/2 = [um*1e2]

def lambda_AC(sec, f):
	""" Compute electrotonic length (taken from stdlib.hoc) """
	return 1e5*math.sqrt(sec.diam/(4*math.pi*f*sec.Ra*sec.cm))

def electrotonic_length(sec, gleak, f):
	if f <= 0:
		return lambda_DC(sec, gleak)
	else:
		return lambda_AC(sec, f)

def inputresistance_inf(sec, gleak, f):
	""" Input resistance for semi-infinite cable in units of [Ohm*1e6] """
	lamb = electrotonic_length(sec, gleak, f)
	R_m = 1./gleak # units [Ohm*cm^2]
	return 1e2 * R_m/(math.pi*sec.diam*lamb) # units: [Ohm*cm^2]/[um^2] = [Ohm*1e8]

def inputresistance_sealed(sec, gleak, f):
	""" Input resistance of finite cable with sealed end in units of [Ohm*1e6] """
	x = sec.L/electrotonic_length(sec, gleak, f)
	return inputresistance_inf(sec, gleak, f) * (math.cosh(x)/math.sinh(x))

def inputresistance_leaky(sec, gleak, f, R_end):
	""" Input resistance of finite cable with leaky end in units of [Ohm*1e6]
	@param R_end	input resistance of connected cables at end of section
					in units of [Ohm*1e6]
	"""
	R_inf = inputresistance_inf(sec, gleak, f)
	x = sec.L/electrotonic_length(sec, gleak, f)
	return R_inf * (R_end/R_inf*math.cosh(x) + math.sinh(x)) / (R_end/R_inf*math.sinh(x) + math.cosh(x))

def inputresistance_tree(rootsec, f, glname):
	""" Compute input resistance to branching tree """
	childsecs = rootsec.children()
	gleak = getattr(rootsec, glname)

	# Handle leaf sections
	if not any(childsecs):
		return inputresistance_sealed(rootsec, gleak, f)

	# Calc input conductance of children
	g_end = 0.
	for childsec in childsecs:
		g_end += 1./inputresistance_tree(childsec, f, glname)
	return inputresistance_leaky(rootsec, gleak, f, 1./g_end)

if __name__ == '__main__':
	plotconductances(treestruct()[1], 1, loadgstruct('gcaT_CaT'), includebranches=[1,2,5])
	# plotchanneldist(0, 'gcaL_HVA')
	# dend0tree, dend1tree = treechannelstruct()
	# gtstruct = loadgeotopostruct(0)