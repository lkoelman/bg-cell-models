"""
Electrotonic analysis in NEURON

@author Lucas Koelman
"""

import math
from neuron import h

def lambda_DC(sec, gleak):
	"""
	Compute electrotonic length constant of section in units of micron [um]
	"""
	# Convert membrane resistance to same units as Ra
	# R_m = 1./(gleak*math.pi*sec.diam*1e-4) # r_m = R_m [Ohm*cm^2] /(pi*d) [Ohm*cm]
	R_m = 1./gleak # units [Ohm*cm^2]
	return 1e2 * math.sqrt(sec.diam*R_m/(4*sec.Ra)) # units: ([um]*[Ohm*cm^2]/(Ohm*cm))^1/2 = [um*1e2]


def lambda_AC(sec, f):
	"""
	Compute electrotonic length constant (taken from stdlib.hoc)
	"""
	return 1e5 * math.sqrt(sec.diam/(4*math.pi*f*sec.Ra*sec.cm))


def calc_lambda_AC(f, diam, Ra, cm):
	"""
	Compute electrotonic length constant
	"""
	return 1e5 * math.sqrt(diam/(4*math.pi*f*Ra*cm))


def calc_lambda(f, diam, Ra, gleak, cm):
	"""
	Compute electrotonic length constant
	"""
	if f <= 0:
		R_m = 1./gleak # units [Ohm*cm^2]
		return 1e2 * math.sqrt(diam*R_m/(4*Ra))
	else:
		return 1e5 * math.sqrt(diam/(4*math.pi*f*Ra*cm))


def sec_lambda(sec, gleak, f):
	"""
	Compute electrotonic length constant at given frequency
	"""
	if f <= 0:
		return lambda_DC(sec, gleak)
	else:
		return lambda_AC(sec, f)


def seg_lambda(seg, gleak, f):
	"""
	Compute length constant of segment
	"""
	Ra = seg.sec.Ra # Ra is section property
	if f <= 0:
		if isinstance(gleak, str):
			Rm = 1./getattr(seg, gleak)
		else:
			Rm = 1./gleak # units [Ohm*cm^2]
		return 1e2 * math.sqrt(seg.diam*Rm/(4*Ra)) # units: ([um]*[Ohm*cm^2]/(Ohm*cm))^1/2 = [um*1e2]
	else:
		return 1e5 * math.sqrt(seg.diam/(4*math.pi*f*Ra*seg.cm))


def seg_L_elec(seg, gleak, f):
	"""
	Electrotonic length of segment
	"""
	return (seg.sec.L/seg.sec.nseg)/seg_lambda(seg, gleak, f)


def min_nseg_hines(sec, f=100.):
	"""
	Minimum number of segments based on electrotonic length
	"""
	return int(sec.L/(0.1*lambda_AC(sec, f))) + 1


def min_nseg_marasco(sec):
	"""
	Minimum number of segments based on electrotonic length
	"""
	return int((sec.L/(0.1*lambda_AC(sec,100.))+0.9)/2)*2 + 1  


def calc_min_nseg_hines(f, L, diam, Ra, cm):
	lamb_AC = 1e5 * math.sqrt(diam/(4*math.pi*f*Ra*cm))
	return int(L/(0.1*lamb_AC)) + 1


def inputresistance_inf(sec, gleak, f):
	"""
	Input resistance for semi-infinite cable in units of [Ohm*1e6]
	"""
	lamb = sec_lambda(sec, gleak, f)
	R_m = 1./gleak # units [Ohm*cm^2]
	return 1e2 * R_m/(math.pi*sec.diam*lamb) # units: [Ohm*cm^2]/[um^2] = [Ohm*1e8]


def inputresistance_sealed(sec, gleak, f):
	"""
	Input resistance of finite cable with sealed end in units of [Ohm*1e6]
	"""
	x = sec.L/sec_lambda(sec, gleak, f)
	return inputresistance_inf(sec, gleak, f) * (math.cosh(x)/math.sinh(x))


def inputresistance_leaky(sec, gleak, f, R_end):
	"""
	Input resistance of finite cable with leaky end in units of [Ohm*1e6]
	
	@param R_end	input resistance of connected cables at end of section
					in units of [Ohm*1e6]
	"""
	R_inf = inputresistance_inf(sec, gleak, f)
	x = sec.L/sec_lambda(sec, gleak, f)
	return R_inf * (R_end/R_inf*math.cosh(x) + math.sinh(x)) / (R_end/R_inf*math.sinh(x) + math.cosh(x))


def inputresistance_tree(rootsec, f, glname):
	"""
	Compute input resistance to branching tree
	"""

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