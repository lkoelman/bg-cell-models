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


def measure_Zin(seg, Z_freq, linearize_gating, init_func):
	"""
	Measure input impedance using NEURON Impedance tool

	See Impedance class documentation at:
	http://www.neuron.yale.edu/neuron/static/new_doc/analysis/programmatic/impedance.html

	NOTES: 

	- linearize_gating corresponds to checkbox 'include dstate/dt contribution' in NEURON GUI Impedance Tool
		- (see equations in Impedance doc) 
		- 0 = calculation with current values of gating vars
		- 1 = linearize gating vars around V
	"""
	# Initialize the cell
	if init_func is not None:
		init_func()
	
	imp = h.Impedance() # imp = h.zz # previously created
	imp.loc(seg.x, sec=seg.sec) # injection site
	imp.compute(Z_freq, int(linearize_gating)) # compute A(x->loc) for all x where A is Vratio/Zin/Ztransfer

	Zin = imp.input(seg.x, sec=seg.sec)
	return Zin


def segs_at_dX(cur_seg, dX, f, gleak):
	"""
	Get next segments (in direction 0 to 1 end) at electrotonic distance dX
	from current segment (dX = L/lambda).

	I.e. get the segment at X(cur_seg) + dX

	Assumes entire section has same diameter
	"""
	nseg = cur_seg.sec.nseg
	sec_L = cur_seg.sec.L
	sec_dL = sec_L / nseg
	sec_lamb = [seg_lambda(seg, gleak, f) for seg in cur_seg.sec]
	sec_Xi = [sec_dL / sec_lamb[i] for i in xrange(nseg)]
	sec_Xtot = sum(sec_Xi) # total L/lambda
	sec_Xacc = [sum((sec_Xi[j] for j in xrange(i)), 0.0) for i in xrange(nseg)] # accumulated X to left border
	sec_dx = 1.0/nseg

	# Get electrotonic length from start of Section to current x
	cur_i = min(int(cur_seg.x/sec_dx), nseg-1)
	l_x = cur_i*sec_dx		# x of left boundary
	l_X = sec_Xacc[cur_i]	# L/lambda of left boundary

	frac = (cur_seg.x - l_x) / sec_dx
	assert 0.0 <= frac <= 1.01
	frac = min(1.0, frac)
	cur_X = l_X + frac*sec_Xi[i]

	# Get length to next discretization step
	bX = cur_X + dX
	next_segs = []

	if bX <= sec_Xtot:

		# Find segment that X falls in
		i_b = next((i for i, Xsum in enumerate(sec_Xacc) if Xsum+sec_Xi[i] >= bX))
		
		# Interpolate
		frac = (bX - sec_Xacc[i_b]) / sec_Xi[i_b]
		assert 0.0 <= frac <= 1.01
		frac = min(1.0, frac)
		
		b_x = i_b*sec_dx + frac*sec_dx
		next_segs.append(cur_seg.sec(b_x))

	else:
		ddX = bX - sec_Xtot # how far to advance in next section
		next_segs = []
		for child_sec in cur_seg.sec.children():
			next_segs.extend(segs_at_dX(child_sec(0.0), ddX, f, gleak))

	return next_segs