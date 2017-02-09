"""
Manual reduction of Gillies & Willshaw (2006) STN neuron model


@author Lucas Koelman
@date	09-02-2017
"""

# Math modules
import math
import numpy as np

# Make sure other modules are on Python path
import sys, os.path
scriptdir, scriptfile = os.path.split(__file__)
modulesbase = os.path.normpath(os.path.join(scriptdir, '..'))
sys.path.append(modulesbase)

# Enable logging
import logging
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module

# Load NEURON
import neuron
h = neuron.h

# Our own modules
import reduction_tools as redtools
from reduction_tools import ExtSecRef, Cluster, getsecref # for convenience

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

def loadgstruct(gext):
	""" Return structured array with conductance values for given
		channel.

	@param gext	channel name/file extension in sth-data folder
	"""
	gfile = os.path.normpath(os.path.join(scriptdir, 'sth-data', 'cell-'+gext))
	gstruct = np.loadtxt(gfile, dtype={'names': ('dendidx', 'branchidx', 'x', 'g'),
									   'formats': ('i4', 'i4', 'f4', 'f4')})
	return np.unique(gstruct) # unique and sorted rows

def gillies_gstructs():
	""" Load all structured arrays with conductance values """
	gfromfile = ["gk_KDR", "gk_Kv31", "gk_Ih", "gk_sKCa", "gcaT_CaT", "gcaN_HVA", "gcaL_HVA"]
	gmats = {gname: loadgstruct(gname) for gname in gfromfile}
	return gmats

def calc_gdist_params(gname, secref, orsecrefs, path_indices=None, xgvals=None):
	""" Calculate parameters of the linear conductance distribution
		as defined in the Gillies & Willshaw paper (A/B/C/D)

	@type	orsecrefs	list(h.SectionRef)
	@param	orsecrefs	References to sections in original model. Eac SectionRef
						must have properties pathL0, pathL1, and pathL_elec
						containing the electrotonic path length up to the 0 and
						1-end, and up to each segment

	@return			tuple (L0, g0), (L1, g1), (Lstart, gstart), (Lend, gend), (gmin, gmax)
					gmin: 	min gbar along path
					gmax:	max gbar along path
					L0: 	electrotonic length of first segment with gbar > gmin
					L1:		electrotonic length of last segment with gbar > gmin
					Lstart:	electrotonic length of first segment on path
					Lend:	electrotonic length of last segment on path
	"""
	# First descent along the dendritic path and save pairs (L_elec, gval)
	if secref is None:
		return
	first_call = xgvals is None
	if first_call:
		xgvals = []
	if path_indices is None:
		path_indices = (1,2,4,6,8) # longest path in tree
	secfilter = lambda secref: secref.table_index in path_indices

	# Measure current section
	if secfilter(secref):
		for i, seg in enumerate(secref.sec):
			xgvals.append((secref.pathL_elec[i], getattr(seg, gname)))
	
	# Measure children
	for childsec in secref.sec.children():
		childref = getsecref(childsec, orsecrefs)
		calc_gdist_params(gname, childref, orsecrefs, path_indices, xgvals=xgvals) # this updates xgvals

	if first_call:
		xgvals = sorted(xgvals, key=lambda xg: xg[0]) # sort by ascending x (L_elec)
		xvals, gvals = zip(*xgvals)
		gmin = min(gvals)
		gmax = max(gvals)
		xg0 = next(xg for xg in xgvals if xg[1] > gmin, xgvals[0])
		xg1 = next(xg for xg in reversed(xgvals) if xg[1] > gmin, xgvals[-1])
		return xg0, xg1, xgvals[0], xgvals[-1], (gmin, gmax)

def interp_gbar_linear_dist(L_elec, bounds, path_bounds, g_bounds):
	""" Linear interpolation of gbar according to the given
		electrotonic path length and boundaries.

	@param	L_elec		electrotonic path length where gbar is desired

	@param	bounds		((L0, g0), (L1, g1)): electrotonic path length of
						first and last segment with gbar > gmin and their
						gbar values

	@param	path_bounds	((L_start, g_start), (L_end, g_end)): electrotonic
						path length of first and last segment on path and
						their gbar values

	@param	g_bounds	min and max gbar value
	"""
	L_a, g_a = bounds[0]
	L_b, g_b = bounds[1]
	L_min, g_start = path_bounds[0]
	L_max, g_end = path_bounds[1]
	gmin, gmax = g_bounds

	if L_a <= L_elec <= L_b:
		if L_b == L_a:
			alpha == 0.5
		else:
			alpha = (L_elec - L_a)/(L_b - L_a)
		if alpha > 1.0:
			alpha = 1.0
		if alpha < 0.0
			alpha = 0.0
		return g_a + alpha * (g_b - g_a)
	elif L_elec < L_a:
		if L_a == L_min:
			return g_a # proximal distribution, before bounds
		else:
			return g_min # distal distribution, before bounds
	else: #L_elec > L_b:
		if L_b == L_max:
			return g_b # distal distribution, after bounds
		else:
			return gmin # proximal distribution, after bounds

def findseg_L_elec(L_elec, orsecrefs):
	""" Find segments with similar electrotonic path length

	@type	L_elec		float
	@param	L_elec		electrotonic path length

	@type	orsecrefs	list(h.SectionRef)
	@param	orsecrefs	references to sections in original model with
						electrotonic path lengths to 0 and 1 ends assigned

	@return				two lists bound_segs, bound_L containing pairs of 
						boundary segments, and their electrotonic lengths
	"""
	# find original sections with L_elec(0.0) <= L_elec <= L_elec(1.0)
	or_path_secs = [secref for secref in orsecrefs if (secref.pathL0 <= L_elec <= secref.pathL1)]
	if len(or_path_secs) == 0:
		# find section where L(1.0) is closest to L_elec
		L_closest = min([abs(L_elec-secref.pathL1) for secref in orsecrefs])
		# all sections where L_elec-L(1.0) is within 5% of this
		or_path_secs = [secref for secref in orsecrefs if (0.95*L_closest <= abs(L_elec-secref.pathL1) <= 1.05*L_closest)]
		logger.debug("Electrotonic path length %f dit not map onto any original section:" + 
						" extrapolating from sections {} sections".format(len(or_path_secs)))
	logger.debug("Found {} sections in original model with same path length".format(len(or_path_secs)))

	# in each section: find segment at same elecrotonic length and average gbar
	bound_segs = [] # bounding segments
	bound_L = [] # electrotonic path length of bounding segments
	for secref in or_path_secs:
		# in each section find the two segments with L_elec(seg_a) <= L_elec <= L_elec(seg_b)
		segs_internal = [seg for seg in secref.sec]

		if L_elec <= secref.pathL_elec[0]:
			first_seg = segs_internal[0] # first segment after zero-area start node
			first_L = secref.pathL_elec[0]
			bound_segs.append((first_seg, first_seg))
			bound_L.append((first_L, first_L))

		elif L_elec >= secref.pathL_elec[-1]:
			last_seg = segs_internal[-1]
			last_L = secref.pathL_elec[-1]
			bound_segs.append((last_seg, last_seg))
			bound_L.append((last_L, last_L))

		else: # interpolate
			segs_internal = [seg for seg in secref.sec] # all sections

			if len(segs_internal) == 1: # single segment: just use midpoint
				midseg = segs_internal[0]
				midL = secref.pathL_elec[0]
				bound_segs.append((midseg, midseg))
				bound_L.append((midL, midL))

			else: # INTERPOLATE
				# Get lower bound
				lower = ((i, pathL) for i, pathL in enumerate(secref.pathL_elec) if L_elec >= pathL)
				i_a, L_a = next(lower, (0, secref.pathL_elec[0]))
				# Get higher bound
				higher = ((i, pathL) for i, pathL in enumerate(secref.pathL_elec) if L_elec <= pathL)
				i_b, L_b = next(higher, (-1, secref.pathL_elec[-1]))
				# Append bounds
				bound_segs.append((segs_internal[i_a], segs_internal[i_b]))
				bound_L.append((L_a, L_b))
	# Return pairs of boundary segments and boundary electrotonic lengths
	return bound_segs, bound_L


def interp_gbar_neighbors(L_elec, gname, bound_segs, bound_L):
	""" For each pair of boundary segments (and corresponding electrotonic
		length), do a linear interpolation of gbar in the segments according
		to the given electrotonic length. Return the average of these
		interpolated values.

	@type	gname		str
	@param	gname		full conductance name (including mechanism suffix)

	@type	bound_segs	list(tuple(Segment,Segment))
	@param	bound_segs	pairs of boundary segments

	@type	bound_L		list(tuple(float, float))
	@param	bound_segs	electrotonic lengths of boundary segments

	@return		gbar_interp: the average interpolated gbar over all boundary
				pairs
	"""
	gbar_interp = 0.0
	for i, segs in enumerate(bound_segs):
		seg_a, seg_b = segs
		L_a, L_b = bound_L[i]

		# Linear interpolation of gbar in seg_a and seg_b according to electrotonic length
		if L_elec <= L_a:
			gbar_interp += getattr(seg_a, gname)
			continue
		if L_elec >= L_b:
			gbar_interp += getattr(seg_b, gname)
			continue
		if L_b == L_a:
			alpha == 0.5
		else:
			alpha = (L_elec - L_a)/(L_b - L_a)
		if alpha > 1.0:
			alpha = 1.0 # if too close to eachother
		gbar_a = getattr(seg_a, gname)
		gbar_b = getattr(seg_b, gname)
		gbar_interp += gbar_a + alpha * (gbar_b - gbar_a)

	gbar_interp /= len(bound_segs) # take average
	return gbar_interp

def interpconductances(sec, dendidx, path, glist=None):
	""" Interpolate conductances along given path of branches 

	@type	sec			Hoc.Section()
	@param	sec			section to set conductances for

	@type	dendidx		int
	@param	dendidx		index of the dendritic tree

	@type	path		sequence of int
	@param	path		indices of branches in full model, specifying path
						along dendritic tree from soma outward

	@type	glist		dict<str, str>
	@param	glist		list of conductances to set (including mechanism suffix)
	"""

	# Load channel conductances from file
	allgmats = reduction_tools.loadgstructs()
	if glist is None:
		glist = list(gillies_glist)

	# Na & NaL are not from file
	h("default_gNa_soma = 1.483419823e-02") 
	h("default_gNa_dend = 1.0e-7")
	h("default_gNaL_soma = 1.108670852e-05")
	h("default_gNaL_dend = 0.81e-5")

	# branch indices along longest path
	geostruct = reduction_tools.loadgeotopostruct(dendidx)
	pathL = np.array([geostruct[i-1]['L'] for i in path]) # length of each branch along path

	# Distributed conductances: interpolate each conductance along longest path
	for seg in sec:
		lnode = seg.x*sum(pathL) # equivalent length along longest path

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
			if gname in glist:
				setattr(seg, gname, gval)


def setconductances(sec, dendidx, fixbranch=None, fixloc=None, glist=None):
	""" Set conductances at the node/midpoint of each segment
		by interpolating values along longest path
		(e.g. along branch 1-2-5 in dend1)

	@param dendidx		index of the denritic tree where longest path should
						be followed (1/0/-1)

	@param fixbranch	if you want to map the section to a fixed branch
						instead of following the longest path, provide its index

	@param fixloc		if you want to map all segments/nodes
						to a fixed location on the mapped branch,
						provide a location (0<=x<=1)

	@type	glist		dict<str, str>
	@param	glist		list of conductances to set (including mechanism suffix)
	"""

	# Load channel conductances from file
	allgmats = reduction_tools.loadgstructs()
	if glist is None:
		glist = list(gillies_glist)

	# Na & NaL are not from file
	h("default_gNa_soma = 1.483419823e-02")
	h("default_gNa_dend = 1.0e-7")
	h("default_gNaL_soma = 1.108670852e-05")
	h("default_gNaL_dend = 0.81e-5")

	# branch indices along longest path
	if dendidx == 1:
		geostruct = reduction_tools.loadgeotopostruct(dendidx)
		longestpath = np.array([1,2,5])
		pathL = np.array([geostruct[i-1]['L'] for i in longestpath])
	elif dendidx == 0:
		geostruct = reduction_tools.loadgeotopostruct(dendidx)
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