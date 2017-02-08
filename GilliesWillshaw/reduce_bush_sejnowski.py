"""
Reduce Gillies & Willshaw (2006) STN neuron model using the method
described in Bush & Sejnowski (1993)


@author Lucas Koelman
@date	02-02-2017
"""

# Python modules
import math
import pickle

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
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library
# Load own NEURON mechanisms
NRN_MECH_PATH = os.path.normpath(os.path.join(scriptdir, 'nrn_mechs'))
neuron.load_mechanisms(NRN_MECH_PATH)

# Our own modules
import reduction_tools as redtools
from reduction_tools import ExtSecRef, Cluster, getsecref # for convenience
import reduction_analysis as analysis

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

def interp_gbar(L_elec, gname, bound_segs, bound_L):
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

def equivalent_sections(clusters, orsecrefs, f_lambda, use_segments=True, area_scaling=True):
	""" Compute properties of equivalent section for cluster
		from its member sections.

	@type	clusters	list(Cluster)
	@param	clusters	list of Cluster objects (containing cluster label 
						and properties)

	@type	orsecrefs	list(SectionRef)
	@param	orsecrefs	references to all  sections in the cell

	@type	area_scaling	bool
	@param	area_scaling	If true: use the ratio of original area to new area
							to scale Cm and all conductances (including gpas) in
							each section so their total value from the full model
							is conserved

	@post				Equivalent section properties are available as
						attributed on given Cluster object

	@return	eq_refs		list(SectionRef) with references to equivalent section
						for each cluster (in same order as param clusters)
	"""
	# List with equivalent section for each cluster
	eq_secs = [None] * len(clusters)

	# Calculate cluster properties
	logger.debug("Calculating cluster properties...")
	for cluster in clusters:
		if use_segments: # SEGMENT-BASED CLUSTERING
			# Gather member segments
			clu_filter = lambda secref, iseg: secref.cluster_labels[iseg] == cluster.label
			clu_segs = [seg for ref in orsecrefs for i, seg in enumerate(ref.sec) if clu_filter(ref, i)]

			# Compute equivalent properties
			cluster.eqL = sum(seg.sec.L/seg.sec.nseg for seg in clu_segs) / len(clu_segs)
			cluster.eqdiam = math.sqrt(sum(seg.diam**2 for seg in clu_segs))
			cluster.eqRa = sum(seg.sec.Ra for seg in clu_segs) / len(clu_segs)
			cluster.or_area = sum(seg.area() for seg in clu_segs)
			cluster.or_cmtot = sum(seg.cm*seg.area() for seg in clu_segs)
			cluster.or_cm = cluster.or_cmtot / cluster.or_area
			cluster.or_gtot = dict((gname, 0.0) for gname in glist) # sum of gbar multiplied by area
			for gname in glist:
				cluster.or_gtot[gname] += sum(getattr(seg, gname)*seg.area() for seg in clu_segs)

		else: # SECTION-BASED CLUSTERING
			# Gather member sections
			clu_secs = [secref for secref in orsecrefs if secref.cluster_label==cluster.label]
			logger.debug("Cluster %s contains %i sections" % (cluster.label, len(clu_secs)))

			# Compute equivalent properties
			cluster.eqL = sum(secref.sec.L for secref in clu_secs) / len(clu_secs)
			cluster.eqdiam = math.sqrt(sum(secref.sec.diam**2 for secref in clu_secs))
			cluster.eqRa = sum(secref.sec.Ra for secref in clu_secs) / len(clu_secs)
			cluster.or_area = sum(sum(seg.area() for seg in secref.sec) for secref in clu_secs)
			cluster.or_cmtot = sum(sum(seg.cm*seg.area() for seg in secref.sec) for secref in clu_secs)
			cluster.or_gtot = dict((gname, 0.0) for gname in glist) # sum of gbar multiplied by area
			for gname in glist:
				cluster.or_gtot[gname] += sum(sum(getattr(seg, gname)*seg.area() for seg in secref.sec) for secref in clu_secs)
	logger.debug("Done calculating cluster properties.\n\n")

	# Create equivalent sections and passive electric structure
	logger.debug("Building passive section topology...")
	for i, cluster in enumerate(clusters):
		# Create equivalent section
		if cluster.label in [sec.name() for sec in h.allsec()]:
			raise Exception('Section named {} already exists'.format(cluster.label))
		h("create %s" % cluster.label)
		sec = getattr(h, cluster.label)

		# Set geometry 
		sec.L = cluster.eqL
		sec.diam = cluster.eqdiam
		cluster.eq_area = sum(seg.area() for seg in sec) # should be same as cluster eqSurf

		# Passive electrical properties (except Rm/gleak)
		sec.Ra = cluster.eqRa

		# Append to list of equivalent sections
		eq_secs[i] = sec

		logger.debug("Summary for cluster '%s' : L=%f \tdiam=%f \tRa=%f" % (cluster.label,
							cluster.eqL, cluster.eqdiam, cluster.eqRa))

	# Connect equivalent sections
	for i, clu_i in enumerate(clusters):
		for j, clu_j in enumerate(clusters):
			if clu_j is not clu_i and clu_j.parent_label == clu_i.label:
				eq_secs[j].connect(eq_secs[i], clu_j.parent_pos, 0)

	# Set active properties and finetune
	for i, cluster in enumerate(clusters):
		logger.debug("Scaling properties of cluster %s ..." % clusters[i].label)
		sec = eq_secs[i]

		# Insert all mechanisms
		for mech in mechs_chans.keys():
			sec.insert(mech)

		# Scale passive electrical properties
		area_ratio = cluster.or_area / cluster.eq_area
		logger.debug("Ratio of areas is %f" % area_ratio)

		# Scale Cm
		eq_cm1 = cluster.or_cm * area_ratio
		eq_cm2 = cluster.or_cmtot / cluster.eq_area # more accurate than cm * or_area/eq_area
		eq_cm = eq_cm2
		logger.debug("Cm scaled by ratio is %f (equivalently, cmtot/eq_area=%f)" % (eq_cm1, eq_cm2))

		# Scale Rm
		or_gleak = cluster.or_gtot[gleak_name] / cluster.or_area
		eq_gleak = or_gleak * area_ratio # same as reducing Rm by area_new/area_old
		logger.debug("gleak scaled by ratio is %f (old gleak is %f)" % (eq_gleak, or_gleak))

		# Set number of segments based on rule of thumb electrotonic length
		sec.nseg = redtools.calc_min_nseg_hines(100., sec.L, sec.diam, sec.Ra, eq_cm)

		# Save Cm and conductances for each section for reconstruction
		cluster.nseg = sec.nseg # save for reconstruction
		cluster.eq_gbar = dict((gname, [float('NaN')]*cluster.nseg) for gname in glist)
		cluster.eq_cm = [float('NaN')]*cluster.nseg

		# Set Cm and gleak (Rm) for each segment
		if area_scaling:
			for j, seg in enumerate(sec):
				setattr(seg, 'cm', eq_cm)
				setattr(seg, gleak_name, eq_gleak)
				cluster.eq_cm[j] = eq_cm
				cluster.eq_gbar[gleak_name][j] = eq_gleak

		# Get active conductances
		active_glist = list(glist)
		active_glist.remove(gleak_name) # get list of active conductances

		# Set initial conductances by interpolation
		for j, seg in enumerate(sec):
			L_elec = redtools.seg_path_L_elec(seg, f_lambda, gleak_name)
			bound_segs, bound_L = findseg_L_elec(L_elec, orsecrefs)
			for gname in active_glist:
				gval = interp_gbar(L_elec, gname, bound_segs, bound_L)
				seg.__setattr__(gname, gval)
				cluster.eq_gbar[gname][j] = gval

		# Re-scale gbar distribution to yield same total gbar (sum(gbar*area))
		if area_scaling:
			for gname in active_glist:
				eq_gtot = sum(getattr(seg, gname)*seg.area() for seg in sec)
				if eq_gtot <= 0.:
					eq_gtot = 1.
				or_gtot = cluster.or_gtot[gname]
				for j, seg in enumerate(sec):
					# conserves gtot_or since: sum(g_i*area_i * or_area/eq_area) = or_area/eq_area * sum(gi*area_i) ~= or_area/eq_area * g_avg*eq_area = or_area*g_avg
					# seg.__setattr__(gname, getattr(seg, gname)*clusters[i].or_area/clusters[i].eq_area)
					gval = getattr(seg, gname) * or_gtot/eq_gtot
					seg.__setattr__(gname, gval)
					cluster.eq_gbar[gname][j] = gval # save for reconstruction

		# Check gbar calculation
		for gname in active_glist:
			gtot_or = cluster.or_gtot[gname]
			gtot_eq_scaled = sum(getattr(seg, gname)*seg.area() for seg in sec)
			logger.debug("conductance %s : gtot_or = %.3f ; gtot_eq = %.3f",
							gname, gtot_or, gtot_eq_scaled)

		# Debugging info:
		logger.debug("Created equivalent section '%s' with \n\tL\tdiam\tcm\tRa\tnseg\
		\n\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n\n", clusters[i].label, sec.L, sec.diam, sec.cm, sec.Ra, sec.nseg)
	
	return eq_secs

def rebuild_sections(clusters):
	""" Build the reduced model from a previous reduction where the equivalent
		section properties are stored in each Cluster object.
	"""
	eq_secs = [None] * len(clusters)

	# Create equivalent section for each cluster
	for i, cluster in enumerate(clusters):
		# Create equivalent section
		if cluster.label in [sec.name() for sec in h.allsec()]:
			raise Exception('Section named {} already exists'.format(cluster.label))
		h("create %s" % cluster.label)
		sec = getattr(h, cluster.label)
		eq_secs[i] = sec

		# Set geometry & global passive properties
		sec.L = cluster.eqL
		sec.diam = cluster.eqdiam
		sec.Ra = cluster.eqRa
		sec.nseg = cluster.nseg
		logger.debug("Summary for cluster '%s' : L=%f \tdiam=%f \tRa=%f" % (cluster.label,
							cluster.eqL, cluster.eqdiam, cluster.eqRa))

		# Insert all mechanisms
		for mech in mechs_chans.keys():
			sec.insert(mech)
		# Get active conductances
		active_glist = list(glist)
		active_glist.remove(gleak_name) # get list of active conductances

		# Set ion styles
		sec.push()
		h.ion_style("na_ion",1,2,1,0,1)
		h.ion_style("k_ion",1,2,1,0,1)
		h.ion_style("ca_ion",3,2,1,1,1)
		h.pop_section()

		# Set Cm and conductances for each section (RANGE variables)
		for j, seg in enumerate(sec):
			# passive properties
			setattr(seg, 'cm', cluster.eq_cm[j])
			setattr(seg, gleak_name, cluster.eq_gbar[gleak_name][j])
			# active conductances
			for gname in active_glist:
				if len(cluster.eq_gbar[gname] != cluster.nseg):
					raise Exception("Number of gbar values does not match number of segments")
				if any(math.isnan(g) for g in cluster.eq_gbar[gname]):
					raise Exception("Conductance vector contains NaN values")
				seg.__setattr__(gname, cluster.eq_gbar[gname][j])

	# Connect equivalent sections
	for i, clu_i in enumerate(clusters):
		for j, clu_j in enumerate(clusters):
			if clu_j is not clu_i and clu_j.parent_label == clu_i.label:
				eq_secs[j].connect(eq_secs[i], clu_j.parent_pos, 0)
	return eq_secs

def save_clusters(clusters, filepath):
	""" Save list of Cluster objects to file """
	clu_file = open(filepath, "wb")
	try:
		pickle.dump(clusters, clu_file)
	finally:
		clu_file.close()

def load_clusters(filepath):
	""" Load list of Cluster objects from file """
	clu_file = open(filepath, "rb")
	try:
		clusters = pickle.load(clu_file)
		return clusters
	finally:
		clu_file.close()

def reduce_bush_sejnowski():
	""" Reduce STN cell according to Bush & Sejnowski (2016) method """

	############################################################################
	# 0. Load full model to be reduced (Gillies & Willshaw STN)
	for sec in h.allsec():
		if not sec.name().startswith('SThcell'): # original model sections
			h.delete_section()
	if not hasattr(h, 'SThcells'):
		h.xopen("createcell.hoc")

	# Make sections accesible by name and index
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendLrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0] # 0 is left tree
	dendRrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1] # 1 is right tree
	allsecrefs = [somaref] + dendLrefs + dendRrefs
	for j, dendlist in enumerate((dendLrefs, dendRrefs)):
		for i, secref in enumerate(dendlist):
			secref.table_tree = j
			secref.table_index = i+1

	############################################################################
	# 1. Cluster into functional regions/according to electrotonic length

	# Compute properties used for clustering
	logger.info("Computing electrotonic path lengths...")
	f_lambda = 100. # frequency for electrotonic length
	redtools.assign_electrotonic_length(somaref, allsecrefs, f_lambda, 
										gleak_name, allseg=True)

	# Cluster soma
	somaclu = Cluster('soma')
	somaref.cluster_label = 'soma'
	somaref.cluster_labels = ['soma'] * somaref.sec.nseg
	somaref.table_index = 0
	clusters = [somaclu]

	# Cluster segments in each dendritic tree
	thresholds = (0.4, 1.0) # (0.4, 1.0) determine empirically for f=100Hz
	logger.info("Clustering left tree (dend0)...")
	redtools.clusterize_sec_electrotonic(dendLrefs[0], allsecrefs, thresholds, clusters)

	logger.info("Clustering right tree (dend1)...")
	redtools.clusterize_sec_electrotonic(dendRrefs[0], allsecrefs, thresholds, clusters)

	# Determine cluster topology
	redtools.assign_topology(clusters, ['soma', 'trunk', 'smooth', 'spiny'])

	############################################################################
	# 2. Create equivalent sections (compute ppties from cluster sections)

	# Create new sections
	logger.info("Creating equivalent sections...")
	eq_secs = equivalent_sections(clusters, allsecrefs, f_lambda, 
				use_segments=False, area_scaling=True)
	
	# Delete original model sections & set ion styles
	for sec in h.allsec(): # makes each section the CAS
		if sec.name().startswith('SThcell'): # original model sections
			h.delete_section()
		else: # equivalent model sections
			h.ion_style("na_ion",1,2,1,0,1)
			h.ion_style("k_ion",1,2,1,0,1)
			h.ion_style("ca_ion",3,2,1,1,1)

	# Plot some conductances (compare with full model)
	# eq_refs = [ExtSecRef(sec=sec) for sec in eq_secs]
	# eq_somaref = next(ref for ref in eq_refs if ref.sec.name().startswith('soma'))
	# analysis.plot_tree_ppty(eq_somaref, eq_refs, 'gk_sKCa', 
	# 			lambda ref:True, lambda ref:ref.sec.name())

	############################################################################
	# 3. Finetuning/fitting of active properties

	# TODO: fitting using neurotune
	return clusters, eq_secs

if __name__ == '__main__':
	reduce_bush_sejnowski()
