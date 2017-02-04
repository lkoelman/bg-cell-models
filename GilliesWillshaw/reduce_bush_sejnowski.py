"""
Reduce Gillies & Willshaw (2006) STN neuron model using the method
described in Bush & Sejnowski (1993)


@author Lucas Koelman
@date	02-02-2016
"""

# Python modules
import math

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

def interp_gbar_electrotonic(tarseg, gname, gleak_name, f, orsecrefs):
	""" Interpolate channel density (max conductance) according
		to electrotonic path.

	@type	tarseg			Segment
	@param	tarseg			segment in reduced model for which gbar is desired

	@type	gname		str
	@param	gname		full conductance name (including mechanism suffix)

	@type	gleak_name	str
	@param	gleak_name	name of leak conductance

	@type	f			float
	@param	f			frequency for calculating electrotonic length

	@type	orsecrefs	list(h.SectionRef)
	@param	orsecrefs	references to sections in original model with
						electrotonic path lengths to 0 and 1 ends assigned

	@return				mean max conductance at equivalent electrotonic
						path lengths in full model

	ALGORITHM
	- calculate electrotonic path length L_elec of seg in reduced model
	- in each original section where L0 <= L <= L1: find segment by linear interpolation
	- take average gbar of all sections in original model that meet this criterion
	"""
	# calculate electrotonic path length of given segment
	L_elec = redtools.seg_path_L_elec(tarseg, f, gleak_name)

	# find original sections with similar electrotonic path length
	or_path_secs = [secref for secref in orsecrefs if (secref.pathL0 <= L_elec <= secref.pathL1)]
	logger.debug("Found {} sections in original model with same path length".format(len(or_path_secs)))

	# find corresponding segments and compute average gbar
	gbar_interp = 0.0
	for secref in or_path_secs:
		if L_elec <= secref.pathL0:
			gbar_interp += getattr(secref.sec(0.0), gname)
		elif L_elec >= secref.pathL1:
			gbar_interp += getattr(secref.sec(1.0), gname)
		else: # interpolate
			segs = [seg for seg in secref.sec] # all sections

			if len(segs) == 1:
				# single segment: just use midpoint
				gbar_interp += getattr(secref.sec(0.5), gname)

			else: # INTERPOLATE
				# Get lower bound
				lower = ((i, pathL) for i, pathL in enumerate(secref.pathL_elec) if L_elec >= pathL)
				i_a, L_a = next(lower, (0, secref.pathL_elec[0]))
				if L_elec <= L_a:
					gbar_interp += getattr(segs[i_a], gname)
					continue

				# Get higher bound
				higher = ((i, pathL) for i, pathL in enumerate(secref.pathL_elec) if L_elec <= pathL)
				i_b, L_b = next(higher, (-1, secref.pathL_elec[-1]))
				if L_elec >= L_b:
					gbar_interp += getattr(segs[i_b], gname)
					continue

				# case L_a < L_elec < L_b
				if L_b == L_a:
					delta == 0.5
				else:
					delta = (L_elec - L_a)/(L_b - L_a)
				if delta > 1.0:
					delta = 1.0 # if too close to eachother
				gbar_a = getattr(segs[i_a], gname)
				gbar_b = getattr(segs[i_b], gname)
				gbar_interp += (gbar_a + delta * (gbar_b - gbar_a))

	gbar_interp /= len(or_path_secs) # take average

	# find corresponding segments and compute average gbar
	# gbar_interp = 0.0
	# for secref in or_path_secs:
	# 	x_path = (L_elec - secref.pathL0)/(secref.pathL1 - secref.pathL0)
	# 	or_path_seg = secref.sec(x_path) # segment at corresponding electrotonic length
	# 	gbar_interp += getattr(or_path_seg, gname)
	# gbar_interp /= len(or_path_secs)

	return gbar_interp

def equivalent_sections(clusters, orsecrefs, f_lambda, use_segments=True):
	""" Compute properties of equivalent section for cluster
		from its member sections.

	@type	clusters	list(Cluster)
	@param	clusters	list of Cluster objects (containing cluster label 
						and properties)

	@type	orsecrefs	list(SectionRef)
	@param	orsecrefs	references to all  sections in the cell

	@post				Equivalent section properties are available as
						attributed on given Cluster object
	"""
	# List with equivalent section for each cluster
	eq_secs = [None] * len(clusters)
	eq_secrefs = [None] * len(clusters)

	# Calculate cluster properties
	logger.debug("Calculating cluster properties...")
	for cluster in clusters:
		if use_segments: # SEGMENT-BASED CLUSTERING
			# Gather member segments
			clu_filter = lambda secref, iseg: secref.cluster_labels[iseg] == cluster.label
			clu_segs = [seg for ref in orsecrefs for i, seg in enumerate(ref.sec) if clu_filter(ref, i)]

			# Compute equivalent properties
			cluster.eqL = math.sqrt(sum(seg.sec.L/seg.sec.nseg for seg in clu_segs)) / len(clu_segs)
			cluster.eqdiam = math.sqrt(sum(seg.diam**2 for seg in clu_segs))
			cluster.eqRa = sum(seg.sec.Ra for seg in clu_segs)/len(clu_segs)
			cluster.or_area = sum(seg.area() for seg in clu_segs)
			cluster.or_cmtot = sum(seg.cm*seg.area() for seg in clu_segs)
			cluster.or_gtot = dict((gname, 0.0) for gname in glist) # sum of gbar multiplied by area
			for gname in glist:
				cluster.or_gtot[gname] += sum(getattr(seg, gname)*seg.area() for seg in clu_segs)

		else: # SECTION-BASECD CLUSTERING
			# Gather member sections
			clu_secs = [secref for secref in orsecrefs if secref.cluster_label==cluster.label]

			# Compute equivalent properties
			cluster.eqL = math.sqrt(sum(secref.sec.L for secref in clu_secs)) / len(clu_secs)
			cluster.eqdiam = math.sqrt(sum(secref.sec.diam**2 for secref in clu_secs))
			cluster.eqRa = sum(secref.sec.Ra for secref in clu_secs)/len(clu_secs)
			cluster.or_area = sum(sum(seg.area() for seg in secref.sec) for secref in clu_secs)
			cluster.or_cmtot = sum(sum(seg.cm*seg.area() for seg in secref.sec) for secref in clu_secs)
			cluster.or_gtot = dict((gname, 0.0) for gname in glist) # sum of gbar multiplied by area
			for gname in glist:
				cluster.or_gtot[gname] += sum(sum(getattr(seg, gname)*seg.area() for seg in secref.sec) for secref in clu_secs)

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
		sec.cm = cluster.or_cmtot / cluster.eq_area

		# Set number of segments based on rule of thumb electrotonic length
		sec.nseg = redtools.min_nseg_hines(sec, f=100.)

		# Append to list of equivalent sections
		eq_secs[i] = sec
		eq_secrefs[i] = ExtSecRef(sec=sec)

		logger.debug("Created section '%s' with nseg=%d" % (cluster.label, sec.nseg))

	# Connect equivalent sections
	for i, clu_i in enumerate(clusters):
		for j, clu_j in enumerate(clusters):
			if clu_j is not clu_i and clu_j.parent_label == clu_i.label:
				eq_secs[j].connect(eq_secs[i], clu_j.parent_pos, 0)
	logger.info("Connected equivalent sections, topology is as follows:")
	h.topology() # prints tree topology

	# Set active properties and finetune
	for i, sec in enumerate(eq_secs):
		# Insert all mechanisms
		for mech in mechs_chans.keys():
			sec.insert(mech)

		# Set initial conductances by interpolation
		for seg in sec:
			for gname in glist:
				gval = interp_gbar_electrotonic(seg, gname, gleak_name, f_lambda, orsecrefs)
				seg.__setattr__(gname, gval)

		# Re-scale gbar distribution to yield same total gbar (sum(gbar*area))
		for gname in glist:
			for seg in sec:
				# conserves gtot_or since: sum(g_i*area_i * or_area/eq_area) = or_area/eq_area * sum(gi*area_i) ~= or_area/eq_area * g_avg*eq_area = or_area*g_avg
				seg.__setattr__(gname, getattr(seg, gname)*clusters[i].or_area/clusters[i].eq_area)

			# Check calculation
			gtot_or = cluster.or_gtot[gname]
			gtot_eq_scaled = sum(getattr(seg, gname)*seg.area() for seg in sec)
			logger.debug("conductance %s : gtot_or = %.3f ; gtot_eq = %.3f",
							gname, gtot_or, gtot_eq_scaled)

		# Debugging info:
		logger.debug("Created equivalent section '%s' with \n\tL\tdiam\tcm\tRa\tnseg\
		\n\t%.3f\t%.3f\t%.3f\t%.3f\t%d", cluster.label, sec.L, sec.diam, sec.cm, sec.Ra, sec.nseg)


# def reduce_bush_sejnowski():
#	""" Reduce STN cell according to Bush & Sejnowski (2016) method """
if __name__ == '__main__':

	############################################################################
	# 0. Load full model to be reduced (Gillies & Willshaw STN)
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
	somaref.cluster_labels = ['soma'] * somaref.sec.nseg
	somaref.table_index = 0
	somaclu.parent_label = 'soma'
	somaclu.parent_pos = 0.0
	clusters = [somaclu]

	# Cluster segments in each dendritic tree
	thresholds = (0.4, 1.0) # (0.4, 1.0) determine empirically for f=100Hz
	logger.info("Clustering left tree (dend0)...")
	redtools.clusterize_seg_electrotonic(dendLrefs[0], allsecrefs, thresholds, clusters)

	logger.info("Clustering right tree (dend1)...")
	redtools.clusterize_seg_electrotonic(dendRrefs[0], allsecrefs, thresholds, clusters)

	# Determine cluster topology
	redtools.assign_topology(clusters, ['soma', 'trunk', 'smooth', 'spiny'])

	############################################################################
	# 2. Create equivalent sections (compute ppties from cluster sections)

	# Create new sections
	logger.info("Creating equivalent sections...")
	eq_secrefs = equivalent_sections(clusters, allsecrefs, f_lambda)
	
	# Delete original model sections & set ion styles
	for sec in h.allsec(): # makes each section the CAS
		if sec.name().startswith('SThcell'): # original model sections
			h.delete_section()
		else: # equivalent model sections
			h.ion_style("na_ion",1,2,1,0,1)
			h.ion_style("k_ion",1,2,1,0,1)
			h.ion_style("ca_ion",3,2,1,1,1)

	############################################################################
	# 3. Finetuning/fitting of active properties

	# TODO: fitting using neurotune
	# return clusters, eq_secrefs

# if __name__ == '__main__':
# 	reduce_bush_sejnowski()