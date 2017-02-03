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
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
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
glist = [gname+'_'+mech for mech,chans in mechs_chans.iteritems() for gname in chans]

def interp_gbar_electrotonic(seg, gname, allsecrefs):
	""" Interpolate channel density (max conductance) according
		to electrotonic path.

	ALGORITHM
	- calculate electrotonic path length L_elec of seg in reduced model
	- in each original section where L0 <= L <= L1: find segment by linear interpolation
	- take average gbar of all sections in original model that meet this criterion
	"""
	pass

def equivalent_sections(clusters, allsecrefs):
	""" Compute properties of equivalent section for cluster
		from its member sections.

	@type	clusters	list(Cluster)
	@param	clusters	list of Cluster objects (containing cluster label 
						and properties)

	@type	allsecrefs	list(SectionRef)
	@param	allsecrefs	references to all sections in the cell

	@post				Equivalent section properties are available as
						attributed on given Cluster object
	"""
	eq_secs = []
	eq_secrefs = []
	for cluster in clusters:
		# Gather member sections
		clu_secs = [secref for secref in allsecrefs if secref.cluster_label==cluster.label]

		# Compute equivalent properties
		cluster.eqL = math.sqrt(sum(secref.sec.L for secref in clu_secs)) / len(clu_secs)
		cluster.eqdiam = math.sqrt(sum(secref.sec.diam**2 for secref in clu_secs))
		cluster.eqRa = sum(secref.sec.Ra for secref in clu_secs)/len(clu_secs)
		cluster.or_area = sum(sum(seg.area() for seg in secref.sec) for secref in clu_secs)
		cluster.or_cmtot = sum(sum(seg.cm*seg.area() for seg in secref.sec) for secref in clu_secs)

		# Create equivalent section
		if cluster.label in [sec.name() for sec in h.allsec()]:
			raise Exception('Section named {} already exists'.format(cluster.label))
		h("create %s" % cluster.label)
		sec = getattr(h, cluster.label)
		sec.push()
		eq_secs.append(sec)
		eq_secrefs.append(ExtSecRef(sec=sec))

		# Set geometry 
		sec.L = cluster.eqL
		sec.diam = cluster.eqdiam
		eq_area = sum(seg.area() for seg in sec) # should be same as cluster eqSurf
		surf_fact = cluster.or_area/eq_area # scale factor: ratio areas original/equivalent

		# Passive electrical properties (except Rm/gleak)
		sec.Ra = cluster.eqRa
		sec.cm = cluster.or_cmtot / eq_area

		# Set number of segments based on rule of thumb electrotonic length
		sec.nseg = redtools.min_nseg_hines(sec, f=100.)

		# Insert all mechanisms
		for mech in mechs_chans.keys():
			sec.insert(mech)

		# Set initial conductances (by interpolation or constant)
		for seg in sec:
			for gname in glist:
				# - calculate electrotonic path length L of seg in equivalent section
				# - in each original section where L0 <= L <= L1: find segment by linear interpolation
				# - take average gbar of all sections in original model that meet this criterion
				gval = interp_gbar_electrotonic(seg, gname, allsecrefs)
				seg.__setattr__(gname, gval)

		# Re-scale gbar distribution to yield same total gbar (sum(gbar*area))
		for gname in glist():
			gtot_eq = sum(getattr(seg, gname)*seg.area() for seg in sec)
			gtot_or = cluster.gtot_sum[gname]
			if gtot_eq <= 0. : gtot_eq = 1.
			for seg in sec:
				# NOTE: old method, almost same but less accurate
				# seg.__setattr__(gname, getattr(seg, gname)*surf_fact)
				# NOTE: this conserves gtot since sum(g_i*area_i * gtot_or/gtot_eq) = gtot_or/gtot_eq*sum(gi*area_i) = gtot_or/gtot_eq*gtot_eq
				seg.__setattr__(gname, getattr(seg, gname)*gtot_or/gtot_eq)
			# Check calculation
			gtot_eq_scaled = sum(getattr(seg, gname)*seg.area() for seg in sec)
			logger.debug("== Conductance %s ===\nOriginal: gtot = %.3f\nEquivalent: gtot = %.3f",
							gname, gtot_or, gtot_eq_scaled)

		# Debugging info:
		logger.debug("Created equivalent section '%s' with \n\tL\tdiam\tcm\tRa\tri\tpathri0\tpathri1\tnseg\
		\n\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d", cluster.label, sec.L, sec.diam, sec.cm, sec.Ra, 
		sec_ri, pathri0, pathri1, sec.nseg)

		# Unset CAS
		h.pop_section()


	# Connect equivalent sections
	for i, clu_i in enumerate(clusters):
		for j, clu_j in enumerate(clusters):
			if clu_j is not clu_i and clu_j.parent_label == clu_i.label:
				eq_secs[j].connect(eq_secs[i], clu_j.parent_pos, 0)


def reduce_bush_sejnowski():
	""" Reduce STN cell according to Bush & Sejnowski (2016) method """

	# Load Gillies model
	h.xopen("createcell.hoc")
	# Make sections accesible by name and index
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendLrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0] # 0 is left tree
	dendRrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1] # 1 is right tree
	allsecrefs = [somaref] + dendLrefs + dendRrefs

	### 1. Cluster into functional regions/according to electrotonic length

	logger.info("Computing electrotonic path lengths...")
	f_lambda = 100. # frequency for electrotonic length
	redtools.assign_electrotonic_length(dendLrefs, allsecrefs, f_lambda, 'gpas_STh')
	redtools.assign_electrotonic_length(dendRrefs, allsecrefs, f_lambda, 'gpas_STh')

	logger.info("Clustering sections...")
	# Cluster soma
	somaclu = Cluster('soma')
	somaref.cluster_label = 'soma'
	somaclu.parent_label = 'soma'
	somaclu.parent_pos = 0.0
	clusters = [somaclu]
	# Cluster dendritic trees
	thresholds = (1.0, 2.0) # TODO: determine empirically
	redtools.clusterize_electrotonic(dendLrefs[0], allsecrefs, thresholds, clusters)
	redtools.clusterize_electrotonic(dendRrefs[0], allsecrefs, thresholds, clusters)

	### 3. Compute properties of equivalent sections and create them
	logger.info("Merging within-cluster sections...")
	for cluster in clusters:
		merge_cluster(cluster, allsecrefs) # store equivalent properties on Cluster objects


	### 3. Scale passive properties

	### 4. Finetuning/fitting of active properties
	pass

if __name__ == '__main__':
	reduce_bush_sejnowsky()