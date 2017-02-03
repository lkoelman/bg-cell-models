"""
Reduce Gillies & Willshaw (2006) STN neuron model using the method
described in Marasco & Migliore (2012)


@author Lucas Koelman
@date	5-12-2016
"""

# Python modules
import math
PI = math.pi
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__) # create logger for this module
import sys
import os.path
scriptdir, scriptfile = os.path.split(__file__)
modulesbase = os.path.normpath(os.path.join(scriptdir, '..'))
sys.path.append(modulesbase)

# NEURON modules
import neuron
h = neuron.h

# Load NEURON function libraries
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

# Load NEURON mechanisms
# add this line to nrn/lib/python/neuron/__init__.py/load_mechanisms()
# from sys import platform as osplatform
# if osplatform == 'win32':
# 	lib_path = os.path.join(path, 'nrnmech.dll')
NRN_MECH_PATH = os.path.normpath(os.path.join(scriptdir, 'nrn_mechs'))
neuron.load_mechanisms(NRN_MECH_PATH)

# Own modules
import reduction_tools as redtools
from reduction_tools import ExtSecRef, Cluster, getsecref # for convenience
import reduction_analysis as analysis

# Global variables (convert to class members in future)
gillies_mechs_chans = {'STh': ['gpas'], # passive/leak channel
				'Na': ['gna'], 'NaL': ['gna'], # Na channels
				'KDR': ['gk'], 'Kv31': ['gk'], 'sKCa':['gk'], # K channels
				'Ih': ['gk'], # nonspecific channels
				'CaT': ['gcaT'], 'HVA': ['gcaL', 'gcaN'], # Ca channels
				'Cacum': []} # No channels

mechs_chans = gillies_mechs_chans

def glist():
	""" 
	Return list of conductance names accessible as properties
	on a section/segment after all mechanisms in mechs_chans.keys()
	have been inserted
	"""
	return [gname+'_'+mech for mech,chans in mechs_chans.iteritems() for gname in chans]

def merge_parallel(childrefs, allsecrefs):
	"""
	Merge parallel branched sections into one equivalent section
	"""
	# Initialize combined properties of branched sections (children)
	L_br = 0.
	diam_br = 0.
	Ra_br = 0.
	rin_br = 1.
	eqsurf_sum = 0.
	ri_sum = 0.
	gtot_br = dict((gname, 0.0) for gname in glist()) # sum of gbar multiplied by area
	cmtot_br = 0. # sum of cm multiplied by area

	# Update combined properties using the dimensions/electrical properties
	# of each child according to <br> expressions in Marasco (2012) 
	for childref in childrefs:
		# get equivalent child properties
		L_child, diam_child, Ra_child, ri_child, cmtot_child, gtot_child = merge_sequential(childref, allsecrefs)
		eqsurf_child = PI*diam_child*L_child

		# combine according to <br> (parallel) expressions
		eqsurf_sum += eqsurf_child
		L_br += eqsurf_child*L_child # LENGTH: eq (1) - weight by area
		diam_br += diam_child**2 # RADIUS: eq (2) - 2-norm of radii
		Ra_br += Ra_child # SPECIFIC AXIAL RESISTANCE - average Ra
		rin_br *= ri_child # ABSOLUTE AXIAL RESISTANCE - parallel conductances
		ri_sum += ri_child # need sum in final calculation

		# Distributed properties
		cmtot_br += cmtot_child
		for gname in glist():
			gtot_br[gname] += gtot_child[gname]

		# mark child as absorbed
		childref.absorbed = True
		childref.visited = True

	# Finalize <br> calculation (MUST BE VALID IF ONLY ONE CHILDREF)
	L_br /= eqsurf_sum # eq. L_br
	diam_br = math.sqrt(diam_br) # eq. rho_br
	Ra_br = Ra_br/len(childrefs) # average Ra (NOTE: not used, cluster Ra calc from dimensions & ri)
	cross_br = PI*diam_br**2/4. # cross-section area
	rax_br = Ra_br*L_br/cross_br/100. # absolute axial resistance of section with equivalent dimensions and Ra
	if len(childrefs) > 1:
		rin_br /= ri_sum # eq. r_a,br: product/sum

	return L_br, diam_br, Ra_br, rin_br, rax_br, cmtot_br, gtot_br

def merge_sequential(rootref, allsecrefs):
	""" 
	Merge sequential sections into one equivalent sections
	"""
	# Get references
	rootsec = rootref.sec
	allchildrefs = [getsecref(sec, allsecrefs) for sec in rootsec.children()]
	childrefs = [ref for ref in allchildrefs if ref.cluster_label==rootref.cluster_label]

	# Gather properties for root sec
	L_root = rootsec.L
	diam_root = rootsec.diam
	Ra_root = rootsec.Ra
	ri_root = sum(seg.ri() for seg in rootsec) # absolute axial resistance between 0-1 ends

	# Distributed properties
	cmtot_seq = sum(seg.cm*seg.area() for seg in rootsec) # sum of cm multiplied by area
	gtot_seq = dict((gname, 0.0) for gname in glist()) # sum of gbar multiplied by area
	for gname in glist():
		gtot_seq[gname] += sum(getattr(seg, gname)*seg.area() for seg in rootsec)

	# Handle leaf sections
	if not any(childrefs):
		return L_root, diam_root, Ra_root, ri_root, cmtot_seq, gtot_seq

	# Combine properties of parallel branched sections (children)
	L_br, diam_br, Ra_br, rin_br, rax_br, cmtot_br, gtot_br = merge_parallel(childrefs, allsecrefs)

	# use <seq> expressions in Marasco (2012) to merge equivalent child into parent
	L_seq = L_root + L_br # L_seq equation
	Ra_seq = (Ra_root + Ra_br)/2.
	ri_seq = ri_root + rin_br # var 'newri2' in Marasco code used for ri calculation
	rax_seq = ri_root + rax_br # var 'newri' in Marasco code used for diam calculation
	diam_seq = math.sqrt(Ra_seq*L_seq*4./PI/rax_seq/100.) # rho_seq equation (conserves ri_seq)
	
	# Keep track of total conductance/capacitance
	cmtot_seq += cmtot_br
	for gname in glist():
		gtot_seq[gname] += gtot_br[gname]

	return L_seq, diam_seq, Ra_seq, ri_seq, cmtot_seq, gtot_seq

def collapse_subtree(rootref, allsecrefs):
	""" 
	Recursively merge within-cluster connected sections in subtree
	of the given node using <br> and <seq> expressions in Marasco (2012).
	"""
	# Collapse is equal to sequential merge of the root and equivalent parallel circuit of its children
	return merge_sequential(rootref, allsecrefs)

def chan_densities(cluster, allsecrefs):
	"""
	Calculate channel densities for all sections in given cluster

	ALGORITHM:
	- for each section:
		- for each segment in section, save gbar and axial path resistance
			- axial path resistance obtained by interpolating pathri0 & pathri1
	"""
	clu_secs = [secref for secref in allsecrefs if secref.cluster_label==cluster.label]

	# Keep a dict that maps gname to a collection of data points (pathri, gbar)
	cluster.pathri_gbar = dict((gname, []) for gname in glist())
	for secref in clu_secs:
		for seg in secref.sec:
			for gname in glist():
				if not hasattr(seg, gname):
					continue # section doesn't have ion channel: skip
				# Add data point
				seg_pathri = secref.pathri0 + seg.x*(secref.pathri1-secref.pathri0)
				seg_gbar = getattr(seg, gname)
				cluster.pathri_gbar[gname].append((seg_pathri, seg_gbar))

def calc_gbar(cluster, gname, pathri):
	"""
	Calculate gbar for a point on the equivalent section of given cluster
	given the axial path resistance to that point.
	"""
	gbar_pts = cluster.pathri_gbar[gname] # list of (pathri, gbar) data points
	gbar_pts.sort(key=lambda pt: pt[0]) # sort by pahtri ascending
	eq_path_x = (pathri - cluster.pathri0) / (cluster.pathri1 - cluster.pathri0)
	if eq_path_x <= 0.:
		return gbar_pts[0]
	elif eq_path_x >= 1.:
		return gbar_pts[-1]
	else:
		# average of points that lie within pathri +/- 11% of segment axial resistance
		deltari = 0.11*(cluster.pathri1-cluster.pathri0)
		surr_pts = [pt for pt in gbar_pts if (pt[0] >= pathri-deltari) and (pt[0] <= pathri+deltari)]
		if not any(surr_pts):
			# take average of two closest points
			gbar_pts.sort(key=lambda pt: abs(pt[0]-pathri))
			surr_pts = [pt for i, pt in enumerate(gbar_pts) if i < 2]
		return sum(pt[1] for pt in surr_pts)/len(surr_pts) # take average


def merge_cluster(cluster, allsecrefs, average_trees):
	"""
	Merge sections in cluster

	@param average_trees	If True, the input resistance is not conserved for (i.e.
						no parallel circuit of disconnected subtrees in cluster) but
						the specific axial resistance is calculated so that
						the absolute axial resistance of the equivalent section 
						is the average of all disconnected subtrees in the cluster.
						This is the method used in Marasco (2013) (RaMERGINGMETHOD=1).
							If False, the input resistance is conserved: the specific 
						axial resistance is the average of the cluster and the diameter 
						is calculated so that the absolute axial resistance will be 
						equivalent to the parallel circuit of all the unconnected subtrees, 
						preserving input resistance.

	ALGORITHM
	- find the next root of a within-cluster connected subtree
		- i.e. a section without a parent in the same cluster
		- this can be an isolated sections (no mergeable children within cluster)
	- collapse subtree of that root section
	- update equivalent cluster properties using the <eq> expressions in Marasco (2012)

	"""
	# Gather sections and mark/flag them
	clu_secs = [secref for secref in allsecrefs if secref.cluster_label==cluster.label]
	for sec in clu_secs:
		sec.absorbed = False
		sec.visited = False

	# Calculate min/max path resistance in cluster (full model)
	for sec in clu_secs:
		redtools.calc_path_ri(sec) # assigns pathri0/pathri1
	cluster.orMaxpathri = max(secref.pathri1 for secref in clu_secs)
	cluster.orMinpathri = min(secref.pathri0 for secref in clu_secs)

	# Calculate distributions of ion channel densities
	chan_densities(cluster, allsecrefs)

	# Initialize equivalent properties
	cluster.eqL = 0.
	cluster.eqdiam = 0.
	cluster.eqri = 0.
	cluster.orSurfSum = sum(sum(seg.area() for seg in secref.sec) for secref in clu_secs)
	cluster.eqSurfSum = 0. # sum of surface calculated from equivalent dimensions
	cluster.cmtot_sum = 0.
	cluster.gtot_sum = dict((gname, 0.0) for gname in glist())

	# utility function to check if sec has parent within cluster
	def has_clusterparent(secref):
		return secref.has_parent() and (getsecref(secref.parent, clu_secs) is not None)

	# Find connected subtrees within cluster and merge/collapse them
	rootfinder = (sec for sec in clu_secs if (not sec.visited and not has_clusterparent(sec))) # compiles generator function
	for secref in rootfinder:
		rootref = redtools.clusterroot(secref, clu_secs) # make sure it is a cluster root

		# Collapse subtree
		logger.debug("Collapsing subtree of cluster root %s", repr(rootref))
		L_eq, diam_eq, Ra_eq, ri_eq, cmtot_eq, gtot_eq = collapse_subtree(rootref, allsecrefs)

		# Combine properties of collapse sections using <eq> expressions
		surf_eq = L_eq*PI*diam_eq
		cluster.eqSurfSum += surf_eq
		cluster.eqL += L_eq * surf_eq
		cluster.eqdiam += diam_eq**2
		cluster.eqri += ri_eq

		# Save distributed properties
		cluster.cmtot_sum += cmtot_eq
		for gname in glist():
			cluster.gtot_sum[gname] += gtot_eq[gname]

		# Mark as visited
		rootref.visited = True

	# Check each section either absorbed or rootsec
	# assert not any(not sec.absorbed and has_clusterparent(sec) for sec in clu_secs), (
	assert all(sec.absorbed or not has_clusterparent(sec) for sec in clu_secs), (
			'Each section should be either absorbed or be a root within the cluster')

	# Finalize <eq> calculation
	cluster.eqL /= cluster.eqSurfSum # LENGTH: equation L_eq
	cluster.eqdiam = math.sqrt(cluster.eqdiam) # RADIUS: equation rho_eq
	cluster.eqri /= sum(not sec.absorbed for sec in clu_secs) # ABSOLUTE AXIAL RESISTANCE: equation r_a,eq
	if average_trees:
		cluster.eqRa = PI*(cluster.eqdiam/2.)**2*cluster.eqri*100./cluster.eqL # conserve eqri as absolute axial resistance
	else:
		cluster.eqRa = sum(secref.sec.Ra for secref in clu_secs)/len(clu_secs) # average Ra in cluster
	cluster.eqSurf = cluster.eqL*PI*cluster.eqdiam # EQUIVALENT SURFACE

	# Debugging info
	logger.debug("Merged cluster '%s': equivalent properties are:\
		\n\teqL\teqdiam\teqRa\teqri\teqSurf\
		\n\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f",
		cluster.label, cluster.eqL, cluster.eqdiam, cluster.eqRa, cluster.eqri, cluster.eqSurf)

def lambda_f(f, diam, Ra, cm):
	""" Compute electrotonic length (taken from stdlib.hoc) """
	return 1e5*math.sqrt(diam/(4*math.pi*f*Ra*cm))

def min_nseg_hines(sec):
	""" Minimum number of segments based on electrotonic length """
	return int(sec.L/(0.1*lambda_f(100., sec.diam, sec.Ra, sec.cm))) + 1

def min_nseg_marasco(sec):
	""" Minimum number of segments based on electrotonic length """
	return int((sec.L/(0.1*lambda_f(100., sec.diam, sec.Ra, sec.cm))+0.9)/2)*2 + 1  

def equivalent_sections(clusters, allsecrefs, gradients):
	""" Create the reduced/equivalent cell by creating 
		a section for each cluster 

	@param clusters		list of Cluster objects containing data
						for each cluster

	@param allsecrefs	list of SectionRef (mutable) with first element
						a ref to root/soma section

	@param gradients	if True, gbar in each segment is determined from
						the average gbar at the same path resistance in the
						original model (i.e. nonuniform gbar). If False,
						a uniform gbar is used in each equivalent section.

	@return				list of SectionRef containing equivalent Section 
						for each cluster (in same order) as well as min
						and max path resistance for each cluster/section
						as properties pathri0/pathri1 on SectionRef objects
	"""
	# Create equivalent section for each clusters
	# eq_secs = [h.Section() for clu in clusters]
	for clu in clusters:
		h("create %s" % clu.label) # ensures references are not destroyed and names are clear
	eq_secs = [getattr(h, clu.label) for clu in clusters]
	eq_secrefs = [ExtSecRef(sec=sec) for sec in eq_secs]

	# Connect sections
	for i, clu_i in enumerate(clusters):
		for j, clu_j in enumerate(clusters):
			if clu_j is not clu_i and clu_j.parent_label == clu_i.label:
				eq_secs[j].connect(eq_secs[i], clu_j.parent_pos, 0)

	# Set dimensions, passive properties, active properties
	for i, secref in enumerate(eq_secrefs):
		sec = secref.sec
		sec.push() # Make section the CAS
		cluster = clusters[i]

		# Store some cluster properties on SectionRef
		secref.cluster_label = cluster.label
		secref.order = cluster.order

		# Set geometry 
		sec.L = cluster.eqL
		sec.diam = cluster.eqdiam
		sec_area = sum(seg.area() for seg in sec) # should be same as cluster eqSurf
		surf_fact = cluster.orSurfSum/cluster.eqSurf # scale factor: ratio areas original/equivalent

		# Passive electrical properties (except Rm/gleak)
		sec.cm = cluster.cmtot_sum / sec_area
		sec.Ra = cluster.eqRa

		# Set number of segments based on rule of thumb electrotonic length
		sec.nseg = min_nseg_hines(sec)

		# calculate min/max path resistance in equivalent section (cluster)
		pathri0, pathri1 = redtools.calc_path_ri(eq_secrefs[i])
		cluster.pathri0 = pathri0
		cluster.pathri1 = pathri1
		sec_ri = sum(seg.ri() for seg in sec)
		assert (pathri0 < pathri1), ('Axial path resistance at end of section '
									 'should be higher than at start of section')
		assert (-1e-6 < (pathri1 - pathri0) - sec_ri < 1e-6), ('absolute axial '
						'resistance not consistent with axial path resistance')

		# Insert all mechanisms and set conductances
		for mech in mechs_chans.keys():
			sec.insert(mech)
		for seg in sec:
			for gname in glist():
				if gradients:
					# Look for average gbar value at points with same path resistance
					seg_pathri = pathri0 + seg.x*(pathri1-pathri0)
					gval = calc_gbar(cluster, gname, seg_pathri)
				else:
					gval = cluster.gtot_sum[gname] / sec_area # yields same sum(gbar*area) as in full model
				seg.__setattr__(gname, gval)
		
		# Re-scale gbar distribution to yield same total gbar (sum(gbar*area))
		if gradients:
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

	return eq_secs, eq_secrefs # return both or secs will be deleted

def cluster_sections(rootrefs, allsecrefs, custom=True):
	""" Cluster all sections 

	@param custom	if False, use Strahler's numbers with two thresholds
					for clustering dendritic sections. If True, use a custom
					clustering method/criterion defined in local function
					clusterfun().

	@return			list of Cluster objects that store each cluster's label,
					parent label, position on parent, order
	"""
	somaref, dendLroot, dendRroot = rootrefs[:]

	# Cluster soma
	somaref.cluster_label = 'soma'
	somaclu = Cluster('soma')
	somaclu.parent_label = 'soma'
	somaclu.parent_pos = 0.0
	somaclu.order = 0
	clusters = [somaclu]

	def clusterfun(secref):
		""" Assign cluster label based on diameter """
		if secref.sec.diam <= 1.0: # diameters: see Gilles & Willshaw (2006) fig. 1
			return 'spiny'
		else:
			return 'trunk'

	# Cluster dendritic trees
	if custom:
		redtools.clusterize_custom(dendLroot, allsecrefs, clusterfun, clusters, 
									labelsuffix='_0')
		redtools.clusterize_custom(dendRroot, allsecrefs, clusterfun, clusters, 
									labelsuffix='_1', parent_pos=0.)
	else:
		redtools.clusterize_strahler(dendLroot, allsecrefs, thresholds=(1,2), 
									clusterlist=clusters, labelsuffix='_0')
		redtools.clusterize_strahler(dendRroot, allsecrefs, thresholds=(1,2),
									clusterlist=clusters, labelsuffix='_1', parent_pos=0.)
	# Debug info
	for cluster in clusters:
		logger.debug("Cluster '{0}'' has parent cluster '{1}'".format(cluster.label, cluster.parent_label))

	return clusters

def label_order(label):
	""" Return order (distance from soma) based on label """
	if label.startswith('soma'):
		return 0
	elif label.startswith('trunk'):
		return 1
	elif label.startswith('smooth'):
		return 2
	elif label.startswith('spiny'):
		return 3
	else:
		return 4

def reduce_gillies(customclustering, average_trees):
	""" Reduce Gillies & Willshaw STN neuron model 

	@param customclustering		see param 'customclustering' in function
								cluster_sections()
	@param average_trees		see param 'average_trees' in function
								merge_cluster
	"""

	# Initialize Gillies model
	h.xopen("createcell.hoc")

	# Make sections accesible by both name and index + allow to add attributes
	somaref = ExtSecRef(sec=h.SThcell[0].soma)
	dendLrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend0]
	dendRrefs = [ExtSecRef(sec=sec) for sec in h.SThcell[0].dend1]
	alldendrefs = dendLrefs + dendRrefs
	allsecrefs = [somaref] + alldendrefs
	or_refs = (somaref, dendLrefs, dendRrefs)

	# Assign Strahler numbers
	logger.info("Assingling Strahler's numbers...")
	redtools.assign_strahler_order(dendLrefs[0], dendLrefs, 0)
	redtools.assign_strahler_order(dendRrefs[0], dendRrefs, 0)
	somaref.order = 0 # distance from soma
	somaref.strahlernumber = dendLrefs[0].strahlernumber # same as root of left tree

	# Cluster sections
	logger.info("Clustering sections...")
	rootrefs = (somaref, dendLrefs[0], dendRrefs[0])
	clusters = cluster_sections(rootrefs, allsecrefs, custom=customclustering)

	# Merge sections within each cluster: 
	# i.e. calculate properties of equivalent section for each cluster
	logger.info("Merging within-cluster sections...")
	for cluster in clusters:
		merge_cluster(cluster, allsecrefs, average_trees) # store equivalent properties on Cluster objects

	# Create equivalent section for each cluster
	logger.info("Creating equivalent sections...")
	eq_secs, eq_secrefs = equivalent_sections(clusters, allsecrefs, gradients=True)

	# Sort equivalent sections
	eq_sorted = sorted(eq_secrefs, key=lambda ref: label_order(ref.cluster_label))
	eq_somaref = next(ref for ref in eq_secrefs if ref.cluster_label.startswith('soma'))
	eq_dendLrefs = [ref for ref in eq_secrefs if ref.cluster_label.endswith('0')]
	eq_dendRrefs = [ref for ref in eq_secrefs if ref.cluster_label.endswith('1')]
	eq_somasec = eq_somaref.sec
	eq_dendLsecs = [ref.sec for ref in eq_dendLrefs]
	eq_dendRsecs = [ref.sec for ref in eq_dendRrefs]

	# Compare full/reduced model
	eq_secs = (eq_somasec, eq_dendLsecs, eq_dendRsecs)
	eq_refs = (eq_somaref, eq_dendLrefs, eq_dendRrefs)
	analysis.compare_models(or_refs, eq_refs, [])

	# Delete original model sections
	for sec in h.allsec(): # makes each section the CAS
		if sec.name().startswith('SThcell'): # original model sections
			h.delete_section()
		else: # equivalent model sections
			h.ion_style("na_ion",1,2,1,0,1)
			h.ion_style("k_ion",1,2,1,0,1)
			h.ion_style("ca_ion",3,2,1,1,1)

	logger.info("Equivalent tree topology:")
	if logger.getEffectiveLevel() >= logging.DEBUG:
		h.topology() # prints topology
	
	# return data structures
	return clusters, eq_secs, eq_refs

if __name__ == '__main__':
	clusters, eq_secs, eq_refs = reduce_gillies(True, False)