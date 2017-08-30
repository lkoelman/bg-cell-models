"""
Reduce model by folding/collapsing branches according to algorithm described in following articles:

- Stratford, K., Mason, A., Larkman, A., Major, G., and Jack, J. J. B. (1989) - The modelling of pyramidal neurones in the visual cortex

- Fleshman, J. W., Segev, I., and Burke, R. E. (1988) - Electrotonic architecture of type-identified alpha-motoneurons in the cat spinal cord. J. Neurophysiol. 60: 60 85.

- Clements, J., and Redman, S. (1989) - Cable properties of cat spinal motoneurones measured by combining voltage clamp, current clamp and intracellular staining. J. Physiol. 409: 63 87.

- Algorithm overview in 'Methods in Neural Modeling', Chapter 3.4, equations 3.21 - 3.22

@author Lucas Koelman

@date	28-08-2017
"""

import re

import folding
import redutils
import interpolation as interp

from redutils import get_sec_props_obj
from cluster import Cluster, assign_topology_attrs

from common import treeutils
from common.treeutils import getsecref, next_segs, seg_index, interp_seg
from common.electrotonic import calc_lambda

from neuron import h

# Global parameters
f_lambda = 0.0
alphabet_uppercase = [chr(i) for i in xrange(65,90+1)] # A-Z are ASCII 65-90

# logging of DEBUG/INFO/WARNING messages
import logging
logging.basicConfig(format='%(message)s %(levelname)s:@%(filename)s:%(lineno)s', level=logging.DEBUG)
logname = "reduction" # __name__
logger = logging.getLogger(logname) # create logger for this module

################################################################################
# Folding Algorithm
################################################################################

def next_eq_diam(a_seg, a_i, root_X, dX, cluster, reduction):
	"""
	Compute equivalent diameter at next discretization step (a_i+1) * dX
	in the Stratford et al. (1989) reduction method.

	@param	a_seg		nrn.Segment : last segment encountered when ascending the tree.

	@param	a_i			discretization step index of last computed diameter
						(use 0 for first call)

	@param	root_X		electrotonic distance of the first segment (the subtree
						root segment) from soma or root of whole tree

	@param	dX			discretization step size
	"""
	allsecrefs = reduction.all_sec_refs
	gleak_name = reduction.gleak_name
	gbar_list = reduction.active_gbar_names

	# electrotonic distance to last segment where diam was calculated
	last_X = root_X + a_i*dX
	target_X = last_X + dX

	# electrotonic distance to last visited segment (not necessarily used)
	a_ref = getsecref(a_seg.sec, allsecrefs)
	a_index = seg_index(a_seg)
	a_X0 = a_ref.seg_path_Lelec0[a_index]
	a_X1 = a_ref.seg_path_Lelec1[a_index]
	a_X = interp_seg(a_seg, a_X0, a_X1)
	assert target_X > a_X # should be guaranteed by last call

	# Get child segments
	child_segs = next_segs(a_seg) # segments/nodes connected to this one
	if not any(child_segs):
		if (a_seg.x == 1.0):
			logger.debug((a_i*"-") + "Folding: reached end of branch @ {}".format(a_seg))
			return # reached end of branch
		else:
			child_segs = [a_seg.sec(1.0)] # test end of branch
	
	# Get new segment at (a_i+1) * dX
	for b_seg in child_segs:

		# NOTE: depth-first tree traversal: look for next point ta calculate diam
		#       (next discretization step) along current branch, then ascend further

		# Get electronic distance to following/adjacent segment
		b_ref = getsecref(b_seg.sec, allsecrefs)
		b_index = seg_index(b_seg)
		b_X0 = b_ref.seg_path_Lelec0[b_index]
		b_X1 = b_ref.seg_path_Lelec1[b_index]
		b_X = interp_seg(b_seg, b_X0, b_X1)
		# last_dX = b_X - last_X

		# If distance to last used segment is smaller than step size: ascend to next segment
		if target_X > b_X: # (if dX > last_dX)
			
			# Examples: (s=last_seg, a=a_seg, b=b_seg, b=target_seg)
			#	[-----|-s-a-|b--t-] - [t----|-----|-----]
			#	[-----|---sa|--b--] - [t----|-----|-----]
			logger.debug((a_i*"-") + "Skipping {} since bX={} < tarX={}".format(b_seg, b_X, target_X))

			# Ascend but don't increase index
			next_eq_diam(b_seg, a_i, root_X, dX, cluster, reduction)

		else: # target_X <= b_X
			# target X value is between previous and next segment -> get new segment by interpolation
			
			# Examples: (s=last_seg, a=a_seg, b=b_seg, b=target_seg)
			#	[-----|-s-a-|t--b-] - [-----|-----|-----]
			#	[-----|---s-|a-t--] - [b----|-----|-----]
			logger.debug((a_i*"-") + "Interpolating between {} (X={}) and {} (X={})".format(a_seg, a_X, b_seg, b_X))
			
			fX = (target_X - a_X) / (b_X - a_X)
			new_seg = None
		
			if b_seg.sec.same(a_seg.sec): # segment A and B are in same cylinder
				
				# Examples: (s=last_seg, a=a_seg, b=b_seg, b=target_seg)
				#	[-----|-s-a-|t--b-] - [t----|-----|-----]
				
				# linear interpolation
				assert b_seg.x > a_seg.x
				d_x = b_seg.x - a_seg.x
				new_x = a_seg.x + (fX * d_x)
				new_seg = a_seg.sec(new_x)

			else: # segment B is in next cylinder (Section)

				# Examples: (s=last_seg, a=a_seg, b=b_seg, b=target_seg)
				#	[-----|---s-|a-t--] - [b----|-----|-----]

				if fX <= 0.5: # new segment is in cylinder A

					d_x = 1.0 - a_seg.x
					frac = fX/0.5
					new_x = a_seg.x + (frac * d_x)
					new_seg = a_seg.sec(new_x)

				else: # new segment is in cylinder B

					frac = (fX-0.5) / 0.5
					new_x = frac*b_seg.x
					new_seg = b_seg.sec(new_x)

			# Save attributes at this electrotonic distance from root
			step_attrs = {}
			step_attrs['step_diams'] = [new_seg.diam]
			step_attrs['step_Rm'] = [1.0 / getattr(new_seg, gleak_name)]
			step_attrs['step_Ra'] = [new_seg.sec.Ra]
			step_attrs['step_cm'] = [new_seg.cm]
			step_attrs['step_gbar'] = [dict(((gname, getattr(new_seg, gname)) for gname in gbar_list))]

			# Save them on cluster object
			for attr_name, new_seg_vals in step_attrs.iteritems():

				# Get cluster attribute
				step_attr_list = getattr(cluster, attr_name)
				
				# Save values for new segment
				if len(step_attr_list) <= a_i:
					# First entry (first branch for this step) : start new list
					step_attr_list.append(new_seg_vals)
				else:
					# Subsequent branches: add to list for this step
					step_attr_list.extend(new_seg_vals)


			# Ascend and increase index
			next_eq_diam(new_seg, a_i+1, root_X, dX, cluster, reduction)

		return


def calc_folds(target_Y_secs, i_pass, reduction, dX=0.1):
	"""
	Do folding operation

	@return			list of Cluster objects with properties of equivalent
					Section for each set of collapsed branches.
	"""

	allsecrefs = reduction.all_sec_refs
	gbar_list = reduction.active_gbar_names

	clusters = []
	for j_zip, root_ref in enumerate(target_Y_secs):

		# Fold name
		p = re.compile(r'(?P<cellname>\w+)\[(?P<cellid>\d+)\]\.(?P<seclist>\w+)\[(?P<secid>\d+)\]')
		pmatch = p.search(root_ref.sec.name())
		if pmatch:
			pdict = pmatch.groupdict()
			name_sanitized = "{}_{}".format(pdict['seclist'], pdict['secid'])
		else:
			name_sanitized = re.sub(r"[\[\]\.]", "", root_ref.sec.name())
		fold_label = "fold_{0}".format(name_sanitized)

		# Make Cluster object that represents collapsed segments
		cluster = Cluster(fold_label)

		# Branch parameters
		cluster.step_diams = []
		cluster.step_Rm = []
		cluster.step_Ra = []
		cluster.step_cm = []
		cluster.step_gbar = []

		# Topological info
		cluster.parent_seg = root_ref.sec(1.0)

		# Ascend subtree from root and compute diameters
		start_seg = root_ref.sec(1.0) # end of cylinder
		start_X = root_ref.pathLelec1
		logger.debug("Folding: start tree ascent @ {}".format(start_seg))
		next_eq_diam(start_seg, 0, start_X, dX, cluster, reduction)

		# Equivalent parameters
		cluster.num_sec = len(cluster.step_diams)
		cluster.eq_diam =	[0.0] * cluster.num_sec
		cluster.eq_L =		[0.0] * cluster.num_sec
		cluster.eq_Rm =		[0.0] * cluster.num_sec
		cluster.eq_Ra =		[0.0] * cluster.num_sec
		cluster.eq_cm =		[0.0] * cluster.num_sec
		cluster.eq_lambda =	[0.0] * cluster.num_sec
		cluster.eq_gbar =	[0.0] * cluster.num_sec

		# Finalize diam calculation
		
		for i_step, diams in enumerate(cluster.step_diams):

			# Number of parallel branches found at this step
			num_parallel = len(diams)

			# Equivalent properties at this step
			cluster.eq_diam[i_step] = sum((d**(3.0/2.0) for d in diams)) ** (2.0/3.0)

			# Passive electrical properties
			cluster.eq_Rm[i_step] = sum(cluster.step_Rm[i_step]) / num_parallel
			cluster.eq_Ra[i_step] = sum(cluster.step_Ra[i_step]) / num_parallel
			cluster.eq_cm[i_step] = sum(cluster.step_cm[i_step]) / num_parallel

			# Combine dict of gbar for each parallel section
			cluster.eq_gbar[i_step] = dict(((gname, 0.0) for gname in gbar_list))
			for gdict in cluster.step_gbar[i_step]:
				for gname, gval in gdict.iteritems():
					cluster.eq_gbar[i_step][gname] += (gval / num_parallel) # average of parallel branches

			# Physical length 
			# NOTE: it is important to use the same lambda equation as the one
			#       used for electrotonic path lengths
			cluster.eq_lambda[i_step] = calc_lambda(f_lambda, cluster.eq_diam[i_step], cluster.eq_Ra[i_step], 
													1.0/cluster.eq_Rm[i_step], cluster.eq_cm[i_step])
			
			cluster.eq_L[i_step] = dX * cluster.eq_lambda[i_step]

		# Save cluster
		clusters.append(cluster)

	return clusters


def new_fold_section(cluster, i_step, j_sec, reduction):
	"""
	Start new Section for discretization step i of the cluster.
	"""
	sec, ref = treeutils.create_hoc_section("{}_{}".format(
								cluster.label, alphabet_uppercase[j_sec]))
	
	sec.L		= cluster.eq_L[i_step]
	sec.nseg	= 1
	sec.diam	= cluster.eq_diam[i_step]
	sec.Ra		= cluster.eq_Ra[i_step]
	sec.cm		= cluster.eq_cm[i_step]

	# Insert mechanisms
	for mech in reduction.mechs_gbars_dict:
		sec.insert(mech)

	# Set conductances
	setattr(sec(1.0), reduction.gleak_name, 1.0/cluster.eq_Rm[i_step])
	for gname in reduction.active_gbar_names:
		setattr(sec(1.0), gname, cluster.eq_gbar[i_step][gname])

	return sec, ref


def extend_fold_section(cluster, i_step, sec, reduction):
	"""
	Extend existing fold Section with a new segment for given discretization step.
	"""

	sec.L			= sec.L + cluster.eq_L[i_step]
	sec.nseg		= sec.nseg + 1
	sec(1.0).diam	= cluster.eq_diam[i_step]
	sec(1.0).cm		= cluster.eq_cm[i_step]
	# NOTE: Ra is not RANGE var -> only set once per Section

	# Set conductances
	setattr(sec(1.0), reduction.gleak_name, 1.0/cluster.eq_Rm[i_step])
	for gname in reduction.active_gbar_names:
		setattr(sec(1.0), gname, cluster.eq_gbar[i_step][gname])



def make_substitute_folds(clusters, reduction):
	"""
	Make equivalent Sections for each cluster, and substitute them into cell
	to replace the folded compartments.
	"""

	# Create one Section for each region with uniform diam
	# (uniform diam also means uniform length: see equation for physical length)
	diam_tolerance = 0.05 # 5 percent

	for i_clu, cluster in enumerate(clusters):

		# Save equivalent sections
		cluster.eq_refs = []

		# Create first Section for this fold
		cur_diam = cluster.eq_diam[0]
		cur_sec, cur_ref = new_fold_section(cluster, 0, 0, reduction)
		cur_sec.connect(cluster.parent_seg, 0.0) # for electrotonic properties
		
		num_fold_secs = 1
		cluster.eq_refs.append(cur_ref)

		# Create Section for each region with uniform diam
		for i_step, diam in enumerate(cluster.eq_diam):

			if (1.0 - diam_tolerance) <= (diam / cur_diam) <= (1.0 + diam_tolerance):
				
				# Uniform: extend current Section with new segment
				extend_fold_section(cluster, i_step, cur_sec, reduction)

			else:
				
				# Not uniform: start new Section
				new_sec, new_ref = new_fold_section(cluster, i_step, num_fold_secs, reduction)
				cluster.eq_refs.append(new_ref)

				# Connect to previous sections
				new_sec.connect(cur_sec(1.0), 0.0)

				# Update current value
				num_fold_secs += 1
				cur_diam = diam
				cur_sec, cur_ref = new_sec, new_ref


		# Set nonuniform active conductances
		# TODO: plot response and test different dX
		for eq_ref in cluster.eq_refs:

			# first calculate electrotonic properties of equivalent sections
			redutils.sec_path_props(eq_ref, f_lambda, reduction.gleak_name)

			# interpolate conductance in each segment
			for j_seg, seg in enumerate(eq_ref.sec):
			
				# Get adjacent segments along path
				seg_X0, seg_X1 = eq_ref.seg_path_Lelec0[j_seg], eq_ref.seg_path_Lelec1[j_seg]
				seg_X = interp_seg(seg, seg_X0, seg_X1)
				bound_segs, bound_X = interp.find_adj_path_segs('path_L_elec', seg_X, reduction.interp_path)

				# Set conductances by interpolating neighbors
				for gname in reduction.active_gbar_names:
					gval = interp.interp_gbar_linear_neighbors(seg_X, gname, bound_segs, bound_X)
					seg.__setattr__(gname, gval)


		# Disconnect substituted Sections
		clu_root_sec = cluster.eq_refs[0].sec
		redutils.sub_equivalent_Y_sec(clu_root_sec, cluster.parent_seg, [], 
								reduction.all_sec_refs, reduction.mechs_gbars_dict, 
								delete_substituted=True)

		# Set ion styles
		for ref in cluster.eq_refs:
			redutils.set_ion_styles(ref.sec, **reduction._ion_styles)



################################################################################
# Interface Functions (FoldReduction)
################################################################################


def preprocess_impl(reduction):
	"""
	Preprocess cell for Stratford reduction.

	(Implementation of interface declared in reduce_cell.CollapseReduction)

	@param	reduction		reduce_cell.CollapseReduction object
	"""
	
	dendL_secs = list(h.SThcell[0].dend0)
	dendR_secs = list(h.SThcell[0].dend1)

	allsecrefs = reduction.all_sec_refs
	dend_lists = [dendL_secs, dendR_secs]

	# Assign indices used in Gillies code (sth-data folder)
	for somaref in reduction._soma_refs:
		somaref.tree_index = -1
		somaref.table_index = 0

	for secref in reduction._dend_refs:
		for i_dend, dendlist in enumerate(dend_lists):
			if any([sec.same(secref.sec) for sec in dendlist]):
				secref.tree_index = i_dend
				secref.table_index= next((i+1 for i,sec in enumerate(dendlist) if sec.same(secref.sec)))

	# Choose representative path for interpolation
	interp_tree_id = 1
	interp_table_ids = (1,3,8) # from soma to end of SThcell[0].dend1[7]
	reduction.interp_path = []

	# Save properties along this path
	for ref in allsecrefs:
		if (ref.tree_index==interp_tree_id) and (ref.table_index in interp_table_ids):
			
			# calculate properties
			redutils.sec_path_props(ref, f_lambda, reduction.gleak_name)

			# Save properties
			reduction.interp_path.append(get_sec_props_obj(ref, reduction.mechs_gbars_dict,
										['pathL_elec'], ['pathLelec0', 'pathLelec1']))
	
		


def prepare_folds_impl(reduction):
	"""
	Prepare next collapse operation: assign topology information
	to each Section.

	(Implementation of interface declared in reduce_cell.CollapseReduction)

	NOTE: topology is ONLY used for when using multiple folding passes, 
	      for determining the folding branch point
	"""
	all_refs = reduction.all_sec_refs
	root_ref = reduction._soma_refs[0]

	# Assign topology info (order, level, strahler number)
	root_ref.order = 0
	root_ref.level = 0
	assign_topology_attrs(root_ref, all_refs, root_order=0)
	root_ref.strahlernumber = max((ref.strahlernumber for ref in all_refs))
	
	# For each segment: compute path length, path resistance, electrotonic path length
	for secref in reduction.all_sec_refs:
		redutils.sec_path_props(secref, f_lambda, reduction.gleak_name)
		secref.max_passes = 100


def calc_folds_impl(reduction, i_pass, dX=0.1):
	"""
	Collapse branches at branch points identified by given criterion.
	"""
	# Get sections
	allsecrefs = reduction.all_sec_refs

	# Find collapsable branch points
	# target_Y_secs = folding.find_collapsable(allsecrefs, i_pass, 'highest_level')
	target_Y_secs = reduction._fold_root_refs

	# Do collapse operation at each branch points
	clusters = calc_folds(target_Y_secs, i_pass, reduction, dX=dX)

	# Save results
	reduction.clusters = clusters


def make_folds_impl(reduction):
	"""
	Make equivalent Sections for branches that have been folded.
	"""
	clusters = reduction.clusters

	# Make new sections
	make_substitute_folds(clusters, reduction) # SectionRef stored on each Cluster

	# Update Sections
	new_dend_refs = sum((cluster.eq_refs for cluster in clusters), []) # concatenate lists
	reduction.update_refs(dend_refs=new_dend_refs) # prepare for next iteration