"""
Reduce model by folding/collapsing branches according to algorithm described in following articles:

- Stratford, K., Mason, A., Larkman, A., Major, G., and Jack, J. J. B. (1989) - The modelling of pyramidal neurones in the visual cortex

- Fleshman, J. W., Segev, I., and Burke, R. E. (1988) - Electrotonic architecture of type-identified alpha-motoneurons in the cat spinal cord. J. Neurophysiol. 60: 60 85.

- Clements, J., and Redman, S. (1989) - Cable properties of cat spinal motoneurones measured by combining voltage clamp, current clamp and intracellular staining. J. Physiol. 409: 63 87.

- Algorithm overview in 'Methods in Neural Modeling', Chapter 3.4, equations 3.21 - 3.22

@author Lucas Koelman

@date	28-08-2017
"""

import folding
from common.treeutils import next_segs, seg_index, interp_seg
from common.electrotonic import calc_lambda

# Global parameters
f_lambda = 0.0
alphabet_uppercase = [chr(i) for i in xrange(65,90+1)] # A-Z are ASCII 65-90

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
	gbar_list = reduction.active_gbars

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
		return
	
	# Get new segment at (a_i+1) * dX
	for b_seg in child_segs:

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

			# Ascend but don't increase index
			next_eq_diam(b_seg, a_i, root_X, dX, cluster, allsecrefs)

		else: # target_X <= b_X
			# target X value is between previous and next segment -> get new segment by interpolation
			
			# Examples: (s=last_seg, a=a_seg, b=b_seg, b=target_seg)
			#	[-----|-s-a-|t--b-] - [-----|-----|-----]
			#	[-----|---s-|a-t--] - [b----|-----|-----]
			
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
			step_attrs['step_gbar'] = [dict((gname, getattr(new_seg, gname) for gname in gbar_list))]

			# Save them on cluster object
			for attr_name, new_seg_attr in step_attrs.iteritems():

				# Get cluster attribute
				step_attr_list = getattr(cluster, attr_name)
				
				# Save values for new segment
				if len(step_attr_list) <= a_i:
					# First entry (first branch for this step) : start new list
					step_attr_list.append(new_seg_attr)
				else:
					# Subsequent branches: add to list for this step
					step_attr_list.extend(new_seg_attr)


			# Ascend and increase index
			next_eq_diam(new_seg, a_i+1, root_X, dX, cluster, allsecrefs)


def calc_collapses(target_Y_secs, i_pass, allsecrefs, dX=0.1):
	"""
	Do folding operation

	@return			list of Cluster objects with properties of equivalent
					Section for each set of collapsed branches.
	"""

	clusters = []
	for j_zip, root_ref in enumerate(target_Y_secs):

		# Fold name
		name_sanitized = re.sub(r"[\[\]\.]", "", root_ref.sec.name())
		fold_label = "foldAt_{0}".format(name_sanitized)

		# Make Cluster object that represents collapsed segments
		cluster = Cluster(fold_label)

		# Equivalent parameters
		cluster.step_diams = []
		cluster.step_Rm = []
		cluster.step_Ra = []
		cluster.step_cm = []

		# Topological info
		cluster.parent_seg = root_ref(1.0)

		# Ascend subtree from root and compute diameters
		start_seg = root_ref.sec(1.0) # end of cylinder
		start_X = root_ref.pathLelec1
		dX = 0.1
		next_eq_diam(start_seg, 0, start_X, dX, cluster, allsecrefs)

		# TODO: don't use as fine dX: check papers for what value they used
		#		- use course dX, then refine\
		#		- make one Section for


		# Finalize diam calculation
		cluster.num_sec = len(cluster.step_diams)
		for i_step, diams in enumerate(cluster.step_diams):

			# Number of parallel branches found at this step
			num_parallel = len(diams)

			# Equivalent properties at this step
			cluster.eq_diam[i_step] = sum((d**(3.0/2.0) for d in diams)) ** (2.0/3.0)

			# Passive electrical properties
			cluster.eq_Rm[i_step] = sum(cluster.step_Rm[i_step]) / num_parallel
			cluster.eq_Ra[i_step] = sum(cluster.step_Ra[i_step]) / num_parallel
			cluster.eq_cm[i_step] = sum(cluster.step_cm[i_step]) / num_parallel

			# Physical length 
			# NOTE: it is important to use the same lambda equation as the one
			#       used for electrotonic path lengths
			cluster.eq_lambda[i_step] = calc_lambda(f_lambda, cluster.eq_diam[i_step], cluster.eq_Ra[i_step], 
													1.0/cluster.eq_Rm[i_step], cluster.eq_cm[i_step])
			
			cluster.eq_L[i_step] = dX * cluster.eq_lambda[i_step]

		# Save cluster
		clusters.append(cluster)


def start_new_section(cluster, i_step, j_sec):
	"""
	Start new Section for discretization step i of the cluster.
	"""
	sec, ref = treeutils.create_hoc_section("{}_{}".format(
								cluster.label, alphabet_uppercase[j_sec]))
	
	sec.L		= cluster.eq_L[i_step]
	sec.diam	= cluster.eq_diam[i_step]
	sec.Ra		= cluster.eq_Ra[i_step]
	sec.cm		= cluster.eq_cm[i_step]

	setattr(sec(0.5), cluster.gleak_name, 1.0/cluster.eq_Rm[i_step])

	return sec, ref


def sub_fold_equivalents(clusters, orsecrefs, mechs_chans):
	"""
	Make equivalent Sections for each cluster, and substitute them into cell
	to replace the folded compartments.
	"""

	# Create one Section for each region with uniform diam
	# (uniform diam also means uniform length: see equation for physical length)
	diam_tolerance = 0.05 # 5 percent
	eq_secs = []
	eq_refs = []

	for i_clu, cluster in enumerate(clusters):

		# Save equivalent sections
		clu_eq_secs = []
		clu_eq_refs = []

		# Create first Section for this fold
		num_fold_secs = 0
		cur_diam = cluster.eq_diam[0]
		cur_sec, cur_ref = start_new_section(cluster, 0, num_fold_secs)
		cur_sec.connect(cluster.parent_seg, 0.0) # for electrotonic properties
		clu_eq_secs.append(cur_sec)
		clu_eq_refs.append(cur_ref)

		# Create Section for each region with uniform diam
		for i_step, diam in enumerate(cluster.eq_diam):

			if (1.0 - diam_tolerance) <= (diam / cur_diam) <= (1.0 + diam_tolerance):
				
				# Uniform: extend current Section with new segment
				cur_sec.L			= cur_sec.L + cluster.eq_L[i_step]
				cur_sec.nseg		= cur_sec.nseg + 1
				cur_sec(1.0).diam	= cluster.eq_diam[i_step]
				cur_sec(1.0).cm		= cluster.eq_cm[i_step]
				setattr(cur_sec(1.0), cluster.gleak_name, 1.0/cluster.eq_Rm[i_step])	

			else:
				
				# Not uniform: start new Section
				num_fold_secs += 1
				cur_diam = diam
				new_sec, new_ref = start_new_section(cluster, i_step, num_fold_secs)
				clu_eq_secs.append(new_sec)
				clu_eq_refs.append(new_ref)

				# Connect to previous sections
				new_sec.connect(cur_sec(1.0), 0.0)

		# Set active properties using interpolation
		glist = [gname+'_'+mech for mech,chans in mechs_chans.iteritems() for gname in chans]
		glist.remove(cluster.gleak_name)

		# TODO: interpolate active poperties (along predefined path, or use average, or just one entry in list)

		# Attach to subtree root segment and disconnect substituted Sections
		redtools.sub_equivalent_Y_sec(clu_eq_secs[0], cluster.parent_seg, [], 
								orsecrefs, mechs_chans, delete_substituted=True)



################################################################################
# Interface Functions (FoldReduction)
################################################################################


def preprocess_impl(reduction):
	"""
	Preprocess cell for Stratford reduction.

	(Implementation of interface declared in reduce_cell.CollapseReduction)

	@param	reduction		reduce_cell.CollapseReduction object
	"""
	pass


def prepare_folds_impl(reduction):
	"""
	Prepare next collapse operation: assign topology information
	to each Section.

	(Implementation of interface declared in reduce_cell.CollapseReduction)
	"""	
	
	# For each segment: compute path length, path resistance, electrotonic path length
	for secref in reduction.all_sec_refs:
		redtools.sec_path_props(secref, f_lambda, gleak_name)


def calc_folds_impl(reduction, i_pass, dX=0.1):
	"""
	Collapse branches at branch points identified by given criterion.
	"""
	# Get sections
	allsecrefs = reduction.all_sec_refs

	# Find collapsable branch points
	target_Y_secs = folding.find_collapsable(allsecrefs, i_pass, 'highest_level')

	# Do collapse operation at each branch points
	clusters = calc_collapses(target_Y_secs, i_pass, allsecrefs, dX=dX)

	# Save results
	reduction.clusters = clusters


def make_folds_impl(reduction):
	"""
	Make equivalent Sections for branches that have been folded.
	"""
	pass