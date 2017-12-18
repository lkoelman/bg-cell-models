"""
Test suite for reduce_cell.py : reductions of specific cell models.
"""

from cersei.collapse.fold_reduction import ReductionMethod, FoldReduction, logger
from common.treeutils import getsecref

from neuron import h

# Make Gillies model files findable
gillies_model_dir = "../GilliesWillshaw"
import sys
sys.path.append(gillies_model_dir)

################################################################################
# Cell model-specific implementations of reduction functions
################################################################################

def assign_new_identifiers(node_ref, all_refs, par_ref=None):
	"""
	Assign identifiers to newly created Sections

	@note	assigned to key 'assign_new_identifiers_func'
	"""
	if not hasattr(node_ref, 'tree_index'):
		node_ref.tree_index = par_ref.tree_index
	
	if not hasattr(node_ref, 'table_index'):
		node_ref.table_index = -1 # unassigned

	# assign a unique cell GID
	if not hasattr(node_ref, 'gid'):
		if node_ref.table_index >= 0:
			node_ref.gid = min(0,node_ref.tree_index)*100 + node_ref.table_index
		else:
			node_ref.gid = node_ref.zip_id

	childsecs = node_ref.sec.children()
	childrefs = [getsecref(sec, all_refs) for sec in childsecs]
	for childref in childrefs:
		assign_new_identifiers(childref, all_refs, parref=node_ref)


def assign_initial_identifiers(reduction):
	"""
	Assign identifiers to Sections.

	@note	assigned to key 'assign_initial_identifiers_func'

	@post	all SectionRef.gid attributes are set
	"""

	allsecrefs = reduction.all_sec_refs

	dendL_secs = list(h.SThcell[0].dend0)
	dendR_secs = list(h.SThcell[0].dend1)
	dend_lists = [dendL_secs, dendR_secs]

	# Assign indices used in Gillies code (sth-data folder)
	def assign_original_indices(reduction):
		"""
		Assign 'STh indices' used in tables to look up section properties.
		"""
		for somaref in reduction._soma_refs:
			somaref.tree_index = -1
			somaref.table_index = 0

		for secref in reduction._dend_refs:
			for i_dend, dendlist in enumerate(dend_lists):
				if any([sec.same(secref.sec) for sec in dendlist]):
					secref.tree_index = i_dend
					secref.table_index= next((i+1 for i,sec in enumerate(dendlist) if sec.same(secref.sec)))
		
	# Assign unique GID to each Section
	assign_original_indices(reduction)
	assign_new_identifiers(reduction._root_ref, allsecrefs)


def get_interpolation_path_sections(reduction):
	"""
	Return sections along path from soma to distal end of dendrites used
	for interpolating dendritic properties.

	@note	assigned to key 'interpolation_path_func'
	"""
	# Choose stereotypical path for interpolation
	interp_tree_id = 1
	interp_table_ids = (1,3,8)
	path_secs = [secref for secref in reduction.all_sec_refs if (
					secref.tree_index == interp_tree_id and 
					secref.table_index in interp_table_ids)]
	return path_secs


def set_ion_styles(reduction):
	"""
	Set correct ion styles for each Section.

	@note	assigned to key 'set_ion_styles_func'
	"""
	# Set ion styles
	for sec in reduction.all_sec_refs:
		sec.push()
		h.ion_style("na_ion",1,2,1,0,1)
		h.ion_style("k_ion",1,2,1,0,1)
		h.ion_style("ca_ion",3,2,1,1,1)
		h.pop_section()


def fix_topology_below_roots(reduction):
	"""
	Assign topology numbers for sections located below the folding roots.

	@note	assigned to key 'fix_topology_func'
	"""
	allsecrefs = reduction.all_sec_refs
	soma_refs = reduction._soma_refs

	dendL_juction = getsecref(h.SThcell[0].dend0[0], allsecrefs)
	dendL_upper_root = getsecref(h.SThcell[0].dend0[1], allsecrefs)

	dendL_juction.order = 1
	dendL_juction.level = 0
	dendL_juction.strahlernumber = dendL_upper_root.strahlernumber+1
	
	for somaref in soma_refs:
		somaref.order = 0
		somaref.level = 0
		somaref.strahlernumber = dendL_juction.strahlernumber


################################################################################
# Cell model-specific tweaks
################################################################################

def adjust_gbar_spontaneous(reduction):
	"""
	Adjust gbar (NaL) to fix spontaneous firing rate.

	@note	put in list assigned to key 'post_tweak_funcs'

	@note	this is a manual model parameter tweak and should not be considered 
			part of the reduction
	"""
	# Apply correction (TODO: remove this after fixing reduction)
	for ref in reduction.all_sec_refs:
		sec = ref.sec
		
		if sec.name().endswith('soma'):
			print("Skipping soma")
			continue
		
		logger.anal("Scaled gna_NaL in section {}".format(sec))
		for seg in sec:
			# seg.gna_NaL = 1.075 * seg.gna_NaL
			seg.gna_NaL = 8e-6 * 1.3 # full model value = uniform 8e-6


################################################################################
# Gillies Model Reduction Experiments
################################################################################


def gillies_marasco_reduction(tweak=True):
	"""
	Make FoldReduction object with Marasco method.
	"""

	import gillies_model
	if not hasattr(h, 'SThcell'):
		gillies_model.stn_cell_gillies()

	# Make sections accesible by name and index
	soma = h.SThcell[0].soma
	dendL = list(h.SThcell[0].dend0) # 0 is left tree
	dendR = list(h.SThcell[0].dend1) # 1 is right tree
	dends = dendL + dendR

	# Get references to root sections of the 3 identical trees
	dendR_root			= h.SThcell[0].dend1[0]
	dendL_upper_root	= h.SThcell[0].dend0[1] # root section of upper left dendrite
	dendL_lower_root	= h.SThcell[0].dend0[2] # root section of lower left dendrite
	fold_roots = [dendR_root, dendL_upper_root, dendL_lower_root]

	# Parameters for reduction
	def stn_setstate():
		# Initialize cell for analyzing electrical properties
		h.celsius = 35
		h.v_init = -68.0
		h.set_aCSF(4)
		h.init()

	# Reduce model
	red_method = ReductionMethod.Marasco
	reduction = FoldReduction([soma], dends, fold_roots, red_method)

	# Reduction parameters
	reduction.gleak_name = gillies_model.gleak_name
	reduction.mechs_gbars_dict = gillies_model.gillies_gdict
	reduction.set_reduction_params(red_method, {
		'Z_freq' :				25.,
		'Z_linearize_gating' :	False,
		'gbar_scaling' :		'area',
		'syn_map_method' :		'Ztransfer',
		# CUSTOM FUNCTIONS #####################################################
		'Z_init_func' :						stn_setstate,
		'assign_initial_identifiers_func':	assign_initial_identifiers,
		'assign_new_identifiers_func':		assign_new_identifiers,
		'interpolation_path_func':			get_interpolation_path_sections,
		'set_ion_styles_func':			set_ion_styles,
		'fix_topology_func':			fix_topology_below_roots,
		'post_tweak_funcs' :			[adjust_gbar_spontaneous] if tweak else [],
	})

	return reduction


def fold_gillies_marasco(export_locals=True):
	"""
	Fold Gillies STN model using given reduction method
	
	@param	export_locals		if True, local variables will be exported to the global
								namespace for easy inspection
	"""
	# Make reduction object
	reduction = gillies_marasco_reduction()
	
	# Do reduction
	reduction.reduce_model(num_passes=7)
	reduction.update_refs()

	if export_locals:
		globals().update(locals())

	return reduction._soma_refs, reduction._dend_refs


if __name__ == '__main__':
	fold_gillies_marasco()