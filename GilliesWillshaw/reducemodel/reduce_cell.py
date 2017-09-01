"""
Object-oriented interface for various compartmental cell reduction methods.

@author Lucas Koelman
@data	24-08-2017
"""

from enum import Enum, IntEnum, unique

from common.treeutils import ExtSecRef, getsecref
from neuron import h

import marasco_foldbased as marasco
import stratford_folding as stratford

# logging of DEBUG/INFO/WARNING messages
import logging
logging.basicConfig(format='%(levelname)s:%(message)s @%(filename)s:%(lineno)s', level=logging.DEBUG)
logname = "reduction" # __name__
logger = logging.getLogger(logname) # create logger for this module

@unique
class ReductionMethod(Enum):
	Rall = 0
	Stratford = 1			# Stratford, K., Mason, A., Larkman, A., Major, G., and Jack, J. J. B. (1989) - The modelling of pyramidal neurones in the visual cortex
	BushSejnowski = 2		# Bush, P. C. & Sejnowski, T. J. Reduced compartmental models of neocortical pyramidal cells. Journal of Neuroscience Methods 46, 159-166 (1993).
	Marasco = 3				# Marasco, A., Limongiello, A. & Migliore, M. Fast and accurate low-dimensional reduction of biophysically detailed neuron models. Scientific Reports 2, (2012).


class FoldReduction(object):
	"""
	Class grouping methods and data used for reducing
	a compartmental cable model of a NEURON cell.
	"""

	# For each step in reduction process: a dict mapping ReductionMethod -> (func, arg_names)
	
	_PREPROC_FUNCS = {
		ReductionMethod.Marasco:	(marasco.preprocess_impl, []),
		ReductionMethod.Stratford:	(stratford.preprocess_impl, []),
	}

	_PREP_FOLD_FUNCS = {
		ReductionMethod.Marasco:	(marasco.prepare_folds_impl, []),
		ReductionMethod.Stratford:	(stratford.prepare_folds_impl, []),
	}

	_CALC_FOLD_FUNCS = {
		ReductionMethod.Marasco:	(marasco.calc_folds_impl, []),
		ReductionMethod.Stratford:	(stratford.calc_folds_impl, ['dX']),
	}

	_MAKE_FOLD_EQ_FUNCS = {
		ReductionMethod.Marasco:	(marasco.make_folds_impl, []),
		ReductionMethod.Stratford:	(stratford.make_folds_impl, []),
	}

	# Make accessible by step
	_REDUCTION_STEPS_FUNCS = {
		'preprocess':		_PREPROC_FUNCS,
		'prepare_fold':		_PREP_FOLD_FUNCS,
		'calculate_fold':	_CALC_FOLD_FUNCS,
		'make_fold':		_MAKE_FOLD_EQ_FUNCS,
	}



	def __init__(self, soma_secs, dend_secs, fold_root_secs, method):
		"""
		Initialize reduction of NEURON cell with given root Section.

		@param	soma_secs		list of root Sections for the cell (up to first branch points).
								This list must not contain any Section in dend_secs

		@param	dend_secs		list of dendritic section for each trunk section,
								i.e. the lists are non-overlapping / containing
								unique Sections

		@param	method			ReductionMethod instance
		"""

		# Parameters for reduction method (set by user)
		self._REDUCTION_PARAMS = dict((method, {}) for method in list(ReductionMethod))

		# Reduction method
		self.active_method = method
		self._mechs_gbars_dict = None

		# Find true root section
		first_root_sec = soma_secs[0]
		first_root_ref = ExtSecRef(sec=soma_secs[0])
		root_sec = first_root_ref.root # pushes CAS
		
		# Save unique sections
		self._soma_refs = [ExtSecRef(sec=sec) for sec in soma_secs]
		self._dend_refs = [ExtSecRef(sec=sec) for sec in dend_secs]

		# Save ion styles
		ions = ['na', 'k', 'ca']
		self._ion_styles = dict(((ion, h.ion_style(ion+'_ion')) for ion in ions))
		h.pop_section() # pops CAS

		# Save root sections
		self._root_ref = getsecref(root_sec, self._soma_refs)
		allsecrefs = self.all_sec_refs
		self._fold_root_refs = [getsecref(sec, allsecrefs) for sec in fold_root_secs]


	@property
	def all_sec_refs(self):
		return list(self._soma_refs) + list(self._dend_refs)


	def get_mechs_gbars_dict(self):
		"""
		Get dictionary of mechanism names and their conductances.
		"""
		return self._mechs_gbars_dict
	

	def set_mechs_gbars_dict(self, val):
		"""
		Set mechanism names and their conductances
		"""
		self._mechs_gbars_dict = val
		self.active_gbar_names = [gname+'_'+mech for mech,chans in val.iteritems() for gname in chans]
		self.active_gbar_names.remove(self.gleak_name)

	# make property
	mechs_gbars_dict = property(get_mechs_gbars_dict, set_mechs_gbars_dict)


	def update_refs(self, soma_refs=None, dend_refs=None):
		"""
		Update Section references after sections have been created/destroyed/substituted.

		@param soma_refs	list of SectionRef to at least all new soma sections
							(may also contain existing sections)

		@param dend_refs	list of SectionRef to at least all new dendritic sections
							(may also contain existing sections)
		"""
		# Destroy references to deleted sections
		self._soma_refs = [ref for ref in self._soma_refs if ref.exists()]
		self._dend_refs = [ref for ref in self._dend_refs if ref.exists()]
		
		# Add newly created sections
		if soma_refs is not None:
			self._soma_refs = list(set(self._soma_refs + soma_refs)) # get unique references

		if dend_refs is not None:
			self._dend_refs = list(set(self._dend_refs + dend_refs)) # get unique references


	def set_reduction_params(self, method, params):
		"""
		Set parameters for given reduction method.
		"""
		self._REDUCTION_PARAMS[method] = params


	def _exec_reduction_step(self, step, method, step_args=None):
		"""
		Execute reduction step 'step' using method 'method'
		"""
		try:
			func, arg_names = self._REDUCTION_STEPS_FUNCS[step][method]
		
		except KeyError:
			raise NotImplementedError("{} function not implemented for "
									  "reduction method {}".format(step, method))
		
		else:
			user_params = self._REDUCTION_PARAMS[method]
			user_kwargs = dict((kv for kv in user_params.iteritems() if kv[0] in arg_names)) # get required args
			
			if step_args is None:
				step_args = []

			func(*step_args, **user_kwargs)


	def preprocess_cell(self, method):
		"""
		Pre-process cell: calculate properties & prepare data structures
		for reduction procedure

		@param	method		ReductionMethod instance: the reduction method that we
								should preprocess for.

		@pre		The somatic and dendritic sections have been set

		@post		Computed properties will be available as attributes
					on Section references in _soma_refs and _dend_refs,
					in addition to other side effects specified by the
					specific preprocessing function called.
		"""
		self._exec_reduction_step('preprocess', method, step_args=[self])


	def prepare_folds(self, method):
		"""
		Prepare next fold operation.
		"""
		self._exec_reduction_step('prepare_fold', method, step_args=[self])


	def calc_folds(self, method, i_pass):
		"""
		Fold branches at branch points identified by given criterion.
		"""
		self._exec_reduction_step('calculate_fold', method, step_args=[self, i_pass])


	def make_fold_equivalents(self, method):
		"""
		Make equivalent Sections for branches that have been folded.
		"""
		self._exec_reduction_step('make_fold', method, step_args=[self])


	def reduce_model(self, num_passes, method=None):
		"""
		Do a fold-based reduction of the compartmental cell model.

		@param	num_passes		number of 'folding' passes to be done. One pass corresponds to
								folding at a particular node level (usually the highest).
		"""
		if method is None:
			method = self.active_method

		# Start reduction process
		self.preprocess_cell(method)

		# Fold one pass at a time
		for i_pass in xrange(num_passes):
			self.prepare_folds(method)
			self.calc_folds(method, i_pass)
			self.make_fold_equivalents(method)





################################################################################
# Reduction Experiments
################################################################################

def fold_gillies_stratford(export_locals=True):
	"""
	Fold Gillies STN model using given reduction method
	
	@param	export_locals		if True, local variables will be exported to the global
								namespace for easy inspection
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
	dendL_juction		= h.SThcell[0].dend0[0]
	dendL_upper_root	= h.SThcell[0].dend0[1] # root section of upper left dendrite
	dendL_lower_root	= h.SThcell[0].dend0[2] # root section of lower left dendrite
	fold_roots = [dendR_root, dendL_upper_root, dendL_lower_root]

	# Reduce model
	red_method = ReductionMethod.Stratford
	reduction = FoldReduction([soma], dends, fold_roots, red_method)
	reduction.gleak_name = gillies_model.gleak_name
	reduction.mechs_gbars_dict = gillies_model.gillies_gdict
	reduction.reduce_model(num_passes=1)

	logger.debug("Successfully folded Gillies model using {}!".format(red_method))

	if export_locals:
		globals().update(locals())

	return reduction._soma_refs, reduction._dend_refs


def fold_gillies_marasco(export_locals=True):
	"""
	Fold Gillies STN model using given reduction method
	
	@param	export_locals		if True, local variables will be exported to the global
								namespace for easy inspection
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
	dendL_juction		= h.SThcell[0].dend0[0]
	dendL_upper_root	= h.SThcell[0].dend0[1] # root section of upper left dendrite
	dendL_lower_root	= h.SThcell[0].dend0[2] # root section of lower left dendrite
	fold_roots = [dendR_root, dendL_upper_root, dendL_lower_root]

	# Reduce model
	red_method = ReductionMethod.Marasco
	reduction = FoldReduction([soma], dends, fold_roots, red_method)
	reduction.gleak_name = gillies_model.gleak_name
	reduction.mechs_gbars_dict = gillies_model.gillies_gdict
	reduction.reduce_model(num_passes=7)
	reduction.update_refs()

	logger.debug("Successfully folded Gillies model using {}!".format(red_method))

	if export_locals:
		globals().update(locals())

	return reduction._soma_refs, reduction._dend_refs


if __name__ == '__main__':
	fold_gillies_marasco()