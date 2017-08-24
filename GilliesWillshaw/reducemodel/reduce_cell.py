"""
Object-oriented interface for various compartmental cell reduction methods.

@author Lucas Koelman
@data	24-08-2017
"""

from enum import Enum, IntEnum, unique

from common.treeutils import ExtSecRef, getsecref
from neuron import h

import reduce_marasco as marasco

@unique
class ReductionMethod(Enum):
	Rall = 0
	StratfordMason = 1		# Stratford, K., Mason, A., Larkman, A., Major, G., and Jack, J. J. B. (1989) - The modelling of pyramidal neurones in the visual cortex
	BushSejnowski = 2		# Bush, P. C. & Sejnowski, T. J. Reduced compartmental models of neocortical pyramidal cells. Journal of Neuroscience Methods 46, 159–166 (1993).
	Marasco = 3				# Marasco, A., Limongiello, A. & Migliore, M. Fast and accurate low-dimensional reduction of biophysically detailed neuron models. Scientific Reports 2, (2012).


class CollapsingReduction(object):
	"""
	Class grouping methods and data used for reducing
	a compartmental cable model of a NEURON cell.
	"""

	_PREPROC_FUNCS = {
		ReductionMethod.Marasco:	marasco.preprocess_impl,
	}

	_PREP_COLLAPSE_FUNCS = {
		ReductionMethod.Marasco:	marasco.prepare_collapse_impl,
	}

	def __init__(self, soma_secs, dends_secs, subtree_root_secs):
		"""
		Initialize reduction of NEURON cell with given root Section.

		@param	soma_secs		list of root Sections for the cell (up to first branch points).
								This list must not contain any Section in dends_secs

		@param	dends_secs		list of dendritic section for each trunk section,
								i.e. the lists are non-overlapping / containing
								unique Sections

		"""

		# Find true root section
		first_root_sec = soma_secs[0]
		first_root_ref = ExtSecRef(sec=soma_secs[0])
		root_sec = first_root_ref.root
		h.pop_section()

		# Save unique sections
		self._soma_refs = [ExtSecRef(sec=sec) for sec in soma_secs]
		self._dends_refs = [[ExtSecRef(sec=sec) for sec in dend] for dend in dends_secs]

		# Save root sections
		self._root_ref = getsecref(root_sec, self._soma_refs)
		allsecrefs = all_sec_refs
		self._subtree_root_refs = [getsecref(sec, allsecrefs) for sec in subtree_root_secs]


	@property
	def all_sec_refs(self):
	    return list(self._soma_refs) + sum(self._dends_refs, [])
	


	def preprocess_cell(self, red_method):
		"""
		Pre-process cell: calculate properties & prepare data structures
		for reduction procedure

		@param	red_method		ReductionMethod instance: the reduction method that we
								should preprocess for.

		@pre		The somatic and dendritic sections have been set

		@post		Computed properties will be available as attributes
					on Section references in _soma_refs and _dends_refs,
					in addition to other side effects specified by the
					specific preprocessing function called.
		"""

		try:
			preproc_func = self._PREPROC_FUNCS[red_method]
		except KeyError:
			raise NotImplementedError("Preprocessing function for reduction method {} not implemented".format(red_method))
		else:
			preproc_func(self)


	def prepare_collapse(self):
		"""
		Prepare next collapse operation.
		"""
		
		try:
			prepare_func = self._PREP_COLLAPSE_FUNCS[red_method]
		except KeyError:
			raise NotImplementedError("Collapse preparation function for reduction method {} not implemented".format(red_method))
		else:
			prepare_func(self)


	def calculate_collapse(self):
		"""
		Collapse branches at branch points identified by given criterion.
		"""
		pass

	def substitute_collapse(self):
		pass


	def build_equivalent(self):
		"""
		Build equivalent Sections for branches that have been previously collapsed.
		"""


################################################################################
# Reduction Experiments
################################################################################

def collapse_gillies_marasco(export_locals=True):
	"""
	Collapse Gillies STN model using Marasco (2012) algorithm.
	"""

	if export_locals:
		globals().update(locals())


if __name__ == '__main__':
	collapse_gillies_marasco()