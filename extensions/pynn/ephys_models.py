"""
Module for working with BluePyOpt cell models in PyNN.

@author		Lucas Koelman
@date		14/02/2018


USEFUL EXAMPLES
---------------

https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/cells.py
https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/standardmodels/cells.py

"""

from pyNN.neuron.cells import NativeCellType


class EphysCellType(NativeCellType):
	"""
	Encapsulates a cell model of type bluepyopt.ephys.cellmodels.CellModel 
	for interoperability with PyNN.
	"""

	def __init__(self, ephys_cell):
		"""
		Create new PyNN cell type that encapsulates the given cell.
		"""
		pass

