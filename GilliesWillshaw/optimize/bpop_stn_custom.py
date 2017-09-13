"""
Optimization of reduced neuron model using BluePyOpt.

@author	Lucas Koelman

@date	12/09/2017



ARCHITECTURE


- old strategy: strategy in optimize_incremental.py is to make number of reduction passes a parameter of the optimization
	- if this is higher than current number of passes: re-initialize cell model and reduce again
	- the run() method calls build_candidate(genes) and then runs the protocol


- new strategy B: copy structure from old optimize_incremental.py and don't use BluePyOpt classes

	+ only use the DEAP + eFEL modules but not the cell model, objectives, etc, like in example at https://github.com/BlueBrain/eFEL/blob/master/examples/deap/deap_efel_neuron2.ipynb

	+ CONS:
		* need deeper understanding of DEAP algorithm to choose suitable parameters

	+ PROS:
		* can easily reuse existing code
"""

