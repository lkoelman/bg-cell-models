"""
Optimization of reduced neuron model using BluePyOpt.

@author	Lucas Koelman

@date	12/09/2017



ARCHITECTURE

- How to make it incremental
	
	- old strategy: strategy in optimize_incremental.py is to make number of reduction passes a parameter of the optimization
		
		- if this is higher than current number of passes: re-initialize cell model and reduce again
		
		- the run() method calls build_candidate(genes) and then runs the protocol
	

	- new strategy A: one Optimization for one cell model created using EPhys module 

		- start from single collapse, save parameters, and write methods to transfer the 'DNA'/parameters to another cell model (re-use DNA from previous optimization)

		- will need to implement own parameters (that the DNA codes for)
			+ control everything via custom MetaParameters
			+ these can be transferred across cell models

		- CONS:
			* have to fit everything into straightjacket of BluePyOpt classes, will will require writing lots of new code, and hinder re-use of existing code

		- PROS:
			* can re-use Bpop examples and code structure for optimization



- Overview of optimization using bluepyopt.EPhys classes:

	- see examples:
		- 'simplecell.ipynb' at https://github.com/BlueBrain/BluePyOpt
		- 'L5PC.ipynb' at https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/L5PC.ipynb
		- MOOC Week 5 Graded exercise (simdata folder)

	- create cell model

	- for each StimulationProtocol: create one ephys.protocols.SweepProtocol
		
		+ allows to add generic stimuli
			- see http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.protocols.html
		
		+ can have multiple Stimuli objects + recording objects
			- http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.stimuli.html
			- http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.recordings.html


	- Define objectives (feature + target value/range)

		+ for each (feature, target_value) combination : create ephys.efeature.eFELFeature 
			- see http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.efeatures.html

		+ create a WeightedSumObjective to calculate fitness
			- http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.objectives.html


	- create Evaluator: ephys.evaluators.CellEvaluator
		+ http://bluepyopt.readthedocs.io/en/latest/ephys/bluepyopt.ephys.evaluators.html


	- create Optimization: bpop.deapext.optimisations.DEAPOptimisation

		+ see http://bluepyopt.readthedocs.io/en/latest/deapext/bluepyopt.deapext.optimisations.html#module-bluepyopt.deapext.optimisations

		+ IBEADEAPOptimisation is same as DEAPOptimisation (subclass without extras)


"""



import bluepyopt as bpop
import bluepyopt.ephys as ephys

from bpop_ext_gillies import StnReducedModel

# Gillies & Willshaw model mechanisms
from gillies_model import gleak_name

################################################################################
# MODEL REGIONS
################################################################################

# seclist_name are names of SectionList declared in the cell model we optimize

somatic_region = ephys.locations.NrnSeclistLocation('somatic', seclist_name='somatic')

dendritic_region = ephys.locations.NrnSeclistLocation('somatic', seclist_name='dendritic')

################################################################################
# MODEL PARAMETERS
################################################################################

# SOMATIC PARAMETERS

soma_rm_param = ephys.parameters.NrnSectionParameter(                                    
						name='gleak_soma',		# assigned name
						param_name=gleak_name,	# NEURON name
						locations=[somatic_region],
						bounds=[0.05, 0.125],	# TODO: set bounds
						frozen=False)

soma_cm_param = ephys.parameters.NrnSectionParameter(
						name='cm_soma',
						param_name='cm',
						bounds=[0.0, 1.0],		# TODO: set cm bounds
						locations=[somatic_region],
						frozen=False)


# DENDRITIC PARAMETERS

# TODO: choose right parameter type (look at instantiate() method)
dend_rm_factor

dend_cm_factor

dend_ra_param

# for set of most important active conductance: scale factor

################################################################################
# OPTIMIZATION EXPERIMENTS
################################################################################

def optimize_main():
	"""
	Optimize a reduced model.
	"""

	red_model = StnReducedModel(name='StnFolded', fold_method='marasco', num_passes=7)