"""
Creation of EFEL parameters for STN model optimization.

@author	Lucas Koelman

@date	3/10/2017

"""

import bluepyopt.ephys as ephys

from bpop_extensions import NrnScaleRangeParameter

# Gillies & Willshaw model mechanisms
import gillies_model
gleak_name = gillies_model.gleak_name

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

soma_gl_param = ephys.parameters.NrnSectionParameter(                                    
						name='gleak_soma',		# assigned name
						param_name=gleak_name,	# NEURON name
						locations=[somatic_region],
						bounds=[7.84112e-7, 7.84112e-3],	# default: 7.84112e-05
						frozen=False)

soma_cm_param = ephys.parameters.NrnSectionParameter(
						name='cm_soma',
						param_name='cm',
						bounds=[0.05, 10.0],	# default: 1.0
						locations=[somatic_region],
						frozen=False)


# DENDRITIC PARAMETERS

# For dendrites, conductances and cm have been scaled by the reduction method.
# We want to keep their spatial profile/distribution, i.e. just scale these.
# Hence: need to define our own parameter that scales these distributions.


dend_gl_factor = NrnScaleRangeParameter(
					name='gleak_dend_scale',
					param_name=gleak_name,
					bounds=[0.05, 10.0],
					locations=[dendritic_region],
					frozen=False)

dend_cm_factor = NrnScaleRangeParameter(
					name='cm_dend_scale',
					param_name=gleak_name,
					bounds=[0.05, 10.0],
					locations=[dendritic_region],
					frozen=False)

dend_ra_param = ephys.parameters.NrnSectionParameter(
					name='Ra_dend',
					param_name='Ra',
					bounds=[50, 500.0],			# default: 150.224
					locations=[dendritic_region],
					frozen=False)

# for set of most important active conductance: scale factor
scaled_gbar = ['gna_NaL', 'gk_Ih', 'gk_sKCa', 'gcaT_CaT', 'gcaL_HVA', 'gcaN_HVA']

# Make parameters to scale channel conductances
dend_gbar_params = []
for gbar_name in scaled_gbar:

	gbar_scale_param = NrnScaleRangeParameter(
						name		= gbar_name + '_dend_scale',
						param_name	= gbar_name,
						bounds		= [0.05, 10.0],
						locations	= [dendritic_region],
						frozen		= False)

	dend_gbar_params.append(gbar_scale_param)


# Groups of parameters to be used in optimizations
passive_params = [soma_gl_param, soma_cm_param, dend_gl_factor, dend_cm_factor, 
				dend_ra_param]

active_params = dend_gbar_params

all_params = passive_params + active_params


# Default values for parameters
default_params = {
	'gleak_soma':		7.84112e-05,
	'cm_soma':			1.0,
	'Ra_dend':			150.224,
	'cm_dend_scale':	1.0,
	'gleak_dend_scale':	1.0,
}
for param in dend_gbar_params:
	default_params[param.name] = 1.0