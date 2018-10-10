"""
Fujita, Kitano et al. (2011) GPe cell model for use in NEURON.

@author     Lucas Koelman

@date       14/09/2018

Usage
-----

Importing this module will make the cell model available in NEURON by loading
the cell template ('class' in usual OO terms).
"""

import neuron
h = neuron.h

# Load NEURON libraries, mechanisms
import os, os.path
script_dir = os.path.dirname(__file__)
neuron.load_mechanisms(os.path.join(script_dir, 'mechanisms'))

# Load Hoc functions for cell model
prev_cwd = os.getcwd()
os.chdir(script_dir)
h.xopen("fujita_createcell.hoc") # instantiates all functions & data structures on Hoc object
os.chdir(prev_cwd)