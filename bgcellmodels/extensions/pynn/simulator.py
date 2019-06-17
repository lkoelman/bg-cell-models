"""
Extensions to NEURON simulator for PyNN

@author     Lucas Koelman
@date       03/06/2019
"""

import os
from pyNN.neuron import simulator
from neuron import h

# Save original _pre_run function
simulator.state.__class__._old_pre_run = simulator.state.__class__._pre_run


def pre_run(self):
    """
    Add support for BBSaveState to pyNN.neuron.simulator._State

    @override   pyNN.neuron.simulator._State._pre_run

    @pre    if self.restore is True, the directory <working_directory/in> must
            exist and contain saved state files

    @see    https://www.neuron.yale.edu/neuron/static/py_doc/simctrl/bbsavestate.html
    """
    if not self.running:
        self._old_pre_run()

        # Restore simulator state
        self.bbss = h.BBSaveState()
        if getattr(self, 'restore', False):
            if not os.path.exists('./in'):
                raise ValueError('Simulator state files must be saved in ./in/')
            self.bbss.restore_test()
            self.bbss.vector_play_init()


def save_state(self):
    """
    Save NEURON simulator state using BBSaveState.

    @post   the directory <working_directory/out> contains state files 
    """
    if self.mpi_rank == 0:
        # BBSaveState expects directory 'out' to exist. So create if not present
        if not os.path.exists('./out'):
            os.makedirs('./out')

    self.bbss.save_test()


simulator.state.__class__._pre_run = pre_run
simulator.state.__class__.save_state = save_state