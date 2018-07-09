"""
Modifications to PyNN Population class.
"""

from pyNN.neuron import Population as NrnPopulation
from extensions.pynn.recording import TraceSpecRecorder

# Monkey-patching of pyNN.neuron.Population class
# pyNN.neuron.Population._recorder_class = TraceSpecRecorder


class Population(NrnPopulation):
    """
    Population class with custom Recorder that accepts NetPyne-style
    trace specifications, and passes position updates on to its cells.
    """

    _recorder_class = TraceSpecRecorder
    all_populations = []


    def __init__(self, *args, **kwargs):
        """
        Initialize population and append to class variable 'all_populations'.
        """
        super(Population, self).__init__(*args, **kwargs)
        Population.all_populations.append(self)


    @NrnPopulation.positions.setter
    def _set_positions(self, pos_array):
        """
        Update cell positions and notify each cell of position update.

        WARNING: positions is assigned before cells are created, so this does
                 NOT call cell_model._update_position() when the Population
                 is initialized with a spatial structure argument.
        """
        super(Population, self)._set_positions(pos_array) # original setter
        if getattr(self, 'all_cells', None) is None:
            return
        for i, (cell_id, is_local) in enumerate(zip(self.all_cells, self._mask_local)):
            if is_local and hasattr(cell_id._cell, '_update_position'):
                cell_id._cell._update_position(pos_array[:, i])



    def _create_cells(self):
        """
        Override Population._create_cells() so that individual cells are
        notified of their 3D position assigned by PyNN.
        """
        super(Population, self)._create_cells()
        pos_array = self.positions # numpy array of shape (3, N)
        for i, (cell_id, is_local) in enumerate(zip(self.all_cells, self._mask_local)):
            if is_local:
                if hasattr(cell_id._cell, '_update_position'):
                    cell_id._cell._update_position(pos_array[:, i])
                if hasattr(cell_id._cell, '_init_lfp'):
                    cell_id._cell._init_lfp()


    # def calculate_lfp(self):
    #     """
    #     Calculate LFP contribution for each cell.
    #     Only used in manual LFP calculation.
    #     """
    #     for (cell_id, is_local) in zip(self.all_cells, self._mask_local):
    #         if is_local:
    #             cell_id._cell._set_imemb_ptr()
    #             cell_id._cell._calculate_lfp()