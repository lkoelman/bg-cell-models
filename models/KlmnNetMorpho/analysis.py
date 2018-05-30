"""
Data analysis for network simulation.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import neo.io

# import common.analysis as sigplot


def read_populaton_segments(outputs, is_directory=True, i_segment=0):
    """
    Read Neo IO data into a dictionary mapping the population label
    to the first segment.

    @param  outputs : str or list(str)
            
            If is_directory is true, the output must be a string and is
            interpreted as a directory containing Neo output files.
            
            If is_directory is false, the output must be a ling of strings
            and is interpreted as the paths to the individual output files.


    @see    http://neo.readthedocs.io/en/latest/io.html
    """
    if is_directory:
        filenames = os.listdir(outputs)
        pop_files = [os.path.join(outputs, f) for f in filenames]
    else:
        pop_files = outputs
    pops_segments = {}

    for pop_file in pop_files:
        # Guess filetype and associated reader object from extension
        reader = neo.io.get_io(pop_file)

        blocks = reader.read()
        assert len(blocks) == 1, "More than one Neo Block in file."
        pop_label = blocks[0].name

        if len(blocks[0].segments)-1 < i_segment:
            raise ValueError("Segment index greater than number of Neo segments"
                             " in file {}".format(pop_file))

        if pop_label in pops_segments:
            raise ValueError("Duplicate population labels in files")
        pops_segments[pop_label] = blocks[0].segments[0]

    return pops_segments


def plot_population_signals(pops_segments):
    """
    Plot all recorded signals for each population.

    @param      pops_segments : Dict[str, neo.Segment]
                Mapping from population label to data segment
    """

    num_pops = len(pops_segments)

    # Plot spikes
    fig_spikes, axes_spikes = plt.subplots(num_pops, 1)
    fig_spikes.suptitle('Spikes for each population')

    i_pop = 0
    pop_spike_colors = 'rgcbm'
    for pop_label, segment in pops_segments.items():

        ax = axes_spikes[i_pop]
        for i_train, spiketrain in enumerate(segment.spiketrains):
            y = spiketrain.annotations.get('source_id', i_train)
            y_vec = np.ones_like(spiketrain) * y
            ax.plot(spiketrain, y_vec,
                    marker='|', linestyle='', snap=True,
                    color=pop_spike_colors[i_pop % 5])
            ax.set_ylabel(pop_label)


        for signal in segment.analogsignals:
            # one figure per trace type
            fig, axes = plt.subplots(signal.shape[1], 1)
            fig.suptitle("Population {} - trace {}".format(
                            pop_label, signal.name))

            # signal matrix has one cell signal per column
            time_vec = signal.times
            y_label = "{} ({})".format(signal.name, signal.units._dimensionality.string)

            for i_cell in range(signal.shape[1]):
                ax = axes[i_cell]
                if 'source_ids' in signal.annotations:
                    label = "id {}".format(signal.annotations['source_ids'][i_cell])
                else:
                    label = "cell {}".format(i_cell)
                
                ax.plot(time_vec, signal[:, i_cell], label=label)
                ax.set_ylabel(y_label)
                ax.legend()

        i_pop += 1

    plt.show(block=False)