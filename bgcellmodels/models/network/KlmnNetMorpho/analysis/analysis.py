"""
Data analysis for network simulation.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import neo.io

# import bgcellmodels.common.analysis as sigplot


def read_population_segments(outputs, is_directory=True, i_segment=0, read_ext=None):
    """
    Read Neo IO data into a dictionary mapping the population label
    to the first segment.

    @param  outputs : str or list(str)
            
            If is_directory is true, the output must be a string and is
            interpreted as a directory containing Neo output files.
            
            If is_directory is false, the output must be a ling of strings
            and is interpreted as the paths to the individual output files.


    @see    http://neo.readthedocs.io/en/latest/io.html

    @return     pops_segments : dict[str, neo.Segment]
                Mapping of population label to data segment.
    """
    if is_directory:
        filenames = os.listdir(outputs)
        pop_files = [os.path.join(outputs, f) for f in filenames]
        if read_ext is not None:
            pop_files = [f for f in pop_files if f.endswith(read_ext)]
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
        pops_segments[pop_label] = blocks[0].segments[i_segment]

    return pops_segments


def plot_population_spikes(pops_segments):
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

        i_pop += 1

    plt.show(block=False)


def plot_population_signals(pops_segments, max_num_signals=20, time_range=None):
    """
    Plot all recorded signals for each population.

    @param      pops_segments : Dict[str, neo.Segment]
                Mapping from population label to data segment
    """

    i_pop = 0
    for pop_label, segment in pops_segments.items():

        # one  separate figure per trace type
        for signal in segment.analogsignals:
            # NOTE: 'signal' is matrix[t, signal_id] with one signal per column

            # Make one axis per signal
            num_signals = min(signal.shape[1], max_num_signals)
            fig, axes = plt.subplots(num_signals, 1)
            fig.suptitle("Population {} - trace {}".format(
                            pop_label, signal.name))

            time_vec = signal.times
            if time_range is None:
                i_start = 0
                i_stop = time_vec.size
            else:
                i_after, = np.where(signal.times >= time_range[0])
                i_beyond, = np.where(signal.times >= time_range[1])
                i_start = i_after[0]
                i_stop = time_vec.size if len(i_beyond)==0 else i_beyond[0]+1
                
            for i_cell in range(num_signals):
                ax = axes[i_cell]
                if 'source_ids' in signal.annotations:
                    label = "id {}".format(signal.annotations['source_ids'][i_cell])
                else:
                    label = "cell {}".format(i_cell)
                
                ax.plot(time_vec[i_start:i_stop], signal[i_start:i_stop, i_cell], label=label)
                ax.set_ylabel("{} ({})".format(
                    signal.name, signal.units._dimensionality.string))
                ax.legend()

        i_pop += 1

    plt.show(block=False)


def conn_matrix_from_text(
        str_mat, 
        char_pos='O', char_zero='.', char_empty=' ',
        val_pos=1.0, val_zero=0.0, val_empty=0.0):
    """
    Convert printed connectivity matrix to numpy array.

    @param  str_mat : str OR unicode
            String representing connectivity matrix with rows separated
            by line breaks.
    """
    str_rows = str_mat.splitlines()
    utf_mat = np.array([[ch for ch in row] for row in str_rows], dtype=unicode)
    
    float_mat = np.zeros_like(utf_mat, dtype=float)
    float_mat[utf_mat == char_zero] = val_zero
    float_mat[utf_mat == char_pos] = val_pos
    float_mat[utf_mat == char_empty] = val_empty
    return float_mat


def plot_connectivity_matrix(
        float_mat=None, str_mat=None, 
        fpath=None, seaborn=False,
        pop_size=10, pop0='A', pop1='B'):
    """
    Plot connectivity matrix given as string.
    """
    if fpath is not None:
        with open(fpath, 'r') as fmat:
            str_mat = fmat.read()

    if float_mat is not None:
        W = float_mat
    elif not isinstance(str_mat, (str, unicode)):
        print("Please provide either matrix in string format or file path.")
    else:
        W = conn_matrix_from_text(str_mat)

    if seaborn:
        # Only looks good for small population size
        import seaborn as sns
        sns.set(style="white")

        # Set up the matplotlib figure
        f, ax = plt.subplots()

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(W,
                    xticklabels=range(pop_size),
                    yticklabels=range(pop_size),
                    mask=None, cmap=cmap, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

        plt.yticks(rotation=0)
        plt.title("Weight matrix")
        plt.show(block=False)
    else:
        from matplotlib import patches
        px = py = pop_size / 10.0

        # Create plot
        fig, ax = plt.subplots(figsize=(8,6))
        # # fig.subplots_adjust(left=0.02) # Less space on left
        # fig.subplots_adjust(right=0.98) # Less space on right
        # fig.subplots_adjust(top=0.96) # Less space on bottom
        # # fig.subplots_adjust(bottom=0.02) # Less space on bottom
        # fig.subplots_adjust(wspace=0) # More space between
        # fig.subplots_adjust(hspace=0) # More space between

        # Plot matrix as image
        colormap = plt.get_cmap('Oranges') # see https://matplotlib.org/examples/color/colormaps_reference.html
        plt.imshow(W, interpolation='none', cmap=colormap)

        # Plot grid lines
        y_popsize, x_popsize = W.shape
        y_nticks = y_popsize/py + 1
        x_nticks = x_popsize/px + 1
        yticks_pos = np.arange(y_nticks)*py
        xticks_pos = np.arange(x_nticks)*px
        for p in yticks_pos:
            # Plot gridlines (population boundaries)
            # plt.plot(np.array([0, x_popsize])-0.5, 
            #          np.array([p, p])-0.5,
            #          'k-', linewidth=1.0, snap=True)
            # plt.plot(np.array([p, p])-0.5,
            #          np.array([0, y_popsize])-0.5, 
            #          'k-', linewidth=1.0, snap=True)
            # Add rectangles on diagonal
            ax.add_patch(patches.Rectangle((p-0.5, p-0.5),
                                            px, # Width
                                            py, # Height
                                            facecolor="none",
                                            edgecolor='g',
                                            linewidth="1"))
        # Grid instead of manual gridlines
        plt.grid(True)

        # Configure the x and y axis
        ax.set_xticks(xticks_pos-0.5)
        ax.set_yticks(yticks_pos-0.5)
        ax.set_xticklabels(xticks_pos)
        ax.set_yticklabels(yticks_pos)
        ax.xaxis.set_ticks_position('top')

        ax.set_xlabel('{} cell index'.format(pop1))
        ax.set_ylabel('{} cell index'.format(pop0))
        
        plt.xlim(-0.5, x_popsize - 0.5)
        plt.ylim(y_popsize - 0.5 ,-0.5)

        # Add color bar to measure weights
        plt.clim(0, abs(W).max())
        plt.colorbar()
        plt.show(block=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot simulation data')

    parser.add_argument('-d', '--data', nargs=1, type=str,
                        metavar='/path/to/config.json',
                        dest='output_path',
                        help='Output file or directory')

    parser.add_argument('-e', '--ext', nargs=1, type=str,
                        metavar='.ext',
                        dest='extension',
                        default=['.mat'],
                        help='File extension, e.g. .mat')

    parser.add_argument('-n', '--nsig', nargs=1, type=int,
                        metavar='<max_num_signals>',
                        default=[10],
                        dest='nsig', help='Number of signal to plot')

    parser.add_argument('-t', '--trange', nargs=2, type=float,
                        default=[1e3, 5e3],
                        dest='trange', help='Time interval to plot')

    args = parser.parse_args() # Namespace object
    parsed_dict = vars(args) # Namespace to dict

    # Read and plot the data
    pops_segments = read_population_segments(
        parsed_dict['output_path'][0],
        read_ext=parsed_dict['extension'][0])
    
    plot_population_spikes(pops_segments)
    plot_population_signals(
        pops_segments, 
        max_num_signals=parsed_dict['nsig'][0],
        time_range=parsed_dict['trange'])