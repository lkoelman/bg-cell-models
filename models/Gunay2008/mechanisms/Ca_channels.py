import solvers.mechanism as mechanism
from solvers.mechanism import (
    State, Current, Parameter,
    state_derivative, state_steadystate, state_timeconst,
    mV, ms, S, cm2, nodim,
    v, exp, log,
)

nrn_mechs = [ # suffixes for Ca-channels
    'CaHVA',
    'Calcium', # not a channel; calcium buffering mechanism
]

def plot_IV_curves():
    """
    Plot I-V curves for K channel mechanisms
    """
    from common.channel_analysis import ivcurve

    from matplotlib import pyplot as plt
    import neuron
    h = neuron.h

    # Load NEURON libraries, mechanisms
    # import os
    # nrn_dll_path = os.path.dirname(__file__)
    # neuron.load_mechanisms(nrn_dll_path)

    # h.CVode().active(1)

    for mech_name in nrn_mechs:
        print('Generating I-V curve for {} ...'.format(mech_name))
        
        if mech_name=='Calcium':
            quantity = 'cai'
            descr = 'current (mA/cm^2)'
        else:
            quantity = 'ica'
            descr = 'concentration (mM)'
        
        ik, v = ivcurve(mech_name, quantity)

        plt.figure()
        plt.plot(v, ik, label=quantity+'_'+mech_name)

        plt.suptitle(mech_name)
        plt.xlabel('v (mV)')
        plt.ylabel(descr)
        plt.legend()
        plt.show(block=False)

if __name__ == '__main__':
    # Plot response of mod file channel
    plot_IV_curves()
