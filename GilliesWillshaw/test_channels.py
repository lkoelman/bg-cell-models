"""
Test gating and dynamics of various Na channel models

@author Lucas Koelman
@date   27-02-2017
"""

from test_stn_Gillies import * # as if we're working in that file

################################################################################
# Experiments
################################################################################

def test_spontaneous(soma, dends_locs, stims):
    """ Run rest firing experiment from original Hoc file

    @param soma         soma section

    @param dends_locs   list of tuples (sec, loc) containing a section
                        and x coordinate to place recording electrode

    @param stims        list of electrodes (IClamp)
    """
    # Set simulation parameters
    dur = 2000
    h.dt = 0.025
    h.celsius = 37 # different temp from paper (fig 3B: 25degC, fig. 3C: 35degC)
    h.v_init = -60 # paper simulations use default v_init
    set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

    # Recording: trace specification
    secs = {'soma': soma}
    traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)

    # Membrane voltages
    traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}
    traceSpecs['t_global'] = {'var':'t'}
    for i, (dend,loc) in enumerate(dends_locs):
        dendname = 'dend%i' % i
        secs[dendname] = dend
        traceSpecs['V_'+dendname] = {'sec':dendname,'loc':loc,'var':'v'}

    # Record ionic currents, open fractions, (in)activation variables
    rec_currents_activations(traceSpecs, 'soma', 0.5)

    # Set up recording vectors
    recordStep = 0.05
    recData = analysis.recordTraces(secs, traceSpecs, recordStep)

    # Simulate
    h.tstop = dur
    h.init() # calls finitialize() and fcurrent()
    h.run()

    # Plot membrane voltages
    recV = collections.OrderedDict([(k,v) for k,v in recData.iteritems() if k.startswith('V_')])
    figs_vm = analysis.plotTraces(recV, recordStep, yRange=(-80,40), traceSharex=True)
    vm_fig = figs_vm[0]
    vm_ax = figs_vm[0].axes[0]

    # Plot ionic currents, (in)activation variables
    figs, cursors = plot_currents_activations(recData, recordStep)

    # Save trace to file
    # V_soma = np.array(recData['V_soma'], ndmin=2)
    # T_soma = np.array(recData['t_global'], ndmin=2)
    # TV_soma = np.concatenate((T_soma, V_soma), axis=0) * 1e-3 # pyelectro expects SI units: seconds, Volts
    # fpath = 'C:\\Users\\lkoelman\\cloudstore_m\\simdata\\fullmodel\\spont_fullmodel_Vm_dt25e-3_0ms_2000ms.csv'
    # np.savetxt(fpath, TV_soma.T, delimiter=',', fmt=['%.3E', '%.7E'])

    plt.show(block=False)
    return recData, figs, cursors

def run_experimental_protocol():
    """
    Run one of the experiments
    """
    # Make dummy cell - will be voltage clamped so channels unimportant for dynamics
    all_Ra = 150.224
    all_cm = 1.0
    soma_L = 18.8
    soma_diam = 18.3112

    # Set channels and conductances
    mechs = list(gillies_mechs) # List of mechanisms to insert
    mechs.append([]) # TODO: insert mechanisms to test
    glist = [gname+'_'+mech for mech,chans in gillies_gdict.iteritems() for gname in chans]
    gbar_default = {
        'gna_Na':   1.483419823e-02, # global default var
        'gna_NaL':  1.108670852e-05, # global default var
        'gcaL_HVA': 0.0009500, # from file
        'gcaN_HVA': 0.0011539, # from file
        'gcaT_CaT': 0.0000000, # from file
        'gk_Ih':    0.0010059, # from file
        'gk_KDR':   0.0038429, # from file
        'gk_Kv31':  0.0134086, # from file
        'gk_sKCa':  0.0000684, # from file
    }

    # Create soma
    soma = h.Section()
    soma.nseg = 1
    soma.Ra = all_Ra
    soma.diam = soma_diam
    soma.L = soma_L
    soma.cm = all_cm
    for mech in mechs:
        soma.insert(mech)
    for k,v in gbar_default.iteritems():
        setattr(soma, k, v)
    setionstyles_gillies(soma)

    # Insert voltage clamp (space clamp)
    clamp = h.SEClamp(soma(0.5)) # NOTE: use SEClamp instead of IClamp to hyperpolarize to same level independent of passive parameters
    clamp.dur1 = 2000
    ptr = clamp._ref_amp1
    ptr_parent = clamp

    # Play vector into voltage clamp
    csv_path = "/Users/luye/cloudstore_m/simdata/fullmodel/spont_fullmodel_Vm_dt25e-3_0ms_2000ms.csv"
    t, v = np.loadtxt(csv_path, delimiter=',', usecols=(0, 1), unpack=True)
    amp_vec = h.Vector(v)
    t_vec = h.Vector(t)
    amp_vec.play(ptr_parent, ptr, t_vec)

    # Run experiment
    test_spontaneous(soma, [], [])


if __name__ == '__main__':
    run_experimental_protocol()