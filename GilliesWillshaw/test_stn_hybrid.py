"""
Hybrid of the Gillies & Willshaw (2006) STN neuron model with Na
channel mechanism from Raman & Bean (2001) accomodating resurgent
component of Na current.

@author Lucas Koelman
@date   14-11-2016
@note   must be run from script directory or .hoc files not found

"""

import numpy as np
import matplotlib.pyplot as plt

import neuron
from neuron import h
nrn = neuron
hoc = h

# Own modules
from common import analysis
from test_stn_Gillies import *

def sigm(x, theta, sigma):
    """ Sigmoid/logistic/smoothy heavyside function
    @param x    x-value or x-axis vector
    @param theta    midpoint of sigmoid
    @param sigma    slope of sigmoid
    """
    return 1./(1.+np.exp(-(x-theta)/sigma))

def sigmshift(x, y0, y1, theta, sigma):
    """ Vertically shifted sigmoid/logistic function
    @param x    x-value or x-axis vector
    @param y0   y for x << theta
    @param y1   y for x >> theta
    @param theta    midpoint of sigmoid
    @param sigma    slope of sigmoid
    """
    return y0 + y1/(1.0+np.exp(-(x-theta)/sigma))

def bellshift(x, y0, y1, theta_a, theta_b, sigma_a, sigma_b):
    """" Vertically shifted bell curve
    @param y0       y outside of bell region
    @param y1       y inside bell region
    @param theta_a  midpoint of first sigmoid
    @param sigma_a  slope of first sigmoid
    @param theta_b  midpoint of second sigmoid
    @param sigma_b  slope of second sigmoid (opposite sign of sigma_a)
    """
    return y0 + y1/(np.exp(-(x-theta_a)/sigma_a) + np.exp(-(x-theta_b)/sigma_b))

def rates_to_steadystates(v, alphafun, betafun):
    """ Steady state activation and time constant from rate functions """
    alpha = alphafun(v)
    beta = betafun(v)
    tau_inf = 1.0 / (alpha + beta)
    act_inf = alpha / (alpha + beta)
    return act_inf, tau_inf

def vtrap(x,y):
    """ Traps for 0 in denominator of rate eqns. """
    res = x/(np.exp(x/y) - 1)
    divzero = np.abs(x/y) < 1e-6
    res[divzero] = y*(1 - x[divzero]/y/2)
    return res

def compare_na_channels():
    """ Compare properties of Na channels in Gillies & Willshaw
        VS Raman & Bean model """

    celsius = 25
    v = np.arange(-100, 100, 0.1)

    ### Gillies & Willshaw ###
    # v-dependence of rate coefficients
    tempb = 23.0
    rest = -60.
    Q10 = 1.980105147
    rate_k = Q10**((celsius-tempb)/10.)
    vadj = v - rest
    # (de)activation variable
    alpham = rate_k * 0.2 * vtrap((13.1-vadj),4.0)
    betam =  rate_k * 0.175 * vtrap((vadj-40.1),1.0)
    # (de)inactivation variable
    alphah = rate_k * 0.08 * np.exp((17.0-vadj)/18.0)
    betah = rate_k * 2.5 / (np.exp((40.0-vadj)/5.0) + 1)

    # plot Gillies & Willshaw values
    plt.figure()
    plt.suptitle('Gillies & Willshaw transient Na rate coefficients')

    plt.subplot(2,1,1)
    plt.plot(v, alpham, label='alpha_m')
    plt.plot(v, betam, label='beta_m')
    plt.xlabel('V (mV)')
    plt.ylabel('(de)activation')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(v, alphah, label='alpha_h')
    plt.plot(v, betah, label='beta_h')
    plt.xlabel('V (mV)')
    plt.ylabel('de(inactivation)')
    plt.legend()

    #### Raman & Bean ###
    # v-dependence of rate coefficients
    q10 = 3.
    qt = q10**((celsius-22.)/10.)
    qt = 1.0
    Con = 0.005 #       (/ms)                   : closed -> inactivated transitions
    Coff = 0.5  #       (/ms)                   : inactivated -> closed transitions
    Oon = 0.75  #       (/ms)                   : open -> Ineg transition
    Ooff = 0.005 #      (/ms)                   : Ineg -> open transition
    alpha = 150. #       (/ms)                   : activation
    beta = 3.    #       (/ms)                   : deactivation
    gamma = 150. #       (/ms)                   : opening
    delta = 40.  #       (/ms)                   : closing, greater than BEAN/KUO = 0.2
    epsilon = 1.75 #    (/ms)                   : open -> Iplus for tau = 0.3 ms at +30 with x5
    zeta = 0.03 #       (/ms)                   : Iplus -> open for tau = 25 ms at -30 with x6
    x1 = 20. #           (mV)                                : Vdep of activation (alpha)
    x2 = -20. #          (mV)                                : Vdep of deactivation (beta)
    x3 = 1e12 #         (mV)                                : Vdep of opening (gamma)
    x4 = -1e12 #        (mV)                                : Vdep of closing (delta)
    x5 = 1e12 #         (mV)                                : Vdep into Ipos (epsilon)
    x6 = -25.
    # Closed_5 <-> Open
    f0O = gamma * np.exp(v/x3) * qt # gamma (closed->open is part of activation or alpha_m)
    b0O = delta * np.exp(v/x4) * qt # delta (open->closed corresponds to deactivaton or beta_m)
    # Open <-> Inactivated_6
    fin_arr = np.zeros_like(v) + Oon * qt # (open->inactivated or part of alpha_h)
    bin_arr = np.zeros_like(v) + Ooff * qt # (inactivated->open or part of beta_h)
    # Open <-> blocked
    fip = epsilon * np.exp(v/x5) * qt # (open->blocked or alpha_B)
    bip = zeta * np.exp(v/x6) * qt # (blocked->open or beta_B)

    # plot Raman & Bean values
    plt.figure()
    plt.suptitle('Raman & Bean Na channel rate coefficients')

    plt.subplot(3,1,1)
    plt.plot(v, f0O, label='gamma')
    plt.plot(v, b0O, label='delta')
    plt.xlabel('V (mV)')
    plt.ylabel('closing')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(v, fin_arr, label='Oon')
    plt.plot(v, bin_arr, label='Ooff')
    plt.xlabel('V (mV)')
    plt.ylabel('inactivation')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(v, fip, label='epsilon')
    plt.plot(v, bip, label='zeta')
    plt.xlabel('V (mV)')
    plt.ylabel('inactivation')
    plt.legend()

    plt.show(block=False)

def test_burstresurgent(soma, dends_locs, stims):
    """ Run rebound burst experiment from original Hoc file

    EXPECTED BEHAVIOUR
    - INa_rsg slowly inactivated during long firing period
    - INa_rsg deinactivated during hyperpolarization

    TODO: 
    - test replacement of Na mechanism with Narsg modified like in Akemann to match the
      timing of the two components of Na current
    - test shorter, more realistic hyperpolarization period (corresponding to volley of IPSPs)
      to see if there the difference is greated between situation with and without Narsg
    """
    # Get electrodes and sections to record from
    dendsec = dends_locs[0][0]
    dendloc = dends_locs[0][1]
    stim1, stim2, stim3 = stims[0], stims[1], stims[2]

    # Set simulation parameters
    dur = 3500
    h.dt = 0.025
    h.celsius = 35 # different temp from paper
    h.v_init = -60 # paper simulations sue default v_init
    set_aCSF(4) # Set initial ion concentrations from Bevan & Wilson (1999)

    # Set up stimulation
    stim1.delay = 0
    stim1.dur = 2000
    stim1.amp = 0.025 # evoke fast spiking -> slow inactivation

    stim2.delay = 2000
    stim2.dur = 500
    stim2.amp = -0.25 # hyperpolarizing pulse -> recovery from inactivation (deinactivation)

    stim3.delay = 2500
    stim3.dur = 1000
    stim3.amp = 0.0

    # Record
    secs = {'soma': soma, 'dend': dendsec}
    traceSpecs = collections.OrderedDict() # for ordered plotting (Order from large to small)
    traceSpecs['V_soma'] = {'sec':'soma','loc':0.5,'var':'v'}

    # Na currents
    traceSpecs['I_Na'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'ina'}
    traceSpecs['I_NaL'] = {'sec':'soma','loc':0.5,'mech':'NaL','var':'inaL'}
    # Na resurgent current related
    if resurgent:
        traceSpecs['I_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'ina'}
        # traceSpecs['sI_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'Itot'} # inactivated fraction
        # traceSpecs['sC_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'Ctot'} # closed fraction
        # traceSpecs['sB_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'B'} # blocked fraction
        # traceSpecs['sO_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Narsg','var':'O'} # open fraction
    natrans=True
    if natrans:
        traceSpecs['sO_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'O'} # open state
        traceSpecs['sI_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'I6'} # inactivated
        traceSpecs['sC_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'Ctot'} # closed state
        traceSpecs['sCI_Narsg'] = {'sec':'soma','loc':0.5,'mech':'Na','var':'Itot'} # closed+inactivated

    # K currents
    traceSpecs['I_KDR'] = {'sec':'soma','loc':0.5,'mech':'KDR','var':'ik'}
    traceSpecs['I_Kv3'] = {'sec':'soma','loc':0.5,'mech':'Kv31','var':'ik'}
    traceSpecs['I_KCa'] = {'sec':'soma','loc':0.5,'mech':'sKCa','var':'isKCa'}
    traceSpecs['I_h'] = {'sec':'soma','loc':0.5,'mech':'Ih','var':'ih'}
    # Ca currents (soma)
    traceSpecs['I_CaL'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iLCa'}
    traceSpecs['I_CaN'] = {'sec':'soma','loc':0.5,'mech':'HVA','var':'iNCa'}
    traceSpecs['I_CaT'] = {'sec':'soma','loc':0.5,'mech':'CaT','var':'iCaT'}
    recordStep = 0.05
    recData = analysis.recordTraces(secs, traceSpecs, recordStep)

    # Simulate
    h.tstop = dur
    h.init() # calls finitialize() and fcurrent()
    h.run()

    # Analyze
    burst_time = [980, 1120]

    # Soma voltage
    recV = {'V_soma':recData['V_soma']}
    analysis.plotTraces(recV, recordStep)

    # Soma currents (relative)
    recI = collections.OrderedDict()
    for k, v in recData.iteritems():
        if k.startswith('I'): recI[k] = recData[k]
    analysis.cumulPlotTraces(recI, recordStep, cumulate=False)

    # Na channel states
    recStates = collections.OrderedDict()
    for k, v in recData.iteritems():
        if k.startswith('s'): recStates[k] = recData[k]
    analysis.cumulPlotTraces(recStates, recordStep, cumulate=False)

    # Overlay voltage signal
    # plt.plot(np.arange(0, dur, recordStep), recData['V_soma'].as_numpy()*1e-3, color='r')
    # plt.show(block=False)


if __name__ == '__main__':
    compare_na_channels()