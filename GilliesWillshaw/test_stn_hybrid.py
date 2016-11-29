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


if __name__ == '__main__':
    compare_na_channels()