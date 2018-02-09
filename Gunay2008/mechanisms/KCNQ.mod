TITLE KCNQ potassium channel for GPe neuron

COMMENT
DESCRIPTION
-----------

KCNQ

Activation kinetics: Gamper, Stockand, Shapiro (2003). J Neurosci 23: 84-95.
GV curve, deact kinetics: Prole & Marrion (2004). Biophys J. 86: 1454-69.

CREDITS
-------

modeled in GENESIS by J.R. Edgerton, 2004
implemented in NEURON by Kitano, 2011
modified in NEURON by Lucas Koelman, 2018 to reflect model Hendrickson et al., 2010
ENDCOMMENT

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
}

NEURON {
    SUFFIX KCNQ
    USEION k READ ek WRITE ik
    RANGE gmax, iKCNQ
}

PARAMETER {
    v (mV)
    dt (ms)
    gmax  = 0.001 (mho/cm2)
    iKCNQ  = 0.0 (mA/cm2)
    ek (mV)

    : m-gate
    theta_m0 = -28.5 (mV)
    k_m = 19.5 (mV)
    tau_m0 = 6.7 (ms)
    tau_m1 = 100.0 (ms)
    sigma_m0 = -25.0 (mV)
    sigma_m1 = -35.0 (mV)
}

STATE {
    m
}

ASSIGNED { 
    ik (mA/cm2)

    minf
    taum (ms)
    theta_m (mV)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik  = gmax*m*m*m*m*(v-ek)
    iKCNQ = ik
}

UNITSOFF

INITIAL {
    theta_m = theta_m0 + (k_m * (log((1 / pow(0.5, 1/4)) - 1)))
    settables(v)
    m = minf
}

DERIVATIVE states {  
    settables(v)
    m' = (minf - m)/taum
}

PROCEDURE settables(v) {
    TABLE minf, taum FROM -100 TO 100 WITH 400

    : m-gate
    minf = 1.0 / (1.0 + exp((theta_m - v)/k_m))
    taum = tau_m0 + (tau_m1 - tau_m0)/(exp((theta_m - v)/sigma_m0) + exp(-(theta_m - v)/sigma_m1))
}

UNITSON
