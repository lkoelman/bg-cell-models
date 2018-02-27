TITLE high-voltage-activated calcium channel for GPe neuron

COMMENT
DESCRIPTION
-----------

HVA Ca2+ Channels 

Voltage-dependent activation from GP data:  
Surmeier Seno and Kitai 1994, J Neurophysio. 71(3):1272-1280

USAGE
-----

First insert Ca buffering mechanism into compartments where you wish to insert
this mechanism.

CREDITS
-------

modeled in GENESIS by J.R. Edgerton, 2004
implemented in NEURON by Kitano, 2011
modified in NEURON by Lucas Koelman, 2018 to reflect model Hendrickson et al., 2010
ENDCOMMENT

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (mM) = (milli/liter)
}

NEURON {
    SUFFIX CaHVA
    USEION ca READ cai,cao,eca WRITE ica
    RANGE gmax, iCaH
}

PARAMETER {
    v (mV)
    dt (ms)
    cai (mM)
    cao (mM)
    eca (mV)
    gmax  = 0.001 (mho/cm2)
    iCaH  = 0.0 (mA/cm2)

    : m-gate
    theta_m0 = -20.0 (mV)
    k_m = 7.0 (mV)
    tau_m0 = 0.2 (ms) : estimated from traces Q10=2.5 adjusted
    tau_m1 = 0.2 (ms)
}

STATE {
    m
}

ASSIGNED { 
    ica (mA/cm2)

    taum (ms)
    minf
    theta_m (mV)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ica  = gmax*m*(v-eca)
    iCaH = ica
}

UNITSOFF

INITIAL {
    theta_m = theta_m0 + (k_m * (log((1/0.5) - 1)))
    settables(v)
    m = minf
}

DERIVATIVE states {  
    settables(v)
    m' = (minf - m)/taum
}

PROCEDURE settables(v) {
    TABLE minf FROM -100 TO 100 WITH 400

    : m-gate
    minf = 1.0 / (1.0 + exp((theta_m - v)/k_m))
    taum = tau_m0 + ((tau_m1 - tau_m0) / (exp( (theta_m - v) / k_m ) + exp(-(theta_m - v) / k_m )))
}

UNITSON
