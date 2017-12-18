"""
Setup code for Balbi et al (2015) motoneuron compartmental cell model for use
with NEURON + Python.

@author Lucas Koelman

@date   15-12-2017
"""

import neuron
h = neuron.h
h.load_file("stdlib.hoc") # Load the standard library
h.load_file("stdrun.hoc") # Load the standard run library

# Load NEURON mechanisms
import os.path
scriptdir, scriptfile = os.path.split(__file__)
NRN_MECH_PATH = os.path.normpath(os.path.join(scriptdir, 'channels'))
neuron.load_mechanisms(NRN_MECH_PATH)

# Channel mechanisms (key = suffix of mod mechanism) : max conductance parameters
balbi_gdict = {
    'gh':       ['ghbar'],      # H channel (Na + K ions)
    'kca2':     ['g'],          # Ca-dependent K channel (K ion)
    'kdrRL':    ['gMax'],       # Delayed Rectifier K Channel (K ion)
    'L_Ca':     ['gcabar'],     # L-type Ca channel (Ca ion, CaL virtual ion)
    'mAHP':     ['gkcamax', 'gcamax'], # Ca-dependent K channel + Ca channel (Ca + K ions)
    'na3rp':    ['gbar'],       # Fast Na current (Na ion)
    'naps':     ['gbar'],       # Persistent Na current (Na ion)
    'pas':      ['g_pas'],      # Passive/leak channel
}
balbi_mechs = list(balbi_gdict.keys()) # all mechanisms
balbi_glist = [gname+'_'+mech for mech,chans in balbi_gdict.iteritems() for gname in chans]
gleak_name = 'gpas_STh'
active_gbar_names = [gname for gname in balbi_glist if gname != gleak_name]


def make_cell_balbi(model_no=1):
    """
    Initialize Balbi et al. cell model

    @param  model_no    model number to load: integer in range 1-14

    @return             dict { region_name<str> : list(Section) }

    @effect             Following variables will be available on Hoc interpreter:

                        * somatic sections:
                            - soma[N] <Section> somatic sections

                        * dendritic sections:
                            - dend[M] <Section> dendritic sections
                        
                        * axon-soma interface
                            - AH[1] <Section> axon hillock
                            - IS[1] <Section> axon initial segment
                        
                        * axonal sections:
                            - node[axonnodes] <Section>
                            - MYSA[paranodes1] <Section>
                            - FLUT[paranodes2] <Section>
                            - STIN[axoninter] <Section>
    """
    secnames_used = ['soma', 'dend', 'AH', 'IS', 'node', 'MYSA', 'FLUT', 'STIN']
    if any((hasattr(h, attr) for attr in secnames_used)):
        raise Exception("Folowing global variables must be unallocated on Hoc interpreter object: {}".format(', '.join(secnames_used)))

    # load model
    h.xopen("createcell_balbi.hoc")
    h.choose_model(model_no)
    
    # Get created sections
    named_sections = { secname: list(getattr(h, secname)) for secname in secnames_used}
    return named_sections


def motocell_steadystate(model_no):
    """
    Load steate-state condition for motoneuron cell model with given cell id

    @param  model_no    model number to load: integer in range 1-14

    @note   h.load_steadystate() uses SaveState.restore() which means 
            Between a save and a restore, it is important not to create or delete sections, NetCon objects, or point processes. Do not change the number of segments, insert or delete mechanisms, or change the location of point processes.
    """
    h.celsius = 37
    h.load_steadystate(model_no)


if __name__ == '__main__':
    named_sec = make_cell_balbi()