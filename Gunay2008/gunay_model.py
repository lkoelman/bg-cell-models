"""
Globus pallidus multi-compartmental model by Gunay C, Edgerton JR, Jaeger D (2008).
Using channel mechanisms ported to NEURON by Kitano (2011)

@author     Lucas Koelman
@date       07-01-2018

@reference  Gunay C, Edgerton JR, Jaeger D (2008) Channel density distributions 
            explain spiking variability in the globus pallidus: a combined physiology 
            and computer simulation database approach. J Neurosci 28:7476-91

            https:#senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=114639


@reference  Fujita T, Fukai T, Kitano K (2012) Influences of membrane properties 
            on phase response curve and synchronization stability in a model 
            globus pallidus neuron. J Comput Neurosci 32(3):539-53
            
            https:#senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=143100

@reference  Based on following example templates:
            https:#github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/l5pc_model.py
"""

import os
import json

import neuron
import bluepyopt.ephys as ephys

script_dir = os.path.dirname(__file__)
NRN_MECH_PATH = os.path.join(script_dir, 'mechanisms')
neuron.load_mechanisms(NRN_MECH_PATH)

# Channel mechanisms (key = suffix of mod mechanism) : max conductance parameters
gbar_dict = {
    # Nonspecific channels
    'HCN':      ['gmax'],
    'leak':     ['gmax'],
    # Na channels
    'NaF':      ['gmax'],
    'NaP':      ['gmax'],
    # K-channels
    'Kv2':      ['gmax'],
    'Kv3':      ['gmax'],
    'Kv4f':     ['gmax'],
    'Kv4s':     ['gmax'],
    'KCNQ':     ['gmax'],
    'SK':       ['gmax'],
    # Calcium channels / buffering
    'CaH':      ['gmax'],
}
gleak_name = 'gmax_leak'

# Mechanism parameters that are changed from default values in original model code
mechs_params_dict = {
    # Nonspecific channels
    'HCN':      ['gmax', 'e'], # HCN channel
    'leak':     ['gmax'],
    # Na channels
    'NaF':      ['gmax'],
    'NaP':      ['gmax'],
    # K-channels
    'Kv2':      ['gmax'],
    'Kv3':      ['gmax'],
    'Kv4f':     ['gmax'],
    'Kv4s':     ['gmax'],
    'KCNQ':     ['gmax'],
    'SK':       ['gmax'],
    # Calcium channels / buffering
    'Calcium':  [''],
    'CaH':      ['gmax', 'e'], # high-voltage-activated calcium channel

}

# All mechanism parameters that are not conductances
mechs_params_nogbar = dict(mechs_params_dict)
for mech, params in mechs_params_nogbar.iteritems():
    for gbar_param in gbar_dict.get(mech, []):
        try:
            params.remove(gbar_param)
        except ValueError:
            pass

# GLOBAL mechanism parameters (assigned using h.param = val)
global_params_list = [
    'ena', 'ek'
]

# List of mechanisms, max conductance params, active conductances
mechs_list = list(mechs_params_dict.keys()) # all mechanisms
gbar_list = [gname+'_'+mech for mech,chans in gbar_dict.iteritems() for gname in chans]
active_gbar_names = [gname for gname in gbar_list if gname != gleak_name]


def define_mechanisms(filename):
    """
    Create list of mechanism descriptions that link NEURON mechanisms to 
    specific regions in the cell, identified by named section lists.

    @param      filename : str
                Filename of json file containing MOD mechanisms for each
                region (section list)
    
    @return     mechanisms: list(ephys.mechanisms.NrnModMechanism)
                List of NEURON mechanism descriptions as Ephys objects.
    """
    full_filename = os.path.join(script_dir, filename)
    
    with open(full_filename) as json_file:
        mech_definitions = json.load(json_file)

    mechanisms = []
    for seclist_name, mod_names in mech_definitions.items():
        
        seclist_loc = ephys.locations.NrnSeclistLocation(
                                        seclist_name,
                                        seclist_name=seclist_name)
        
        for channel in mod_names:
            mechanisms.append(ephys.mechanisms.NrnMODMechanism(
                            name='{}.{}'.format(channel, seclist_name),
                            mod_path=None,
                            suffix=channel,
                            locations=[seclist_loc],
                            preloaded=True))

    return mechanisms


def define_parameters(genesis_params_file, params_mapping_file):
    """
    Create list of parameter descriptions that link (distributions of)
    mechanism parameters to specific regions in the cell, 
    identified by named section lists.

    @param      filename : str
                Filename of json file containing MOD mechanisms for each
                region (section list)
    
    @return     parameters: list(ephys.parameters.NrnParameter)
                List of NEURON parameter descriptions as Ephys objects.
    """

    fullfile = os.path.join(script_dir, genesis_params_file)
    with open(fullfile) as json_file:
        genesis_params = json.load(json_file)
    
    fullfile = os.path.join(script_dir, params_mapping_file)
    with open(fullfile) as json_file:
        param_specs = json.load(json_file)

    parameters = []

    for param_spec in param_specs:

        # Get param name in NEURON
        if 'param_name' in param_spec:
            param_name = param_spec['param_name']
        elif 'mech' in param_spec and 'mech_param' in param_spec:
            param_name = '{}_{}'.format(param_spec['mech_param'], param_spec['mech'])
        else:
            raise ValueError(
                'Not enough information to resolve NEURON parameter name: {}'.format(
                    param_spec))
        
        # 'value' is an expression that maps from original GENESIS parameter name
        # to NEURON parameter value
        if 'value' in param_spec:
            frozen = True

            # Interpret spec: can be expression or value
            spec = param_spec['value']
            if isinstance(spec, (float, int)):
                value = spec
            elif isinstance(spec, str):
                value = eval(spec.format(**genesis_params))
            else:
                raise ValueError(
                    "Unexpected value {} for parameter '{}'".format(
                        value, param_name))
            bounds = None

        elif 'bounds' in param_spec:
            frozen = False
            bounds = param_spec['bounds']
            value = None
        
        else:
            raise Exception(
                'Parameter config has to have bounds or value: {}'.format(
                param_spec))

        # Make Ephys description of parameter
        if param_spec['type'] == 'global':
            parameters.append(
                ephys.parameters.NrnGlobalParameter(
                    name=param_name,
                    param_name=param_name,
                    frozen=frozen,
                    bounds=bounds,
                    value=value))
        
        elif param_spec['type'] in ['section', 'range']:
            
            # Spatial distribution of parameter
            if param_spec['dist_type'] == 'uniform':
                scaler = ephys.parameterscalers.NrnSegmentLinearScaler()
            
            elif param_spec['dist_type'] == 'exp':
                scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                    distribution=param_spec['dist'])
            
            seclist_loc = ephys.locations.NrnSeclistLocation(
                            param_spec['sectionlist'],
                            seclist_name=param_spec['sectionlist'])

            name = '{}.{}'.format(param_name,
                                   param_spec['sectionlist'])

            # Section parameter is uniform in a Section
            if param_spec['type'] == 'section':
                parameters.append(
                    ephys.parameters.NrnSectionParameter(
                        name=name,
                        param_name=param_name,
                        value_scaler=scaler,
                        value=value,
                        frozen=frozen,
                        bounds=bounds,
                        locations=[seclist_loc]))
            
            # RANGE parameters vary over x-loc of Section
            elif param_spec['type'] == 'range':
                parameters.append(
                    ephys.parameters.NrnRangeParameter(
                        name=name,
                        param_name=param_name,
                        value_scaler=scaler,
                        value=value,
                        frozen=frozen,
                        bounds=bounds,
                        locations=[seclist_loc]))
        else:
            raise Exception(
                'Param config type has to be global, section or range: {}'.format(
                param_spec))

    return parameters


def define_morphology(filename, replace_axon):
    """
    Define morphology (don't instantiate yet).

    @note   The morphology is instantiated when cell.instantiate() is called.
    """

    return ephys.morphologies.NrnFileMorphology(
                os.path.join(script_dir, filename),
                do_replace_axon=False)

def get_GENESIS_parameters(write_json_filename=None):
    """
    Set default GPe neuron parameter in NEURON.
    """

    # Parameters from https:#senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=127728
    # GENESIS uses SI units
    #   - Cm : F/m^2
    #   - Rm : Ohm*m^2
    #   - Ra : Ohm*m
    #   - gbar : S/m^2
    #   - E : V
    # NEURON uses units:
    #   - Cm : uF/cm^2  == 1e-6/1e-4 * F/m^2 == 1e-2 * F/m^2    => x 1e2
    #   - Rm : see gbar                                         => x 1e4
    #   - Ra : Ohm*cm   == 1e-2 * Ohm*m                         => x 1e2
    #   - gbar : S/cm^2 == 1/1e-4 * S/m^2 == 1e4 * S/m^2        => x 1e-4
    #   - E : mV == V*1e-3                                      => x 1e3
    
    # NOTE: parameters are copied from GENESIS script and put in dictionary format using regex replace: r"float\s(\w+)\s+\=\s+(-?[\d.]+)" -> '$1' : $2,
    
    ## parameters in script /articleCode/commonGPFull/GP1_defaults.g
    script_gp_defaults = {
        'RA' : 1.74,     # uniform
        'CM' : 0.024,    # all unmyelinated regions
        'CM_my' : 0.00024,   # myelinated axon segments.
        'RM_sd' : 1.47,  # soma
        'RM_ax' : 1.47,   # unmyelinated axonal regions
        'RM_my' : 10,    # myelinated axon segments.
        'ELEAK_sd' : -0.060,    # soma & dend
        'ELEAK_ax' : -0.060,    # axon
        'EREST_ACT' : -0.060,
    }

    ## parameters in script /articleCode/commonGPFull/simdefaults.g
    script_sim_defaults = {
        #Sodium channel kinetics & voltage dependence
        'Vhalfm_NaF' : -0.0324,
        'Km_NaF' : 0.005,
        'taummax_NaF' : 0.000028,
        'taummin_NaF' : 0.000028,

        'V0h_NaF' : -0.048,
        'Kh_NaF' : -0.0028,
        'tauhV0_NaF' : -0.043,
        'tauhmax_NaF' : 0.004,
        'tauhmin_NaF' : 0.00025,   # 0.0002

        'V0s_NaF' : -0.040,
        'Ks_NaF' : -0.0054,
        'mins_NaF' : 0.15,
        'Ktaus1_NaF' : 0.0183,
        'Ktaus2_NaF' : 0.010,
        'tausmax_NaF' : 1,
        'tausmin_NaF' : 0.01,

        'Vhalfm_NaP' : -0.050,
        'V0h_NaP' : -0.057,
        'Kh_NaP' : -0.004,
        'hmin_NaP' : 0.154,
        'V0s_NaP' : -0.01,
        'Abeta_NaP' : 6.94,
        'Bbeta_NaP' : 0.447,

        #Kv2 properties
        'npower_Kv2' : 4,
        'Vhalfn_Kv2' : -0.018,
        'Kn_Kv2' : 0.0091,
        'taunmax_Kv2' : 0.03,
        'taunmin_Kv2' : 0.0001,
        'hmin_Kv2' : 0.2,

        #Kv3 properties
        'npower_Kv3' : 4,
        'Vhalfn_Kv3' : -0.013,    # Actual Vhalf
        'Kn_Kv3' : 0.0078,    # Yields K = 6 mV with Xpower = 4
        'hmin_Kv3' : 0.6,

        #Kv4 properties
        'V0n_Kv4' : -0.049,    # Yields Vhalf = -27 mV when Xpower = 4
        'Kn_Kv4' : 0.0125,    # Yields K = 9.6 mV when Xpower = 4
        'Ktaun1_Kv4' : 0.029,
        'Ktaun2_Kv4' : 0.029,

        'V0h_Kv4' : -0.083,    # changed from -0.072 02/17/2005 to match 
                                            # Tkatch et al
        'Kh_Kv4' : -0.01, # changed from -0.0047 02/17/2005 to match 
                                            # Tkatch et al
        'Ktauh1_Kv4' : 0.010,
        'Ktauh2_Kv4' : 0.010,

        #KCNQ properties
        'Vhalfn_KCNQ' : -0.0285,
        'Kn_KCNQ' : 0.0195,    # Yields K = 15 mV for 1st order Boltzmann
                                        #  when Xpower = 4.

        #SK channel properties
        'EC50_SK' : 0.00035, # SI unit = mM; default = 350 nM.
        'dQ10_SK' : 2,

        #CaHVA properties
        'npower_CaHVA' : 1,
        'Vhalfn_CaHVA' : -0.02,
        'Kn_CaHVA' : 0.007, 

        #Voltage-gated ion channel reversal potentials
        'ENa' : 0.050,
        'ECa' : 0.130,
        'EK' : -0.090,
        'Eh' : -0.03,

        #Calcium concentration parameters
        'B_Ca_GP_conc' : 4.0/3.0*5.2e-12,#3.6e-7 #changed on 10/15/2009 to be consistent with GPcomps.g
        'shell_thick' : 20e-9,  #  meters 
        'tau_CaClearance' : 0.001,   #  time constant for Ca2+ clearance (sec)
    }

    ## parameters in script /articleCode/commonGPFull/actpars.g
    script_actpars = {
        'dendNaF' : 40,
        #Voltage-gated ion channel densities
        'G_NaF_soma' : 2500,
        'G_NaP_soma' : 1,
        'G_Kv2_soma' : 320,
        'G_Kv3_soma' : 640,
        'G_Kv4f_soma' : 160,
        'G_Kv4s_soma' : 160*1.5,
        'G_KCNQ_soma' : 0.4,
        'G_SK_soma' : 50,
        'G_Ca_HVA_soma' : 2, 
        'G_h_HCN_soma' : 0.2,
        'G_h_HCN2_soma' : 0.2*2.5,

        'G_NaF_axon' : 5000,
        'G_NaP_axon' : 40,
        'G_Kv2_axon' : 640,
        'G_Kv3_axon' : 1280,  
        'G_Kv4f_axon' : 1600,
        'G_Kv4s_axon' : 1600*1.5  ,
        'G_KCNQ_axon' : 0.4,
        'G_NaF_dend' : 40,
        'G_NaP_dend' : 1,

        'G_Kv2_dend' : 64,
        'G_Kv3_dend' : 128,   
        'G_Kv4f_dend' : 160,
        'G_Kv4s_dend' : 160*1.5,
        'G_KCNQ_dend' : 0.4,
        'G_SK_dend' : 4,
        'G_Ca_HVA_dend' : 0.15,  
        'G_h_HCN_dend' : 0.2,
        'G_h_HCN2_dend' : 0.2*2.5,
    }

    all_params = {}
    for param_dict in script_sim_defaults, script_gp_defaults, script_actpars:
        all_params.update(param_dict)

    if write_json_filename is not None:
        with open(write_json_filename, 'w') as outfile:
            json.dump(all_params, outfile, indent=4, sort_keys=True)
        print("Wrote parameters to json file {}".format(write_json_filename))

    return all_params


def create_cell():
    """
    Create GPe cell model

    NOTES
    -----

    cell.instantiate():
        
        -> cell.create_empty_cell(cell.name, cell.seclist_names, cell.secarray_names)
            
            -> cell.create_empty_template(...)
                ; Creates new template in hoc using "begintemplate" that contains
                ; a CellRef and a SectionArray + SectionList for each seclist/secarray
        
        -> cell.icell = hoc_template_function()
        
        -> cell.morphology.instantiate(cell.icell) :
            ; Fills secarray/seclist of instantiated template with morphology Sections
           
            -> imorphology = h.Import3d_<fileformat>(file)
            -> importer = h.Import3d_GUI(imorphology, 0)
                ; see file /hoc/import3d/import3d_gui.hoc
            -> importer.instantiate(cell.icell)
                ; instantiates Sections defined by morphology and assigns them
                ; to template's secarrays based on SWC types & seclist variables
    """

    cell = ephys.models.CellModel(
                'GPe',
                morph=define_morphology(
                        'morphology/bg0121b_axonless_GENESIS_import.asc',
                        replace_axon=False),
                mechs=define_mechanisms('config/mechanisms.json'),
                params=define_parameters(
                        'config/parameters_original_GENESIS.json',
                        'config/parameters_mapping_GENESIS.json'))

    # DONE: write mechanisms.json
    #   - [x] see script /articleCode/commonGPFull/GPcomps.g to see which named compartments
    #     defined in .p file get which mechanisms (prototypes copied into compartment)
    #       => protoype is inserted into ALL dendritic sections
    #   - [x] find out how Ephys makes SectionLists
    #       => based on identified SWC types

    # TODO: write parameters.json

    # For model at https:#senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=127728
    # "Comparison of full and reduced globus pallidus models (Hendrickson 2010)"
    # see:
    #   + main script in /articleCode/scripts/genesisScripts/GP1axonless_full_synaptic.g
    #     First it loads variables from following scripts:
    #       + /articleCode/commonGPFull/GP1_defaults.g
    #       + /articleCode/commonGPFull/simdefaults.g
    #       + /articleCode/commonGPFull/actpars.g
    #   + The variables are then used in following scripts, in braces: {varname}
    #     (GP1axonless_full*.g -> make_GP_libary.g -> ...)
    #       + /articleCode/commonGPFull/GP1_axonless.p
    #       + /articleCode/commonGPFull/GPchans.g
    #       + /articleCode/commonGPFull/GPcomps.g

    # For model at https:#senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=136315
    # "Globus pallidus neuron models with differing dendritic Na channel expression 
    # (Edgerton et al., 2010)", see: 
    #   + main script in /run_example/run_vivo_example.g, loads scripts:
    #   + ../common/GP1_constants.g
    #   + ../common/biophysics/GP1_passive.g
    #   + ../common/biophysics/GP1_active.g
    #   + .. see lines with 'getarg' statement for modification of scaling factors/gradients

    # For other models, i.e.
    # - https:#senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=114639
    # - https:#senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=137846
    #   + see main script where following files are loaded:
    #   + ./paspars.g for passive parameters
    #   + ../actpars.g for active parameters

    return cell

if __name__ == '__main__':
    # Make GPe cell
    # gpe_cell = create_cell()
    # gpe_cell.instantiate(sim=neuron.h)

    # Write GENESIS parameters to json
    # outfile = os.path.join(script_dir, 'config/GENESIS_parameters.json')
    # get_GENESIS_parameters(write_json_filename=outfile)

    # Test
    cell = ephys.models.CellModel(
                'GPe',
                morph=define_morphology(
                        'morphology/bg0121b_axonless_GENESIS_import.swc',
                        replace_axon=False),
                mechs=[],
                params=[])

    nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
    cell.instantiate(sim=nrnsim)