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
import re
import json

import neuron
import bluepyopt.ephys as ephys

from common import units, fileutils
h = neuron.h

# Load NEURON libraries, mechanisms
script_dir = os.path.dirname(__file__)
NRN_MECH_PATH = os.path.join(script_dir, 'mechanisms')
neuron.load_mechanisms(NRN_MECH_PATH)
h.load_file("stdlib.hoc")
h.load_file("stdrun.hoc")

# Debug messages
from common import logutils
logger = logutils.getBasicLogger('gunay')
logutils.setLogLevel('quiet', ['gunay'])

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
    'HCN2':     ['gmax', 'e'], # HCN channel
    'pas':      ['g', 'e'],
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
    'CaHVA':    ['gmax', 'e'], # high-voltage-activated calcium channel

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


def parse_json_commented(json_file):
    """
    Parse json file with C-style comments in it.

    JSON format does not allow comments in it, but you can strip them manually.
    """
    input_str = json_file.read()
    input_str = re.sub(r"//.*$", "", input_str, flags=re.M)
    return json.loads(input_str)


def write_json_after_edit(filenames):
    """
    Rewrite JSON files after editing them, stripping comments and validating
    against the JSON schema.

    @note   This is a better alternative to writing our own parser
            that strips comments, since it will also catch any syntax
            errors
    """
    for filename in filenames:
        full_filename = os.path.join(script_dir, filename)

        with open(full_filename) as json_file:
            invalid_json = json_file.read()
            valid_json = fileutils.validate_minify_json(invalid_json)

        write_json_filename = full_filename[:-5] + '.min.json'
        with open(write_json_filename, 'w') as outfile:
            outfile.write(valid_json)

        print("Wrote parameters to json file {}".format(write_json_filename))



def define_mechanisms(filename, exclude_mechs=None):
    """
    Create list of mechanism descriptions that link NEURON mechanisms to 
    specific regions in the cell, identified by named section lists.

    @param      filename : str
                Filename of json file containing MOD mechanisms for each
                region (section list)

    @param      exclude : list(str)
                List of mechanism names to exclude.
    
    @return     mechanisms: list(ephys.mechanisms.NrnModMechanism)
                List of NEURON mechanism descriptions as Ephys objects.
    """
    if exclude_mechs is None:
        exclude_mechs = []

    full_filename = os.path.join(script_dir, filename)
    with open(full_filename) as json_file:
        mech_definitions = json.load(json_file)

    mechanisms = []
    for seclist_name, mod_names in mech_definitions.items():
        
        seclist_loc = ephys.locations.NrnSeclistLocation(
                                        seclist_name,
                                        seclist_name=seclist_name)
        
        for channel in mod_names:
            if channel in exclude_mechs:
                continue
            mechanisms.append(ephys.mechanisms.NrnMODMechanism(
                            name='{}.{}'.format(channel, seclist_name),
                            mod_path=None,
                            suffix=channel,
                            locations=[seclist_loc],
                            preloaded=True))

    return mechanisms


def define_parameters(
        genesis_params_file,
        params_mapping_file,
        exclude_mechs=None,
    ):
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
    if exclude_mechs is None:
        exclude_mechs = []

    fullfile = os.path.join(script_dir, genesis_params_file)
    with open(fullfile) as json_file:
        genesis_params = json.load(json_file)
    
    fullfile = os.path.join(script_dir, params_mapping_file)
    with open(fullfile) as json_file:
        param_specs = json.load(json_file)

    parameters = []

    # Create dummy section so we can query all units
    h('create dummy')
    dummysec = h.dummy
    for mech_name in mechs_params_dict.keys():
        dummysec.insert(mech_name)

    for param_spec in param_specs:

        # Check if parameter should be excluded
        if 'mech' in param_spec and param_spec['mech'] in exclude_mechs:
            logger.debug('Skipping parameter {} because its mechanism '
                        'is in excluded mechanisms list'.format(param_spec))
            continue

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
            frozen = False # TODO: any reason this should True/False?

            # Interpret spec: can be expression or value
            spec = param_spec['value']
            if isinstance(spec, (float, int)):
                value = spec
            elif isinstance(spec, (str, unicode)):
                value = eval(spec.format(**genesis_params))
            else:
                raise ValueError(
                    "Unexpected value {} for parameter '{}'".format(
                        spec, param_name))
            bounds = None

        elif 'bounds' in param_spec:
            frozen = False
            bounds = param_spec['bounds']
            value = None
        
        else:
            raise Exception(
                'Parameter config has to have bounds or value: {}'.format(
                param_spec))

        # Correct units
        if 'units' in param_spec:
            if value is not None:
                quantity = units.Quantity(value, param_spec['units'])
                converted_quantity = units.to_nrn_units(h, param_name, quantity)
                value = converted_quantity.magnitude
            if bounds is not None:
                quantity = units.Quantity(bounds, param_spec['units'])
                converted_quantity = units.to_nrn_units(h, param_name, quantity)
                bounds = converted_quantity.magnitude

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

        p = parameters[-1]
        logger.debug("Created parameter description:\n" + "\n\t".join(
            ["{} : {}".format(k,getattr(p,k)) for k in 'name', 'value', 'bounds']))

    # delete dummy section
    h.delete_section(sec=dummysec)
    del dummysec

    return parameters


def define_morphology(filename, replace_axon):
    """
    Define morphology (don't instantiate yet).

    The morphology determines the named SecArray and SectionLists 
    available as cell attributes.

    @note   The morphology is instantiated when cell.instantiate() is called.
    """

    return ephys.morphologies.NrnFileMorphology(
                os.path.join(script_dir, filename),
                do_replace_axon=False)


def define_cell(exclude_mechs=None):
    """
    Create GPe cell model
    """

    cell = ephys.models.CellModel(
                'GPe',
                morph=define_morphology(
                        'morphology/bg0121b_axonless_GENESIS_import.swc',
                        replace_axon=False),
                mechs=define_mechanisms(
                        'config/mechanisms.min.json',
                        exclude_mechs=exclude_mechs),
                params=define_parameters(
                        'config/params_hendrickson2011_GENESIS.min.json',
                        'config/map_params_hendrickson2011.min.json',
                        exclude_mechs=exclude_mechs))

    # DONE: write mechanisms.json
    #   - [x] find out how Ephys makes SectionLists
    #       => based on identified SWC types

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

    # For model at https:#senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=114639
    # Gunay, Edgerton, and Jaeger (2008). Channel Density Distributions Explain Spiking Variability in the Globus Pallidus: A Combined Physiology and Computer Simulation Database Approach.
    # - main script in /runs/runsample/setup.g loads following scripts:
    #   + /runs/runsample/readGPparams.g
    #       + /common/GP<i>_default.g       -> sets param variables
    #       + /common/actpars.g             -> sets param variables
    #   + /common/make_GP_libary.g          -> uses param variables
    #       + /common.GPchans.g             -> defines mechanisms using parameters
    #       + /common.GPcomps.g             -> defines compartments using parameters

    # For other models, i.e.
    # - 
    # - https:#senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=137846
    #   + see main script where following files are loaded:
    #   + ./paspars.g for passive parameters
    #   + ../actpars.g for active parameters

    return cell


def define_pynn_model(exclude_mechs=None):
    """
    Create GPe cell model
    """
    import extensions.pynn.ephys_models as ephys_pynn

    model = ephys_pynn.EphysModelWrapper(
                'GPe',
                morph=define_morphology(
                        'morphology/bg0121b_axonless_GENESIS_import.swc',
                        replace_axon=False),
                mechs=define_mechanisms(
                        'config/mechanisms.min.json',
                        exclude_mechs=exclude_mechs),
                params=define_parameters(
                        'config/params_hendrickson2011_GENESIS.min.json',
                        'config/map_params_hendrickson2011.min.json',
                        exclude_mechs=exclude_mechs))
    return model


def create_cell():
    """
    Instantiate GPe cell in NEURON simulator.
    """
    cell = define_cell()
    nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
    cell.instantiate(sim=nrnsim)
    return cell, nrnsim


def rewrite_config_files():
    """
    Clean up commented JSON files
    """
    commented_json = [
        'config/mechanisms.json',
        'config/params_hendrickson2011_GENESIS.json',
        'config/map_params_hendrickson2011.json'
    ]
    write_json_after_edit(commented_json)


if __name__ == '__main__':
    # Make GPe cell
    cell, nrnsim = create_cell()
    icell = cell.icell
    from neuron import gui

    # Write GENESIS parameters to json
    # outfile = os.path.join(script_dir, 'config/GENESIS_parameters.json')
    # get_GENESIS_parameters(write_json_filename=outfile)
