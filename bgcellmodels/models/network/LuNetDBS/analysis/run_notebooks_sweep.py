#!/usr/bin/env python

"""
Automatically run a notebook with each simulation output.

Usage
-----

- set all script parameters marked 'SETPARAM'

- Jupyter notebook: edit so variables are read from configuration files and save with empty cells

- Run script: can run multiple versions simultaneously, since script is compiled to bytecode
  upon execution
"""

import os
import subprocess
import re
import uuid
from datetime import datetime
from multiprocessing import Pool


# SETPARAM: List simulation output directories to analyze
output_dirs = """
/home/luye/Documents/simdata/q8_sweep-dbs-phase/LuNetDBS_2019.08.07_18.54.22_job-13298_axons-full-V2_dbs-phase-135
/home/luye/Documents/simdata/q8_sweep-dbs-phase/LuNetDBS_2019.08.07_18.54.26_job-13296_axons-full-V2_dbs-phase-045
/home/luye/Documents/simdata/q8_sweep-dbs-phase/LuNetDBS_2019.08.07_18.54.26_job-13299_axons-full-V2_dbs-phase-180
/home/luye/Documents/simdata/q8_sweep-dbs-phase/LuNetDBS_2019.08.07_18.54.26_job-13301_axons-full-V2_dbs-phase-270
/home/luye/Documents/simdata/q8_sweep-dbs-phase/LuNetDBS_2019.08.07_18.54.28_job-13297_axons-full-V2_dbs-phase-090
/home/luye/Documents/simdata/q8_sweep-dbs-phase/LuNetDBS_2019.08.07_18.54.37_job-13302_axons-full-V2_dbs-phase-315
/home/luye/Documents/simdata/q8_sweep-dbs-phase/LuNetDBS_2019.08.07_18.54.41_job-13300_axons-full-V2_dbs-phase-225
/home/luye/Documents/simdata/q8_sweep-dbs-phase/LuNetDBS_2019.08.07_18.54.55_job-13295_axons-full-V2_dbs-phase-000
""".strip().split()

nb_dir = "/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/analysis"
nb_infile = "lunet_dbs_analysis.ipynb"
nb_path = os.path.join(nb_dir, nb_infile)

# SETPARAM: name of sweep variable
sweep_name = "dbs-phase"

def process_sim_outputs(args):

    sim_outdir, sweep_index = args

    # SETPARAM: pattern for extraction of sweep variable from filename
    sweep_val_pattern = r"phase-([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
    match = re.search(sweep_val_pattern, sim_outdir)
    sweep_val = match.groups()[0]

    # Settings variables passed to executed notebook
    nb_pyvars = {}
    # SETPARAM: interval for signal analysis
    ROI_INTERVAL = (1e3, 3e3)
    # SETPARAM: filename containing recorded signals
    nb_pyvars['matfile_common_pattern'] = '-3000ms'
    ival_sec = [t/1e3 for t in ROI_INTERVAL]
    # SETPARAM: suffix for pickle file and jupyter notebook files
    out_suffix = '{:.1f}s-{:.1f}s'.format(*ival_sec)
    # SETPARAM: refence phase signal and frequency band
    nb_pyvars['reference_phase'] = {'method': 'from_ctx', 'passband': (17.0, 23.0)}

    # prepare executable Python script using property eval(repr(object)) == object.
    nb_pyvars.update({
        'outputs': sim_outdir,
        'ROI_INTERVAL': ROI_INTERVAL,
        'sweep_var_name': sweep_name,
        'sweep_var_value': sweep_val,
        'sweep_index': sweep_index,
        'automatic_execution': True,
        'pickle_filename': 'analysis_results_' + out_suffix,
    })
    print("Config for notebook is:\n{}".format(nb_pyvars))
    nb_pyscript= "\n".join(("{} = {}".format(k, repr(v)) for k,v in nb_pyvars.items()))

    conf_path = os.path.join(nb_dir, "nb_exec_conf_{}.py".format(sweep_index))
    with open(conf_path, 'w') as conf_file:
        conf_file.write(nb_pyscript)

    # Add some environment variables to be read from notebook
    env = dict(os.environ)
    env["NB_CONF_FILE"] = conf_path

    # Specify outputs
    job_id = re.search(r'job-(\w+)', sim_outdir).groups()[0]
    timestamp = datetime.now().strftime('%m.%d-%H:%M') # %Y.%m.%d-%H.%M.%S
    outbase = '{}-{}_job-{}_{}_{}'.format(
                    sweep_name, sweep_val, job_id, out_suffix, timestamp)

    out_dir = os.path.join(sim_outdir, 'analysis')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fname_html = outbase + '.html'
    fpath_html = os.path.join(out_dir, fname_html)
    fname_ipynb = outbase + '.ipynb'
    fpath_ipynb = os.path.join(out_dir, fname_ipynb)


    print("\n{rule}\nPROCESSING: {outdir}\n{rule}\n".format(
          outdir=sim_outdir, rule='-'*80))

    # Execute notebook (optional: --allow-errors)
    nb_exec_cmd = ("jupyter nbconvert --ExecutePreprocessor.timeout=None "
                   "--execute --to notebook {infile} "
                   "--output={outname} "
                   "--output-dir={outdir}").format(infile=nb_path,
                                                   outname=fname_ipynb,
                                                   outdir=out_dir)

    # Remove temp files
    # os.remove(conf_path)

    nb_export_cmd = ("jupyter nbconvert {infile} --to html --template=toc2 "
                   "--output-dir={outdir}").format(infile=fpath_ipynb,
                                                   outdir=out_dir)

    # Use subprocess.check_output to raise Exception when command fails
    print("Executing notebook for output: " + sim_outdir)
    exe_status = subprocess.call(nb_exec_cmd, shell=True, env=env)

    print("Finished! Saving notebook to output directory ...")
    sav_status = subprocess.call(nb_export_cmd, shell=True)

    # print("Adjusting HTML report title ...")
    # nb_name = os.path.splitext(os.path.basename(nb_path))[0]
    # subs_regex = "'1,6s/<title>{}/<title>{}-{}/g'".format(outbase, sweep_name, sweep_val)
    # sed_command = "sed -i {regex} {file}".format(regex=subs_regex, file=fpath_html)
    # rename_status = subprocess.call(sed_command, shell=True)

    if not (exe_status == sav_status == 0):
        raise Exception('Notebook execution or saving failed for output {}'.format(
                         sim_outdir))

    return 0


# for sweep_index, sim_outdir in enumerate(output_dirs):
#     run_notebook(sim_outdir, sweep_index)

# Parallel version
pool = Pool(4)
args1 = output_dirs
args2 = range(len(output_dirs))
args_zip = zip(args1, args2)
results = pool.map(process_sim_outputs, args_zip)

print("Finished executing notebooks.")