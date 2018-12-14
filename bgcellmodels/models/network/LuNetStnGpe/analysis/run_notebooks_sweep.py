#!/usr/bin/env python

"""
Automatically run a notebook with each simulation output.

Usage
-----

- set all script parameters marked 'SETPARAM' appropriately

- can run multiple versions simultaneously, since script is compiled to bytecode
  upon execution
"""

import os
import subprocess
import re

nb_dir = "/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/analysis"
nb_infile = "lunet_stn-gpe_analysis.ipynb"
nb_path = os.path.join(nb_dir, nb_infile)

# SETPARAM: change filenames for simultaneous runs of this script
log_path = os.path.join(nb_dir, "nb_exec_list_copy2.log") # change for copies of this script
conf_path = os.path.join(nb_dir, "nb_exec_conf_copy2.py") # change for copies of this script


# SETPARAM: List simulation output directories to analyze
outputs_clipboard = """
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q7_sweep_gpe-stn-gaba-AB-ratio/LuNetStnGpe_2018.12.13_14.22.55_job-1191908.sonic-head_syn-V18_gpe-stn_gabab-x-0.1_gabaa-x-8
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q7_sweep_gpe-stn-gaba-AB-ratio/LuNetStnGpe_2018.12.13_14.22.56_job-1191909.sonic-head_syn-V18_gpe-stn_gabab-x-0.1_gabaa-x-10
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q7_sweep_gpe-stn-gaba-AB-ratio/LuNetStnGpe_2018.12.13_14.22.56_job-1191910.sonic-head_syn-V18_gpe-stn_gabab-x-0.1_gabaa-x-12
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q7_sweep_gpe-stn-gaba-AB-ratio/LuNetStnGpe_2018.12.13_14.22.59_job-1191907.sonic-head_syn-V18_gpe-stn_gabab-x-0.1_gabaa-x-6
"""

output_dirs = outputs_clipboard.strip().split()

# SETPARAM: name of sweep variable
sweep_name = "gaba_AB_ratio"

for sim_outdir in output_dirs:

    # SETPARAM: pattern for extraction of sweep variable from filename
    sweep_val_pattern = r"gabaa-x-([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
    match = re.search(sweep_val_pattern, sim_outdir)
    sweep_val = match.groups()[0]

    # SETPARAM: settings variables passed to executed notebook
    nb_pyvars = {
        'outputs': sim_outdir,
        'matfile_common_pattern': '-10000ms',
        'ROI_INTERVAL': (6e3, 10e3),
        'reference_phase': {'method': 'from_ctx', 'passband': (20.0, 30.0)},
        'sweep_var_name': sweep_name,
        'sweep_var_value': sweep_val,
        'automatic_execution': True,
        'pickle_filename': 'analysis_results_6.0s-10.0s_V1.pkl',
    }
    # use property eval(repr(object)) == object.
    nb_pyscript= "\n".join(("{} = {}".format(k, repr(v)) for k,v in nb_pyvars.items()))

    with open(conf_path, 'w') as conf_file:
        conf_file.write(nb_pyscript)

    # Add some environment variables to be read from notebook
    env = dict(os.environ)
    env["NB_CONF_FILE"] = conf_path

    # Specify outputs
    job_id = re.search(r'job-(\w+)', sim_outdir).groups()[0]
    ival = nb_pyvars['ROI_INTERVAL']
    # SETPARAM: name of saved notebook/html report
    out_suffix = '{:.1f}s-{:.1f}s_AUTO_V1'.format(ival[0]/1e3, ival[1]/1e3)
    nb_outfile = 'sweep-{}_job-{}_t-{}.ipynb'.format(sweep_val, job_id, out_suffix)
    nb_out_path = os.path.join(sim_outdir, nb_outfile)

    print("\n{rule}\nPROCESSING: {outdir}\n{rule}\n".format(
          outdir=sim_outdir, rule='-'*80))

    # Execute notebook (optional: --allow-errors)
    nb_exec_cmd = ("jupyter nbconvert --ExecutePreprocessor.timeout=None "
                   "--execute --to notebook {infile} "
                   "--output={outname} "
                   "--output-dir={outdir}").format(infile=nb_path,
                                                   outname=nb_outfile,
                                                   outdir=sim_outdir)

    nb_save_cmd = ("jupyter nbconvert {outfile} --to html --template=toc2 "
                   "--output-dir={outdir}").format(outfile=nb_out_path,
                                                   outdir=sim_outdir)

    # Use subprocess.check_output to raise Exception when command fails
    print("Executing notebook for output: " + sim_outdir)
    exe_status = subprocess.call(nb_exec_cmd, shell=True, env=env)

    print("Finished! Saving notebook to output directory ...")
    sav_status = subprocess.call(nb_save_cmd, shell=True)

    if not (exe_status == sav_status == 0):
        raise Exception('Notebook execution or saving failed for output {}'.format(
                         sim_outdir))


print("Finished executing notebooks.")