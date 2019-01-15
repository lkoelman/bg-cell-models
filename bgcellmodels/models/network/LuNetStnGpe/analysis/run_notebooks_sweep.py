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

nb_dir = "/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/analysis"
nb_infile = "lunet_stn-gpe_analysis.ipynb"
nb_path = os.path.join(nb_dir, nb_infile)

# SETPARAM: List simulation output directories to analyze
output_dirs = """
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.56.57_job-1209416.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-1.30
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.56.57_job-1209418.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-1.10
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.56.58_job-1209420.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-0.90
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.56.58_job-1209421.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-0.80
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.56.58_job-1209427.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-0.20
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.57.01_job-1209417.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-1.20
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.57.01_job-1209422.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-0.70
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.57.06_job-1209419.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-1.00
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.57.14_job-1209425.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-0.40
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.57.17_job-1209424.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-0.50
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.57.19_job-1209423.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-0.60
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q8c_sweep-stn-EI_gmax-ctx-stn_BURST/LuNetStnGpe_2019.01.10_17.57.20_job-1209426.sonic-head_syn-V18_f-burst-20-25-30_g-ctx-stn-x-0.30
""".strip().split()

# SETPARAM: change filenames for simultaneous runs of this script
log_path = os.path.join(nb_dir, "nb_exec_list_copy3.log") # change for copies of this script
conf_path = os.path.join(nb_dir, "nb_exec_conf_copy3.py") # change for copies of this script

# SETPARAM: name of sweep variable
sweep_name = "gmax_ctx_stn"

for sim_outdir in output_dirs:

    # SETPARAM: pattern for extraction of sweep variable from filename
    sweep_val_pattern = r"x-([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
    match = re.search(sweep_val_pattern, sim_outdir)
    sweep_val = match.groups()[0]

    # Settings variables passed to executed notebook
    nb_pyvars = {}
    # SETPARAM: interval for signal analysis
    ROI_INTERVAL = (15e3, 19e3)
    # SETPARAM: filename containing recorded signals
    nb_pyvars['matfile_common_pattern'] = '-19000ms'
    ival_sec = [t/1e3 for t in ROI_INTERVAL]
    # SETPARAM: suffix for pickle file and jupyter notebook files
    out_suffix = '{:.1f}s-{:.1f}s_BURST-30hz_phase-from-ctx'.format(*ival_sec)
    # SETPARAM: refence phase signal and frequency band
    nb_pyvars['reference_phase'] = {'method': 'from_ctx', 'passband': (18.0, 32.0)}

    # prepare executable Python script using property eval(repr(object)) == object.
    nb_pyvars.update({
        'outputs': sim_outdir,
        'ROI_INTERVAL': ROI_INTERVAL,
        'sweep_var_name': sweep_name,
        'sweep_var_value': sweep_val,
        'automatic_execution': True,
        'pickle_filename': 'analysis_results_' + out_suffix,
    })
    nb_pyscript= "\n".join(("{} = {}".format(k, repr(v)) for k,v in nb_pyvars.items()))
    with open(conf_path, 'w') as conf_file:
        conf_file.write(nb_pyscript)

    # Add some environment variables to be read from notebook
    env = dict(os.environ)
    env["NB_CONF_FILE"] = conf_path

    # Specify outputs
    job_id = re.search(r'job-(\w+)', sim_outdir).groups()[0]
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