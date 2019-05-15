#!/bin/python

"""
Copy simulation log files in format 'jobname.o12345' and 'jobname.e12345'
to the corresponding output directory of simulation.
"""

import sys, os, re, shutil

if len(sys.argv) < 3 or sys.argv[1] in ('-h', '--help'):
    print("usage: python copy_logs.py /path/to/logsdir /path/to/simdirs")
    sys.exit(1)
else:
    logs_dir = sys.argv[1]
    out_dir = sys.argv[2]

outdir_contents = [os.path.join(out_dir, f) for f in os.listdir(out_dir)]
sim_outdirs = [d for d in outdir_contents if os.path.isdir(d)]

files = os.listdir(logs_dir)
for fname in files:
    matches = re.match(r'.+\.[oe]([0-9]+)$', fname)
    if matches:
        job_id = matches.groups()[0]

        # Find corresponding output directory
        sim_dir = next((d for d in sim_outdirs if 'job-{0}'.format(job_id) in d), None)
        if sim_dir is None:
            continue
        if fname in os.listdir(sim_dir):
            continue
        
        # Copy log file to simulation output dir
        fpath = os.path.join(logs_dir, fname)
        opath = os.path.join(sim_dir, fname)
        shutil.copy2(fpath, opath)
        print("{0} -> {1}".format(fname, sim_dir))
