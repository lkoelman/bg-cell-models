#!/bin/bash -l

################################################################################
# README
################################################################################

# Usage of `sbatch` for job submission: https://slurm.schedmd.com/sbatch.html
# Precedence of SLURM options: 
#       command line options > environment variables > options in batch script

################################################################################
# JOB GENERATION
################################################################################

# Job to be submitted
job_script="${HOME}/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/batchjobs/slurm_runjob_netsim.sh"

# Config files you want to repeat with different seeds (one per line)
outputs_clipboard="
configs/sweeps_dbs-amp/axons-full-V2_dbs-amp-0.1.json
configs/sweeps_dbs-amp/axons-full-V2_dbs-amp-1.json
configs/sweeps_dbs-amp/axons-full-V2_dbs-amp-10.0.json
"
readarray -t configs <<< "${outputs_clipboard}"

# Common options for all simulations
start_seed=888

# Number of cluster nodes and processes per nodes
# SONIC has 24 cores (threads) per node, so max ppn = 24
num_nodes=1
num_ppn=24
num_proc=$((num_nodes*num_ppn))

# Can do it using associative array (dictionary):
declare -A model_args

for sim_config in "${configs[@]}"; do

    if [[ ${sim_config} == "" ]]; then
        continue
    fi

    for seed in {0..0}; do

        # Resources requested using -l
        # walltime ~= 1:20 for 16 ppn, 7000ms, dt=0.025
        walltime="6:00:00"

        # Arguments passed to simulation script using -v
        # NOTE: comment to read option from config, if not it is overridden

        # Cluster configuration
        model_args["numproc"]="${num_proc}"
        
        # Model configuration
        model_args["seed"]="$((start_seed+seed))"
        # change block below for calibration
        # -------------------------
        model_args["dur"]="3000"
        model_args["scale"]="1.0"
        # model_args["nodbs"]="1"
        # model_args["nolfp"]="1"
        # model_args["simdt"]="0.01"
        # -------------------------
        model_args["dd"]="1"
        model_args["morphdir"]="$HOME/workspace/bgcellmodels/bgcellmodels/models/STN/Miocinovic2006/morphologies"
        model_args["configdir"]="$HOME/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/configs"
        model_args["simconfig"]="${sim_config}"
        model_args["cellconfig"]="dummy-cells_axons-cutoff.json"
        model_args["axonfile"]="axon_coordinates_cutoff.pkl"
        
        # Output configuration
        model_args["outdir"]="$HOME/scratch"
        model_args["transientperiod"]="2000.0"
        model_args["writeinterval"]="10e3"
        model_args["reportinterval"]="50.0"
        model_args["progress"]="1"

        # Concatenate arguments, comma-separated
        model_arg_string="dummyvar=0"
        for key in ${!model_args[@]}; do
            model_arg_string+=",${key}=${model_args[${key}]}"
        done
        
        # Make submission command
        # NOTE: different format of short/long arguments: -a arg / --argument=arg
        submit_cmd="sbatch ${job_script} --job-name=LuNetDBS --time=${walltime} --nodes=${num_nodes} --ntasks-per-node=${num_ppn} --export=ALL,${model_arg_string}"

        echo -e "Job submission command is:\n> $submit_cmd"
        eval $submit_cmd
    done
done

echo -n "Waiting for 5 seconds to check job status"
for i in {1..5}; do
    echo -n "."
    sleep 1
done
echo ""

squeue
