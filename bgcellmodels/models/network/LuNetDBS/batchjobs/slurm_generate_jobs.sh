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
configs/sweeps_gsyn/netconf-V6_rec-axmid-EI_gs-gaba-taurec-70.json
"
readarray -t configs <<< "${outputs_clipboard}"

# Common options for all simulations
start_seed=888

# Number of cluster nodes and processes per nodes
# SONIC has 22 cores (threads) per node, so max ppn = 22
num_nodes=1
num_ppn=44
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
        walltime="4:30:00"

        # Arguments passed to simulation script using -v
        # NOTE: comment out option to read it from config, if not it is overridden

        # Cluster configuration
        model_args["numproc"]="${num_proc}"

        # Simulation settings
        model_args["seed"]="$((start_seed+seed))"
        model_args["dur"]="3000"
        model_args["scale"]="1.0"
        # model_args["nodbs"]="1"
        # model_args["nolfp"]="1"
        # model_args["simdt"]="0.025"
        model_args["dd"]="1"

        # Configuration files
        model_args["morphdir"]="$HOME/workspace/bgcellmodels/bgcellmodels/models/STN/Miocinovic2006/morphologies"
        model_args["configdir"]="$HOME/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/configs"
        model_args["simconfig"]="${sim_config}"
        model_args["cellconfig"]="dummy-cells_axons-full_CST2.json"
        model_args["axonfile"]="axon_coordinates_mouselight.pkl"

        # Output configuration
        model_args["outdir"]="$HOME/scratch"
        model_args["transientperiod"]="1000.0"
        model_args["writeinterval"]="10e3"
        model_args["reportinterval"]="50.0"
        # model_args["progress"]="0" # new Sonic updates log files in real-time

        # Concatenate arguments, comma-separated
        model_arg_string="dummyvar=0"
        for key in ${!model_args[@]}; do
            model_arg_string+=",${key}=${model_args[${key}]}"
        done

        # Make submission command
        # NOTE: different format of short/long arguments: -a arg / --argument=arg
        # --ntasks=${num_proc}
        submit_cmd="sbatch --job-name=LuNetDBS --time=${walltime} --nodes=${num_nodes} --ntasks-per-node=${num_ppn} --export=ALL,${model_arg_string} ${job_script}"

        echo -e "Job submission command is:\n> $submit_cmd"
        eval $submit_cmd
    done
done

echo "To cancel a job use: scancel <job-id>"
echo "To cancel all jobs use: scancel -u $UID"
echo -n "Waiting for 5 seconds to check job status"
for i in {1..5}; do
    echo -n "."
    sleep 1
done
echo ""

squeue -u 15203008
