#!/bin/bash -l

# Job to be submitted
job_script="${HOME}/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/batchjobs/runjob_bgnetmodel.sh"

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
        job_resources="walltime=6:00:00,nodes=${num_nodes}:ppn=${num_ppn}"

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
        model_args["outdir"]="$HOME/storage"
        model_args["transientperiod"]="2000.0"
        model_args["writeinterval"]="10e3"
        model_args["reportinterval"]="50.0"
        model_args["progress"]="1"

        # Concatenate arguments, comma-separated
        model_arg_string="dummyvar=0"
        for key in ${!model_args[@]}; do
            model_arg_string+=",${key}=${model_args[${key}]}"
        done

#         model_args="numproc=${num_proc},\
# dur=1e3,scale=0.5,seed=$((start_seed+seed)),dbs=1,lfp=1,dd=1,\
# outdir=$HOME/storage,transientperiod=0.0,writeinterval=5e3,\
# reportinterval=50.0,\
# morphdir=$HOME/workspace/bgcellmodels/bgcellmodels/models/STN/Miocinovic2006/morphologies,\
# configdir=$HOME/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/configs,\
# simconfig=${sim_config},cellconfig=test_cellconfig_5.json,\
# axonfile=axon_coordinates.pkl"
        
        # Make qsub command
        qsub_command="qsub ${job_script} -l ${job_resources} -v ${model_arg_string}"
        echo -e "Submitting qsub command:\n> $qsub_command"
        eval $qsub_command
    done
# for fburst in {5,10,20,25,50,100}; do
#         qsub_command="qsub ${job_script} \
# -l walltime=02:30:00 \
# -v ncell=100,dur=26000,burst=${fburst},config=${conf}"

#         echo -e "Submitting qsub command:\n> $qsub_command"
#         eval $qsub_command
#     done
done

echo -n "Waiting for 5 seconds to check job status"
for i in {1..5}; do
    echo -n "."
    sleep 1
done
echo ""

qstat
