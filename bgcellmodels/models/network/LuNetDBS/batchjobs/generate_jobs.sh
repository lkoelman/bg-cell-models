#!/bin/bash -l

# Job to be submitted
job_script="${HOME}/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/batchjobs/runjob_bgnetmodel.sh"

# Config files you want to repeat with different seeds (one per line)
outputs_clipboard="test_dbs_silent_dbs-amp-1.json
test_dbs_silent_dbs-amp-2.json
test_dbs_silent_dbs-amp-4.json
test_dbs_silent_dbs-amp-5.json"
readarray -t configs <<< "${outputs_clipboard}"

# Common options for all simulations
start_seed=888

num_nodes=1
num_ppn=8
num_proc=$((num_nodes*num_ppn))

# Can do it using associative array (dictionary):
declare -A model_args

for sim_config in "${configs[@]}"; do
    for seed in {0..0}; do

        # Resources requested using -l
        job_resources="walltime=3:00:00,nodes=${num_nodes}:ppn=${num_ppn}"

        # Arguments passed to simulation script using -v
        # category - cluster simulation
        model_args["numproc"]="${num_proc}"
        
        # category - model configuration
        model_args["dur"]="1e3"
        model_args["scale"]="0.5"
        model_args["seed"]="$((start_seed+seed))"
        model_args["dbs"]="1"
        model_args["lfp"]="1"
        model_args["dd"]="1"
        model_args["morphdir"]="$HOME/workspace/bgcellmodels/bgcellmodels/models/STN/Miocinovic2006/morphologies"
        model_args["configdir"]="$HOME/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/configs"
        model_args["simconfig"]="${sim_config}"
        model_args["cellconfig"]="test_cellconfig_5.json"
        model_args["axonfile"]="axon_coordinates.pkl"
        
        # category - outputs
        model_args["outdir"]="$HOME/storage"
        model_args["transientperiod"]="0.0"
        model_args["writeinterval"]="5e3"
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
