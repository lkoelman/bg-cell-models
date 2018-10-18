#!/bin/bash -l

################################################################################
# QSUB CONFIGURATION
################################################################################

# Set the number of nodes & processors per node
#PBS -l nodes=1:ppn=12

# Set the walltime of the job to 1 hour (format is hh:mm:ss)
# - walltime for 100 cells/populations is sim_time * 2031.75 / 10.000 * 1.25 [seconds]
# - to get hh and mm (for hh:mm:ss) do hh = runtime // 3600; mm = runtime % 3600
## PBS -l walltime=00:45:00

# Declares that all environment variables in the qsub command's
# environment are to be exported to the batch job.
#PBS -V

# Working directory where submitted script will be executed ($PBS_O_INITDIR)
# This must be a full path.
#PBS -d /home/people/15203008/workspace/bgcellmodels/bgcellmodels/models/network/KlmnNetMorpho

# Specifies the jobname. The default name is the script name (basename)
## PBS -N BG_network_model

# Defines the path to be used for the standard output stream of the batch job.
# The default output is <job_name.job_number> which is pretty clear
## PBS -o ./output_bgnetmodel.o.log
# Same for standard error stream:
## PBS -e ./output_bgnetmodel.e.log

# E-mail on begin (b), abort (a) and end (e) of job
#PBS -m bae

# E-mail address of recipient
#PBS -M lucas.koelman@ucdconnect.ie


################################################################################
# JOB SCRIPT
################################################################################

# Execute using:
# >>> qsub runjob_lucastest.sh -l walltime=00:45:00 \
# >>> -v ncell=100,dur=10000,seed=15203008,\
# >>> config=~/workspace/bgcellmodels/models/KlmnNetMorpho/configs/simple_config.json,
# >>> outdir=~/storage,id=with_DA_depleted_1


echo -e "
Directory where qsub was called:        $PBS_O_WORKDIR
Working directory for submitted script: $PBS_O_INITDIR
The job id is                           $PBS_JOBID
The job name is                         $PBS_JOBNAME
"

# Setup environment
module load intel-mpi gcc anaconda
source activate localpy27

# Get all the paths
if [ -z "$outdir" ]; then
    outdir="~/storage"
fi

model_dir="${HOME}/workspace/bgcellmodels/bgcellmodels/models/network/KlmnNetMorpho"
model_script=model_parameterized.py
model="${model_dir}/${model_script}"
config_file="configs/${config}"
model_config="${model_dir}/${config_file}"

# Command to be evaluated
mpi_command="mpirun -n 24 python ${model} \
--ncell ${ncell} --dur ${dur} \
--transient-period 0.0 --write-interval 27e3 \
--no-gui --progress --config ${model_config} \
--outdir ${outdir} -id ${PBS_JOBID}"

# Optional arguments passed to python script
opt_names=("seed" "burst" "write-interval" "transient-period")
for optname in "${opt_names[@]}"; do
    if [ -n "${!optname}" ]; then
        mpi_command="${mpi_command} --${optname} ${!optname}"
    fi
done

opt_names=("wi" "tp")
for optname in "${opt_names[@]}"; do
    if [ -n "${!optname}" ]; then
        mpi_command="${mpi_command} -${optname} ${!optname}"
    fi
done

# Optional flags passed to python script
flag_names=("lfp" "no-lfp")
for flagname in "${flag_names[@]}"; do
    if [ -n "${!flagname}" ]; then
        mpi_command="${mpi_command} --${flagname}"
    fi
done

cd $model_dir

# Sanity check
echo -e "
--------------------------------------------------------------------------------
Executing script with following inputs:

- ncell = ${ncell}
- dur = ${dur}
- outdir = ${outdir}

--------------------------------------------------------------------------------
The final command (following '> ') is:

> $mpi_command

--------------------------------------------------------------------------------
Version information for model code:

$(git log -1)

--------------------------------------------------------------------------------
The contents of the model configuration file is:

"

cat $model_config

echo -e "
Model output follows below line:
================================================================================
"

# Evaluate our command
eval $mpi_command
