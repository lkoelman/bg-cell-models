#!/bin/bash

set -eo pipefail

# Works on focal & bionic
apt install -y -qq --no-install-recommends libncurses5-dev libncursesw5-dev libreadline-dev libgsl-dev


# Create conda environment for Python2.7
nrn_env=nrn
conda create -y -n ${nrn_env} python=2.7
conda init
# Copied from .bashrc section created by conda init:
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
# source ~/.bashrc
conda activate ${nrn_env}

# Install Python MPI support (this will also install MPI libs/binaries)
conda install -y -c conda-forge mpi4py openmpi openmpi-mpicc gxx_linux-64 cython

# Get the latest nrn-XX.tar.gz and iv-XX.tar.gz from www.neuron.yale.edu/ftp/neuron/versions
nrn_installdir=$HOME/neuron7
nrn_tarball="${nrn_installdir}/nrn.tar.gz"
mkdir -p $nrn_installdir

wget https://neuron.yale.edu/ftp/neuron/versions/v7.7/nrn-7.7.tar.gz -O ${nrn_tarball}

# Extract all neuron source code
cd $nrn_installdir
tar -xzf ${nrn_tarball}
mv nrn-* nrn

# Alternatively, clone the git repo:
# git clone --branch=7.7 https://github.com/neuronsimulator/nrn.git
# cd nrn
# ./build.sh 
# at this point the rest of the build process is identical as if you
# had just extracted a nrn-X.X.tar.gz file


# Install NEURON
cd ${nrn_installdir}/nrn

# Make sure NEURON is linked to MPI binaries that your Python distribution is using
conda_home=/opt/conda/envs/${nrn_env} # find using `conda info --base`
# export PATH=$PATH:$conda_home/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$conda_home/lib
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH:-}:$conda_home/lib

# TODO: alternatively install openmpi using apt and mpi4py using pip
export MPICC=`which mpicc`
export MPICXX=`which mpicxx`

nrn_python_bin=${conda_home}/bin/python2.7

./configure --prefix=`pwd` \
    --without-iv \
    --without-x \
    --with-nrnpython=$nrn_python_bin \
    --with-mpi \
    --with-paranrn \
    --without-memacs
make
make install
echo "Successfully installed NEURON in ${nrn_installdir}/nrn"

# Install the NEURON Python module.
cd $nrn_installdir/nrn/src/nrnpython
python setup.py install

# Make NEURON executables executable:
nrn_bindir=$nrn_installdir/nrn/bin
cd $nrn_bindir
chmod +x nrnivmodl nrngui neurondemo nrnocmodl
echo "export PATH=\$PATH:$nrn_bindir" >> ~/.bashrc

# Always start in the neuron python environment
echo "conda activate ${nrn_env}" >> ~/.bashrc