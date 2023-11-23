#!/bin/bash
set -eo pipefail

# Install specific versions to prevent conflicts due to use of ">="
python -m pip install -r requirements-minimal.txt

python -m pip install -e BluePyOpt

pushd cd PyNN/pyNN/neuron/nmodl
nrnivmodl
popd
python -m pip install -e PyNN

pushd LFPsim/lfpsim
nrnivmodl
popd
python -m pip install -e LFPsim

# Uinstall pip-installed neo and force installation of patched version
python -m pip uninstall -y neo
set -e
python -m pip install -e python-neo
set +e

# Install bgcellmodels
pip install -e .
python setup.py mechanisms # Build NMODL files automatically