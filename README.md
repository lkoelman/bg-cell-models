
# Cell models

## Overview of BG cell models & network models

See notes in `notes/BG_modelling_studies.md`

## Included cell models

- `./Otsuka/` Otsuka (2004) STN cell model


- `./GilliewWillshaw/` Gillies & Willshaw (2005) STN cell model
	
	+ [original model in NEURON, simplified morphology](https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=74298)

	+ [same channel mechanisms in morphological reconstruction of STN neuron](https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=151460)


- `./Gunay/` Gunay et al. (2008) GPe cell model

	+ [original model in GENESIS](https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=114639)

	+ [adapted model used in subsequent papers](https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=136315)

	+ [NEURON port of channel mechanisms for use in single-compartment model](https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=143100)


## Choice of model parameters

See notes in `notes/BG_observations_experiments.md`

--------------------------------------------------------------------------------

# Installation

These installation instructions are for the Conda Python ditribution.

Our custom PyNN classes have only been tested with PyNN 0.9.2 (git commit
`4b80343ba17afd688a92b3a0755ea73179c9daab`) and BluePyOpt version 1.5.35 
(commit `7d7c65be0bff3a9832d1d5dec3f4ba1c8906fa75`).


## Installation using Pip / setuptools

Install editable version (symlink to this directory):

```sh
pip install -e path_to_repo
# or alternatively:
python setup.py develop

# Build NMODL files automatically:
python setup.py build_mechs
```

## Installation using PYTHONPATH

This will make the module globally available by adding it to the `PYTHONPATH` environment variable.

```bash
# Create Python virtual environment using Conda
conda create -n neuro python=2
source activate neuro

# Install dependencies in new environment
pip install scipy matplotlib cython numba pint elephant PySpike
git clone https://github.com/BlueBrain/BluePyOpt.git
pip install -e ./BluePyOpt # -e only if you want editable version

git clone https://lkmn_ucd@bitbucket.org/lkmn_ucd/bg-cell-models.git bgcellmodels
cd bgcellmodels
git checkout --track origin/nothreadsafe
# append the following to your ~/.bashrc or ~/.bash_profile file:
export PYTHONPATH=$PYTHONPATH:$HOME/workspace/bgcellmodels

# Start jupyter from the environment where nb_conda is installed
source activate
jupyter notebook
```



--------------------------------------------------------------------------------
# Install Supporting Tools (optional)

## NEURON Syntax Definitions

NEURON Syntax definitions for Hoc and NMODL languages [for Sublime Text](https://github.com/jordan-g/NEURON-for-Sublime-Text) and [for VS Code](https://github.com/imatlopez/vscode-neuron).


## nb_conda

Allows you to select a conda environment as Jupyter kernel.

```sh
conda install nb_conda
```

To enable an environment as a kernel, install the module ipykernel in it:

```sh
conda install -n my_environment ipykernel
```

## nbstripout

This facilitates working with git and Jupyter notebooks by stripping output cells before commits. This avoids committing large binary outputs like images embedded into notebooks.

```bash
pip install --upgrade nbstripout
cd bgcellmodels
nbstripout --install
```

## Jupyter extensions

```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

Now extensions can be enabled via the tab 'Nbextensions' in Jupyter. For example, 'Table of Contents' will add a handy navigation widget when working with notebooks.

--------------------------------------------------------------------------------
# Running models

See model-specific `README` file in its subdirectory.

## BluePyOpt optimizations

```sh
# For Parallel BluePyOpt:
# In another terminal session, create ipyparallel instances in new environment
source activate neuro
cd ~/workspace/bgcellmodels
ipcluster start
```