
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

## Installation using PYTHONPATH

This will make the module globally available by adding it to the `PYTHONPATH` environment variable.

```bash
# Create Python virtual environment using Conda
conda install nb_conda # use of environments in Jupyter kernels
conda create -n neuro python=2
source activate neuro

# Install dependencies in new environment
pip install scipy matplotlib cython numba pint elephant PySpike
git clone https://github.com/BlueBrain/BluePyOpt.git
pip install -e ./BluePyOpt # -e only if you want editable version

git clone https://lkmn_ucd@bitbucket.org/lkmn_ucd/bg-cell-models.git bgcellmodels
git checkout --track origin/nothreadsafe
echo "export PYTHONPATH=$PYTHONPATH:/home/myhome/workspace/bgcellmodels" >> ~/.bashrc

# Start jupyter from the root environment where nb_conda is installed
source activate
jupyter notebook

# In another terminal session, create ipyparallel instances in new environment
source activate neuro
cd ~/workspace/bgcellmodels
ipcluster start
```

## Installation using Pip

TODO

```sh
pip install -e ./bg-cell-models
```

## Installation of Extra Tools

### NEURON Syntax Definitions

NEURON Syntax definitions for Hoc and NMODL languages [for Sublime Text](https://github.com/jordan-g/NEURON-for-Sublime-Text) and [for VS Code](https://github.com/imatlopez/vscode-neuron).

### nbstripout

This facilitates working with git and Jupyter notebooks by stripping output cells before commits. This avoids committing large binary outputs like images embedded into notebooks.

```bash
pip install --upgrade nbstripout
cd bgcellmodels
nbstripout --install
```

### Jupyter extensions

```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

Now extensions can be enabled via the tab 'Nbextensions' in Jupyter. For example, 'Table of Contents' will add a handy navigation widget when working with notebooks.

