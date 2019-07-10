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

Our custom PyNN classes have only been tested with:
- PyNN 0.9.2 (commit id `4b80343ba17afd688a92b3a0755ea73179c9daab`)
- BluePyOpt version 1.5.35 (commit id `1456941abe425b4a20fb084eab1cb6415ccfe2b8`).


## Installation

Install editable version (symlink to this directory):

```sh
# Install dependencies
pip install lazyarray pint elephant

git clone https://github.com/BlueBrain/BluePyOpt.git
cd BluePyOpt && git checkout 1456941abe425b4a20fb084eab1cb6415ccfe2b8
pip install -e .

git clone https://github.com/lkoelman/PyNN.git
cd PyNN && git checkout lkmn-multicomp
cd pyNN/neuron/nmodl && nrnivmodl
cd ../../.. & pip install -e .

git clone https://github.com/lkoelman/python-neo.git
cd python-neo && git checkout lkmn-dev # development version with MATLAB annotation support
pip install -e .

git clone https://github.com/lkoelman/LFPsim.git
cd LFPsim/lfpsim && nrnivmodl
pip install -e ./LFPsim


# Install package
git clone https://lkmn_ucd@bitbucket.org/lkmn_ucd/bg-cell-models.git bgcellmodels
cd bgcellmodels && git checkout --track origin/nothreadsafe
python setup.py develop # or: pip install -e path_to_repo
python setup.py mechanisms # Build NMODL files automatically:
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
