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

# TODO

- [ ] explain in README what model belongs to what chapter
- [ ] explain how to run the model

--------------------------------------------------------------------------------

# Installation

These installation instructions are for the Conda Python ditribution.

Our custom PyNN classes have only been tested with:
- PyNN 0.9.2 (commit id `4b80343ba17afd688a92b3a0755ea73179c9daab`)
- BluePyOpt version 1.5.35 (commit id `1456941abe425b4a20fb084eab1cb6415ccfe2b8`).

## Dockerized install

Install the models in a Docker container:

```bash
cd docker
docker build . -t neuron7

cd ..
docker run -v $(pwd):/bgmodel -it neuron7 bash

cd /bgmodel
./install.sh
```

## Installation

Installation in a UNIX environment where NEURON 7.X with MPI support is installed.

First activate the python environment where Neuron and mpi4py are installed.
Then:

```bash
git clone --recurse-submodules https://github.com/lkoelman/bg-cell-models.git
cd bg-cell-models
./install.sh
```


To install Neuron 7 on a cluster, see https://gist.github.com/lkoelman/49e7105b5c54128fe1c35d4e2d6b7273


--------------------------------------------------------------------------------

# Supporting Tools (optional)

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
