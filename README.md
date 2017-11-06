
# Physiological & Experimental Parameters

See notes in `notes/BG_observations_experiments.md`


# Overview of BG cell & network models

See notes in `notes/BG_modelling_studies.md`


# Installation

These installation instructions are for the Conda Python ditribution:

```bash
# Create Python virtual environment using Conda
conda install nb_conda # use of environments in Jupyter kernels
conda create -n neuro python=2
source activate neuro

# Install dependencies in new environment
pip install scipy matplotlib cython numba elephant PySpike
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

## Install useful tools

### nbstripout

```bash
pip install --upgrade nbstripout
cd bgcellmodels
nbstripout install
```

### ipython extensions

```bash
pip install jupyter_contrib_nbextensions
jupyter nbextensions_configurator enable --user
```

Now extensions can be enabled via Jupyter extensions tab