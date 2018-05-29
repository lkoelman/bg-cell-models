# Installation

Our custom PyNN classes have only been tested with PyNN 0.9.2 (git commit ID 
`4b80343ba17afd688a92b3a0755ea73179c9daab`) and BluePyOpt version 1.5.35 
(commit ID `7d7c65be0bff3a9832d1d5dec3f4ba1c8906fa75`).

```sh
# Install PyNN
git clone https://github.com/NeuralEnsemble/PyNN.git
cd PyNN
git checkout 4b80343ba17afd688a92b3a0755ea73179c9daab
cd pyNN/neuron/nmodl
nrnivmodl # compile MOD mechanisms
cd ../../..
pip install -e ./PyNN
pip install lazyarray # PyNN dependency

# Make sure all common and model-specific mechanisms are compiled using nrnivmodl
```
# Code Architecture

See UML diagrams in `extensions/pynn/PyNN_UML_diagrams.html` and the classes
defined in the Python source files in that directory. The diagrams clarify where
our custom classes hook into the PyNN machinery for setting up and running
the simulation.