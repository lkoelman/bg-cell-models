# TODO List

## Porting code from GENESIS

- DONE: write mechanisms.json
  - [X] find out how Ephys makes SectionLists
      => based on identified SWC types

### Gunay (2008)

For model at https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=114639
Gunay, Edgerton, and Jaeger (2008). Channel Density Distributions Explain Spiking Variability in the Globus Pallidus: A Combined Physiology and Computer Simulation Database Approach.

- main script in /runs/runsample/setup.g loads following scripts:
  
  + /runs/runsample/readGPparams.g
      + `/common/GP<i>_default.g`       -> sets param variables
      + `/common/actpars.g`             -> sets param variables
  
  + /common/make_GP_libary.g          -> uses param variables
      + `/common.GPchans.g`             -> defines mechanisms using parameters
      + `/common.GPcomps.g`             -> defines compartments using parameters

### Hendrickson (2011)

For model at https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=127728
"Comparison of full and reduced globus pallidus models (Hendrickson 2010)" see:
  
+ main script in /articleCode/scripts/genesisScripts/GP1axonless_full_synaptic.g
  First it loads variables from following scripts:
    + /articleCode/commonGPFull/GP1_defaults.g
    + /articleCode/commonGPFull/simdefaults.g
    + /articleCode/commonGPFull/actpars.g

+ The variables are then used in following scripts, in braces: {varname}
  (`GP1axonless_full*.g` -> `make_GP_libary.g` -> ...)
    + /articleCode/commonGPFull/GP1_axonless.p
    + /articleCode/commonGPFull/GPchans.g
    + /articleCode/commonGPFull/GPcomps.g

### Edgerton (2010)

For model at https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=136315
"Globus pallidus neuron models with differing dendritic Na channel expression 
(Edgerton et al., 2010)", see: 

  + main script in /run_example/run_vivo_example.g, loads scripts:
  + ../common/GP1_constants.g
  + ../common/biophysics/GP1_passive.g
  + ../common/biophysics/GP1_active.g
  + .. see lines with 'getarg' statement for modification of scaling factors/gradients

### Schulthess (2011)

Model at https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=137846

- see main script where following files are loaded:
  + ./paspars.g for passive parameters
  + ../actpars.g for active parameters