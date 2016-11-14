# TODO #

- first compare balance of currents Otsuka/Gillies/RubinTerman/KhaliqRaman papers and see which currents are responsible for the different features of STN cell dynamics (e.g. bursts, spontaneous pacemaking, ...)
	- plot equilibrium activations variables for each current
	- go through paper Gillies and add experiments to Otsuka test script
	- also add experiments for INaR from DoBean STN/Purkinje & Khaliq Purkinje papers
	- run experiment for each feature in each model

- Construct equivalent reduced model
	- pick the balance of currents & implementations that agree most with other STN physiology papers
		- possibly enhance rebound bursts according to Otsuka model to make sure they are robust
	- then replace the Na current with the more detailed Na channel model of Khaliq & Raman model for Purkinje cells
		- but retain the relative magnitude of gnabar and see that you can still reproduce main features of Otsuka & Gillies models

- design experiment to test if resurgent Na currents promote additional patterning unexplained by models without this current


## TODO breakdown ##
- two-comparment simplification
	- see principles of computational modeling, Ch. 8 & Ch. 4.3.3
		1. Simplify tree morphology (see Ch. 4.3.3)
			- iteratively replace cylinders using technique in Box 3.2 and 3/2 assumption
		2. Tune all gbar and geom/electrical properties of equivalent compartment to reproduce important behaviours/experiments of full model
			- tune coupling of two compartments 
				- see cable equation (NEURON book Ch3 p. 26 eq. 3.47 (pdf p. 114), or) Sterrat book eq. 2.24
				- d and Ra determine input resistance for axial current


# Model comparison #

- comments Encyclopedia of Computational Neuroscience, p. 2911
	- 