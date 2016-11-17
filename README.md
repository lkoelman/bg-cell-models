# TODO PLAN #

- first compare balance of currents Otsuka/Gillies/RubinTerman/KhaliqRaman papers and see which currents are responsible for the different features of STN cell dynamics (e.g. bursts, spontaneous pacemaking, ...)
	- (plot equilibrium activations variables for each current)
	- implement experiments for different firing/physiology features from all papers
		- go through paper Gillies and add experiments to Otsuka test script
		- also add experiments for INaR from DoBean STN/Purkinje & Khaliq Purkinje papers
	- run experiment for each feature in each model

- Construct equivalent reduced model
	- reduction method: see principles of computational modeling, Ch. 8 & Ch. 4.3.3
		1. Simplify tree morphology (see Ch. 4.3.3) & p. 85
			- iteratively replace cylinders using technique in Box 3.2 and 3/2 assumption
		2. Tune gbar ration & and geom+electrical properties of equivalent compartment to reproduce important behaviours/experiments of full model
			- tune coupling of two compartments 
				- see cable equation (NEURON book Ch3 p. 26 eq. 3.47 (pdf p. 114), or) Sterrat book eq. 2.24
				- d and Ra determine input resistance for axial current
	- pick the balance of currents & implementations that agree most with other STN physiology papers
		- possibly enhance rebound bursts according to Otsuka model to make sure they are robust
	- then replace the Na current with the more detailed Na channel model of Khaliq & Raman model for Purkinje cells
		- but retain the relative magnitude of gnabar and see that you can still reproduce main features of Otsuka & Gillies models
			- potentially include two na mechanisms (one for persistent/transient, one for resurgent)
			- sum of new Na currents must be equal to sum of existing Na currents

- design experiment to test if resurgent Na currents promote additional patterning unexplained by models without this current


## TODO NEXT ##

- get parameters state model slow inactivaton & recover
	- check parameters of Kuo & Bean (1994) model for inactivation/recovery model are the same in the Khaliq/Raman/Akeman models published on ModelDB
		- see paper fig. 7 => not same
	- see also [Blair & Bean (2003)](Role of Tetrodotoxin-Resistant Na Current Slow Inactivation in Adaptation of Action Potential Firing in Small-Diameter Dorsal Root Ganglion Neurons)

- See if other currents need to be adjusted to accomodate new current INa_rsg
	- e.g. reduce Ih/HCN or change its parameters

# References/sources #

- Resurgent sodium current model implementation & experiments
	- do ModelDB search for 'narsg'


# Notes #

- comparison of Otsuka/RubinTerman/GilliesWillshaw model: 
	- see Encyclopedia of Computational Neuroscience, p. 2909 (pdf p. 2961)

- voltage dependence of resurgent Na current is similar in STN and Purkinje neurons
	- stated in [Do & Bean (2004)](Sodium currents in subthalamic nucleus neurons from Nav1.6-null mice.)

- the Kuo & Bean (1994) Markov state model gives reasonable simulations of the voltage-dependence of development of inactivation and recovery from inactation for sodium channels that do not give resurgent current
	- [Raman & Bean (2001)](Inactivation and Recovery of Sodium Currents Neurons in Cerebellar Purkinje Neurons: Evidence for Two Mechanisms)


- Raman & Bean (2001) channel model ONLY models INa_rsg compoment of INa
	- i.e. you will still have to account of transient and persistent components
	- this is confirmed by plotting INa and INarsg in code supplied with Akemann and Knoepfel, J.Neurosci. 26 (2006) 4602
		- see