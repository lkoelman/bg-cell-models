# TODO PLAN #

1. first compare balance of currents Otsuka/Gillies/RubinTerman/KhaliqRaman papers and see which currents are responsible for the different features of STN cell dynamics (e.g. bursts, spontaneous pacemaking, ...)
	- (plot equilibrium activations variables for each current)
	- implement experiments for different firing/physiology features from all papers
		- go through paper Gillies and add experiments to Otsuka test script
		- also add experiments for INaR from DoBean STN/Purkinje & Khaliq Purkinje papers
	- run experiment for each feature in each model

2. Construct equivalent reduced model
	- reduction method: see principles of computational modeling, Ch. 8 & Ch. 4.3.3
		1. Simplify tree morphology (see Ch. 4.3.3) & p. 85
			- iteratively replace cylinders using technique in Box 3.2 and 3/2 assumption
		2. Tune gbar ration & and geom+electrical properties of equivalent compartment to reproduce important behaviours/experiments of full model
			- tune coupling of two compartments 
				- see cable equation (NEURON book Ch3 p. 26 eq. 3.47 (pdf p. 114), or) Sterrat book eq. 2.24
				- d and Ra determine input resistance for axial current
	- pick the balance of currents & implementations that agree most with other STN physiology papers
		- possibly enhance rebound bursts according to Otsuka model to make sure they are robust

3. Add bursting mechanism: removal of slow inactivation of Na currents by IPSPs
	- add slowly inactivating Na channels with de-inactivation by IPSPs
		- see refs `DoBean2003/Baufreton2005/Wilson2015`
	- then replace the Na current with the more detailed Na channel model of Do/Bean/Khaliq/Raman model
		- but retain the relative magnitude of gnabar and see that you can still reproduce main features of Otsuka & Gillies models
			- potentially include two na mechanisms (one for persistent/transient, one for resurgent)
			- sum of new Na currents must be equal to sum of existing Na currents
	- (design experiment to test if resurgent Na currents promote additional patterning unexplained by models without this current)

4. Add synapses
	- based on studies synapse distribution
	- use synapse mapping procedure Marasco & adapt topology of reduced model to preserve transformations and I/O characteristics of dendritic tree

5. Reduce GPe model
	- see ref

--------------------------------------------------------------------------------
# TODO NEXT #

- Test model reduction code
	- [x] Check scaling factors actually used VS paper in RedPurk.hoc
		- same as mentioned in paper: original/equivalent surface (if synapses in cluster: only count sections containing synapse toward original surface)
	- [x] See what happens to theoretical Rin in toy model
		- Calculate marasco reduction by hand for simple tree
		- calculated by hand and adapted expressions to make units match
		- in toy tree (secs P/A/B): Rin is conserved witn 0.2% accuracy for toy tree
			- NOTE that cluster root sections are not merged here with Marasco <eq> expressions
	- [x] See what happens to theoretical Rin in Gillies & Willshaw model
		- Calculate and compare input resistance of trees in full/reduced model (using algorithm Sterrat Ch. 4)
		- => if trees averaged/RaMERGINGMETHOD=1: they are not equal (see compare_models())
		- => if trees not averaged/RaMERGINGMETHOD=0: they are practically equal (within 1%)
	- [x] run other tests and see if all fail
		- => they fail, likely due to different input resistance
	- [x] check if mergingYmethod correctly implemented 
		- => NO: i used newri2 for `ri_seq -> diam_seq` (see merge_sequential())
		- however Marasco used newri2 only for `ri_seq` but not for `diam_seq` (see mergingYMethod())
	- [ ] Correct spontaneous firing rate through scaling/fitting
		- [x] read Chapt Sterrat parameter fitting
		- [x] compare values post-/pre-reduction
		- [x] test effect of extra dendrites => increases firing rate (due to more I_NaL provided?)
		- => spontaneous firing rate can be adapted by changing cm/gpas/gnaL. However by tuning these parameters you cannot get the firing as low as in the full model (9 Hz)
	
- Implement slow inactivation
	- [ ] get parameters state model slow inactivaton & recovery from papers (see notes below)
		
- See if other currents need to be adjusted to accomodate new current INa_rsg
	- e.g. reduce Ih/HCN or change its parameters

--------------------------------------------------------------------------------
# References/sources #

- Resurgent sodium current - model implementation & experiments
	- do ModelDB search for 'narsg'

- Cell reduction
	- Sterrat (2011) Ch. 4.3 - 4.4
	- Marasco, A., Limongiello, A., & Migliore, M. (2013). Using Strahler’s analysis to reduce up to 200-fold the run time of realistic neuron models. Scientific Reports, 3. https://doi.org/10.1038/srep02934
	- Encyclopedia of Computational Neuroscience p. 2598 (pdf 2650) - Reduced Morphology Models

--------------------------------------------------------------------------------
# Notes #

- comparison of Otsuka/RubinTerman/GilliesWillshaw model: 
	- see Encyclopedia of Computational Neuroscience, p. 2909 (pdf p. 2961)

## Na channel inactivation ##

- All components of Na current are susceptible to inactivation
- Option 1
	- use Raman & Bean model to implement transient + resurgent
	- use NaL to implement persistent as leak
	- PROBLEM
		- NaL has no inactivation
- Option 2
	- Use Taddese & Bean model to implement transient + persistent
	- Use Raman & Bean model to implement transient + resurgent
	- PROBLEM
		- have to use two separate mechanisms


- voltage dependence of resurgent Na current is similar in STN and Purkinje neurons
	- stated in [Do & Bean (2004)](Sodium currents in subthalamic nucleus neurons from Nav1.6-null mice.)


- Papers/model _slow inactivation_ of Na currents:

	- [Kuo & Bean (1994)](Na Channels Must Deactivate to Recover from Inactivation)
		- fig 7A shows first formulation of markov model that captures detailed interaction between activation and inactivation variables
		- [Raman & Bean (2001)](Inactivation and Recovery of Sodium Currents Neurons in Cerebellar Purkinje Neurons: Evidence for Two Mechanisms) mentions:
			- "the [Kuo & Bean](1994) Markov state model gives reasonable simulations of the voltage-dependence of development of inactivation and recovery from inactivation for sodium channels that do not give resurgent current"
	
	- [Do & Bean (2003)](Subthreshold Na currents and pacemaking of STN neurons: modulation by slow inactivation)
		- only experiments, no model or parameters
		- say in Discussion that mechanism of slow inactivation is same as in:
		- [Taddese & Bean (2002)](Subthreshold sodium current from rapidly inactivating sodium channels drives spontaneous firing of tuberomammillary neurons)
			- gating/Markov model and parameters in Taddese & Bean (2002) - Fig. 7A
	
	- [Akemann W, Knopfel T. (2006)](Interaction of Kv3 K channels & I_Na_rsg Influences the Rate of Spontaneous Firing of Purkinje Neurons)
		- _Transient_ and _Resurgent_ component of Na current implemented in two separate mechanisms
		- They both have the Markov scheme implementing slow inactivation of original Kuo & Bean paper



- Papers/model of _resurgent_ Na current (`I_Na_rsg`)
	
	- [Raman & Bean (2001)](Inactivation and Recovery of Sodium Currents Neurons in Cerebellar Purkinje Neurons: Evidence for Two Mechanisms)
		- ONLY models INa_rsg compoment of INa
			- i.e. you will still have to account of transient and persistent components
			- this is confirmed by plotting INa and INarsg in code supplied with Akemann and Knoepfel, J.Neurosci. 26 (2006) 4602
	
	- [Khaliq, Raman (2003)](The Contribution of Na_rsg to HF Firing in Purkinje Neurons) & [Akeman (2006)](Interaction of Kv3 K channels & I_Na_rsg Influences the Rate of Spontaneous Firing of Purkinje Neurons) models published on ModelDB
		- => see paper fig. 7, Khaliq & Raman model is tuned specifically for the resurgent current, doen NOT model slow inactivation