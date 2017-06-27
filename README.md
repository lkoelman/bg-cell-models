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
# Histology, Physiology, Connectivity

## Subcellular Connectivity

- Refs for calibrating synapses:
	- [Baufreton et al. 2009](Sparse but Selective and Potent Synaptic Transmission From the Globus Pallidus to the Subthalamic Nucleus)
	- [Atherton et al. 2013](Short-term Depression of External Globus Pallidus-Subthalamic Nucleus Synaptic Transmission and Implications for Patterning Subthalamic Activity)
		- set time constants for Tsodyks-Markram model


- Bevan (2017) in Handbook of BG Structure & Function

	- 70-80% of `GPe` neurons targer STN but each STN neuron is targeted by a small fraction of GPe neurons
	
	- Each `GPe-STN` axon synapses onto a number of STN neurons, on proximal somato-dendritic region

	- `GPe` axons synapse onto STN multisynaptically => they have large unitary conductance
	
	- sustained `GPe` input => profound STD (vesicle depletion)
	
	- `GPe` inputs have high rate in vivo => STD likely reflects influence of GPe on STN -> reduced probability of correlated GPe-STN activity

	- `M1, SMA, and pre-SMA` inputs are largely segregated such that M1 inputs are lateral to SMA inputs and pre-SMA inputsare ventral to SMA inputs.

	- `Cortical` inputs terminate on _small diameter dendrites_ and _dendritic spines_ of STN neurons (Bevan et al., 1995) and act through postsynaptic AMPARs and NMDARs with classical drug sensitivities, kinetics, and voltage dependencies (Chu et al., 2015).

	- `THA` inputs also synapse onto dendritic shafts and spines, but more proximal / larger-diam than CTX inputs

	- `PPN` inputs synapse onto dendritic shafts and spines (probably via Glu receptors)


- Baufreton (2009)

	- Juxtacellular labeling of single GP neurons in vivo and stereological estimation of the total number of GABAergic GP-STN synapses suggest that the GP-STN connection is surprisingly sparse: single GP neurons maximally contact only 2% of STN neurons and single STN neurons maximally receive input from 2% of GP neurons. However, GP-STN connectivity may be considerably more selective than even these estimates imply

	- Light and electron microscopic analyses revealed that single GP axons give rise to sparsely distributed terminal clusters, many of which correspond to multiple synapses with individual STN neurons. Application of the minimal stimulation technique in brain slices confirmed that STN neurons receive multisynaptic unitary inputs and that these inputs largely arise from different sets of GABAergic axons


- Fan (2012)

	- DA depletion profoundly increases the strength of GPe-STN inputs through an increase in the number of functional synapses (but no alteration in the number of GPe-STN axon terminals).

	- After DA depletion, there is an increase in the abundance and strength of synaptic contacts between GPe and STN.

## Plasticity

- Chu (2015)

	- LTP of motor cortex-STN transmission is balanced by hLTP of GPe-STN transmission: the ratio of excitation to inhibition pre- and post-induction of hLTP remains similar.

	- hLTP is associated with an increase in probability of GPe-STN transmission: increase in the frequency but not the amplitude of sIPSCs.

	- DAergic lesion (either by 6-OHDA or vehicle) increases frequency of GPe-STN mIPSCs. When NMDAR are knocked down (by cre-eGFP), frequency AND amplitude of these GPe-STN mIPSCs is reduced (independent of DA lesioning method). This indicates that hLTP occured after DAergic lesion, and that this is prevented when NMDAR are knocked out.

	-  After DA depletion, hLTP is occluded, impying that excessive motor cortical patterning of STN has already maximally triggered hLTP/augmented GPe-STN transmission.


## Electrophysiology

- Mallet (2008)

	- Single STN neurons fired bursts or single spikes in time with many, but not necessarily all, cycles of the beta oscillations in LFP (anaesthesized 6-OHDA rats). Coherence between single unit and LFP activity also peaked in the beta range. In other words there was phase-locking of individual unit activity to the oscillatory LFP cycle.


- Mallet (2012)

	- GPe 'Proto' neurons fire antiphase to STN neurons, and target downstream basal ganglia nuclei, including STN. GPe 'Arky' neurons fire in-phase with STN neurons, express preproenkephalin, and only innervate the striatum


- Galvan (2015)

	- After DA depletion, the mean firing rate of STN neurons is incresed (primates/rodents)

- Nambu (2014)

	- Average firing rate of STN neurons increased significantly from 19.8+/-9.7 to 27.5 +/-11.4 upon MPTP treatment


- Sanders (2016)

	- 6-OHDA rat: Increased M1-STN coherence in 5 frequency bands: theta, alpha, low beta, high beta, gamma
	- 6-OHDA rat: increased low frequency power (theta, alpha, low beta) in M1 and STN
	- 6-OHDA rat: decreased gamma power in M1 and STN
	- 6-OHDA rat: increased PAC in STN between 10-15 Hz and ~200 Hz bands


- Wilson, Bevan (2011)

	- The STN has three main firing patterns (in vivo):
		- 1) Regular single spiking pattern (12% cells of humans w/ essential tremor, 22% of anaesthesized rodents). 
		- 2) Irregular pattern w/ long ISI and sporadic doublets and triplets (52% cells of humans w/ essential tremor, 40-70% anaesthesized rodents). 
		- 3) bursting (irregular or rhytmic) (36% cells humans w/ tremor 3-30% anaestesized rodents. In PD the % bursting cells increases and bursts become stronger and more rhytmic.


## Physiology

- Bevan (2017) in Handbook of BG Structure & Function

	- Dopamine depolarizes STN neurons and elevates their autonomous activity ex vivo through activation of post- synaptic Gs-coupled D1-like and Gi-coupled D2-like postsynaptic/extrasynaptic receptors. 

	- D1-like receptor activation is believed to activate a cyclic nucleotide-gated cation channel (Loucif et al., 2008). 

	- D2-like receptor activation has two actions, both of which result in reduced whole-cell K+ conductance: 
		
		- (1.) D2-like (D2/D3) receptor activation reduces Cav2.2 channel conductance through direct binding of Gβγ subunits, which reduces Ca2+-dependent activation of functionally coupled SKCa channels (Ramanathan et al., 2008)

		- (2.) D2-like receptor activation also leads to a reduction in the conductance of voltage-independent K+ channels (Zhu et al., 2002b).

		- Dopamine also acts through presynaptic Gi-coupled D2-like recep- tors to reduce the initial probability of GABAergic GPe and glutamatergic transmission in the STN (Baufreton and Bevan, 2008; Shen and Johnson, 2000).

	- [D5 receptor] constitutive and agonist-stimulated D5 receptor activation potentiates Cav1 channel conductance in STN neurons and thus promotes the generation of action potential bursts triggered from hyperpolarized voltages (Baufreton et al., 2003; Chetrit et al., 2013)


## Pathophysiology

- Baufreton (2005)

	- In the DA depleted state (experimental PD), GABA IPSCs and AMPA EPSCs are amplified in STN compared to the normal state.

	- In the normal state, GABA and glutamate release in STN are suppressed by presynaptic D2 DA receptors (smaller IPSCs and EPSCs).


- Tachibana (2011)

	- Inactivation/silencing of STN with GABA-R agonist ameliorated PD motor signs and decreased 8-15 Hz oscillations and firing rate in STN and GPi
	
	- Blockade of glutamergic inputs (with AMPA and NMDA antagonist) to STN with NMDA and AMPA antagonists suppressed the 8-15 Hz oscillations but increased burst activities (no info about behavioral changes given)

	- Blockade of GABAergic inputs to STN from GPe attenuated 8-15 Hz oscillations and increased the firing rate, but did not induce clear behavioral changes.


- Chiken, Nambu (2016)

	- STN-DBS generated both excitatory and inhibitory postsynaptic potentials in STN neurons through activation of both glutamatergic and GABAergic afferents (Lee and others 2004).

--------------------------------------------------------------------------------
# STN Models

- Cell reduction
	
	- Sterrat (2011) Ch. 4.3 - 4.4
	
	- Marasco, A., Limongiello, A., & Migliore, M. (2013). Using Strahler’s analysis to reduce up to 200-fold the run time of realistic neuron models. Scientific Reports, 3. https://doi.org/10.1038/srep02934
	
	- Encyclopedia of Computational Neuroscience p. 2598 (pdf 2650) - Reduced Morphology Models


- comparison of Otsuka/RubinTerman/GilliesWillshaw model:

	- see Encyclopedia of Computational Neuroscience, p. 2909 (pdf p. 2961)

--------------------------------------------------------------------------------
# Na channel models

## References ##

- Resurgent sodium current - model implementation & experiments
	- do ModelDB search for 'narsg'

## Na channel slow inactivation ##

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