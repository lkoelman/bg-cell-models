Different solvers for the cable equation.

--------------------------------------------------------------------------------
# Plan

## TODO: write code that assembles matrix & does integration step

- see xmind notes Neuron + Arbor -> refer to source code

- first write simple implementation in Python

- then potentially write one in Rust/C/C++ that gets passed cfunc/ufunc/gufunc

## TODO: write Python mechanism class

Features:

- parse .mod files and convert into Python representation
	+ see [Arbor: modcc](https://github.com/eth-cscs/arbor/blob/master/modcc/modcc.cpp)
		* Possibly reuse `Module` and `Parser` object, then use it's state to generate own object instead of generating Cpp code
	+ see [CoreNEURON: mod2c](https://github.com/BlueBrain/mod2c)
	+ google terms "python domain specific language" "python parser"

- get steady-state values as tables

- get equations as Numba functions

## TODO: write Numba cfunc + integrator backend

See Numba documentation: [Creating C callbacks with @cfunc](http://numba.pydata.org/numba-doc/dev/user/cfunc.html).

## TODO: write Numba cuda backend

See Numba documentation: [writing CUDA kernels](http://numba.pydata.org/numba-doc/dev/cuda/kernels.html).

--------------------------------------------------------------------------------
# References

## Online Resources

Example simulators:

- [BRIAN algorithm notes](https://github.com/brian-team/brian/blob/master/brian/experimental/morphology/algorithms.txt)

- [J. Rieke's simple SciPy solver](https://github.com/jrieke/NeuroSim)


Reading out NEURON state in Python:

- https://github.com/WillemWybo/SGF_formalism

## Articles: Matrix Equations Branched Cable


+ [Hines (1984)](Efficient Computation of Branched Nerve Equations)

+ [Hines (1989)](A Program for Simulation of Nerve Equations with Branching Geometries)

+ [Mascagni (1991)](A Parallelizing Algortihm for Computing Solutions to Arbitrarily Branched Cable Neuron Models)

+ [Book Koch (1999) - The Biophysics of Computation](Appendix C - Sparse Matrix Methods for Modeling Single Neurons (p 487))

+ [Book Koch, Segev (1998) - Methods in Neural Modeling](Ch. 14 - Numerical Methods for Neuronal Modeling)