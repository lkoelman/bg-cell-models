# Changes to source code

Changes with respect to original source code:

- file `kca2.mod`: change line `USEION caL READ icaL` to `USEION caL READ icaL VALENCE 2` as in `L_Ca.mod` to prevent NEURON 7.5 from throwing error