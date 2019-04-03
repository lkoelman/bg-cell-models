"""
Geometrical operations on neuron morphologies.

@author     Lucas Koelman
@date       26/11/2018


Notes
-----

Additional libraries to deal with NEURON morphologies:

- PyNeuron-Toolbox:https://github.com/ahwillia/PyNeuron-Toolbox
- btmorph : https://btmorph.readthedocs.io/en/latest/index.html#
- NeuroM : https://github.com/BlueBrain/NeuroM
- AllenSDK : https://github.com/AllenInstitute/AllenSDK
"""

from neuron import h
import numpy as np

def transform_sections(secs, A):
    """
    Apply transformation to sections
    
    @param  secs : list[neuron.Section]
            List of NEURON sections.

    @param  A : np.array
            4x4 transformation matrix in column-major layout
    """
    for sec in secs:
        # Construct matrix with vertices as rows
        num_verts = int(h.n3d(sec=sec))
        src_verts = np.array([
            [h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec), 1.0] 
                for i in xrange(num_verts)])

        # Transform vertex matrix
        new_verts = np.dot(src_verts, A.T)

        # Update 3D info
        for i in xrange(num_verts):
            diam = h.diam3d(i, sec=sec)
            h.pt3dchange(i, new_verts[i,0], new_verts[i,1], new_verts[i,2], diam, sec=sec)
