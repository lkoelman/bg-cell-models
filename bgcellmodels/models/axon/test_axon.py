
# Test aligning axons
from pynrnfem.axon.mcintyre2002 import AxonMcintyre2002

cortico_spinal_tracts = '/home/luye/workspace/fem-neuron-interface/test_data/Yeh2018_CST_left.trk'

axon_builder = AxonMcintyre2002()
axons = align_axons_tractogram(axon_builder, cortico_spinal_tracts,
                                streamline_ids=[0])