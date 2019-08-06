
import json, collections
from bgcellmodels.common.fileutils import VariableIndentEncoder, NoIndent

with open('dummy-cells_axons-full_CST1.json', 'r') as f:
    config = json.load(f, object_pairs_hook=collections.OrderedDict)

# Modifications

## Mod.A : add random rotations to cells
# import numpy as np
# import transforms3d
# for cell_config in config["cells"]:
#     if cell_config["population"] == "GPE":
#         A = np.array(cell_config["transform"])
#         alpha, beta = np.random.random_sample(2) * np.pi
#         z_angle = alpha
#         x_angle = beta * 2
#         R3x3 = transforms3d.euler.euler2mat(z_angle, x_angle, 0, axes='rzxy')
#         R4x4 = np.eye(4)
#         R4x4[:-1,:-1] = R3x3
#         cell_config["transform"] = [list(row) for row in np.dot(A, R4x4)]

## Mod.B : change axon definitions
to_remove = []
for axon_config in config["connections"]:

    # mark axons for removal
    if axon_config["projection"] == "CTX-STN":
        to_remove.append(axon_config)
        # ax_blender_index = int(axon_config["axon"].split(".")[-1])
    
    # delete unused entries
    unused_keys = "pre_gid", "post_gids"
    for k in unused_keys:
        if k in axon_config:
            del axon_config[k]
            
for axon_config in to_remove:
    config["connections"].remove(axon_config)

## Mod.C : append new axon defnitions
for i in range(48):
    config["connections"].append({
        "axon": "axon.CST.drawn-MouseLight.{:03d}".format(i),
        "projection": "CTX-STN"
    })

# Fix indentation in JSON
for cell_config in config["cells"]:
    cell_config["transform"] = [NoIndent(row) for row in cell_config["transform"]]
for axon_config in config["connections"]:
    if "post_gids" in axon_config:
        axon_config["post_gids"] = NoIndent(axon_config["post_gids"])


# Write updated config
string = json.dumps(config, f, cls=VariableIndentEncoder, indent=2, sort_keys=False)
out_file = 'dummy-cells_axons-full_CST2.json'

with open(out_file, 'w') as f:
    # json.dump(config, f, indent=2, sort_keys=False)
    f.write(string)

print("Wrote updated cell config to " + out_file)


