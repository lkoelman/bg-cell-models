
import json, collections
from bgcellmodels.common.fileutils import VariableIndentEncoder, NoIndent

with open('dummy-cells_axons-full_angles-rand.json', 'r') as f:
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
    if axon_config["projection"] == "CTX-STN":
        ax_index = int(axon_config["axon"].split(".")[-1])
        if ax_index <= 118:
            axon_config["axon"] = "axon.CST.from-GPe." + str(ax_index)
        else:
            to_remove.append(axon_config)

for axon_config in to_remove:
    config["connections"].remove(axon_config)

# Fix indentation in JSON
for cell_config in config["cells"]:
    cell_config["transform"] = [NoIndent(row) for row in cell_config["transform"]]

for axon_config in config["connections"]:
    axon_config["post_gids"] = NoIndent(axon_config["post_gids"])


# Write updated config
string = json.dumps(config, f, cls=VariableIndentEncoder, indent=2, sort_keys=False)
out_file = 'dummy-cells_axons-full_CST1.json'

with open(out_file, 'w') as f:
    # json.dump(config, f, indent=2, sort_keys=False)
    f.write(string)

print("Wrote updated cell config to " + out_file)


