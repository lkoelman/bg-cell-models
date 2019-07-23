
import json, collections
import numpy as np
import transforms3d
from bgcellmodels.common.fileutils import VariableIndentEncoder, NoIndent

with open('dummy-cells_axons-full.json', 'r') as f:
    config = json.load(f, object_pairs_hook=collections.OrderedDict)

for cell_config in config["cells"]:
    if cell_config["population"] == "GPE":
        A = np.array(cell_config["transform"])
        alpha, beta = np.random.random_sample(2) * np.pi
        z_angle = alpha
        x_angle = beta * 2
        R3x3 = transforms3d.euler.euler2mat(z_angle, x_angle, 0, axes='rzxy')
        R4x4 = np.eye(4)
        R4x4[:-1,:-1] = R3x3
        cell_config["transform"] = [NoIndent(list(row)) for row in np.dot(A, R4x4)]
        # cell_config["transform"] = [list(row) for row in np.dot(A, R4x4)]
    else:
        cell_config["transform"] = [NoIndent(row) for row in cell_config["transform"]]

for proj_config in config["connections"]:
    proj_config["post_gids"] = NoIndent(proj_config["post_gids"])


string = json.dumps(config, f, cls=VariableIndentEncoder, indent=2, sort_keys=False)
with open('dummy-cells_axons-full_angles-rand.json', 'w') as f:
    # json.dump(config, f, indent=2, sort_keys=False)
    f.write(string)


