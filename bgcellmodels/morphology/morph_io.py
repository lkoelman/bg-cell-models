"""
Reading and writing of neuron morphologies.


Notes
-----

Additional libraries for working with NEURON morphologies:

- PyNeuron-Toolbox: https://github.com/ahwillia/PyNeuron-Toolbox
- btmorph :         https://btmorph.readthedocs.io/en/latest/index.html#
- NeuroM :          https://github.com/BlueBrain/NeuroM
- AllenSDK :        https://github.com/AllenInstitute/AllenSDK
- Hoc2Swc :         https://github.com/JustasB/hoc2swc/
- NeuroMorphoVis:   https://github.com/BlueBrain/NeuroMorphoVis
"""

from neuron import h
from bgcellmodels.common.treeutils import parent, parent_loc
import json, io, re
import numpy as np

from . import morph_3d


def morphology_to_dict(sections):
    """
    Extract morphology info from given sections.

    @return     list(dict()) containing one dict for each section:
                the dict contains its morphological & topological information

    @note       code modified from R.A. McDougal's post at
                https://www.neuron.yale.edu/phpBB/viewtopic.php?f=2&t=3478&p=14758
    """

    # Assign index to each sections (for expressing topology relations)
    section_map = {sec: i for i, sec in enumerate(sections)}
    
    # adds 3D info using simple algorithm if not present
    h.define_shape()
    
    result = []
    for sec in sections:
        parent_sec = parent(sec)

        parent_x = -1 if parent_sec is None else parent_loc(sec, parent_sec)
        parent_id = -1 if parent_sec is None else section_map[parent_sec] # get parent index
        
        n3d = int(h.n3d(sec=sec))
        
        result.append({
            'section_orientation':  h.section_orientation(sec=sec),
            'parent':               parent_id,
            'parent_loc':           parent_x,
            'x':                    [h.x3d(i, sec=sec) for i in xrange(n3d)],
            'y':                    [h.y3d(i, sec=sec) for i in xrange(n3d)],
            'z':                    [h.z3d(i, sec=sec) for i in xrange(n3d)],
            'diam':                 [h.diam3d(i, sec=sec) for i in xrange(n3d)],
            'name':                 sec.hname()           
        })
    
    return result


def morphology_to_SWC(sections, filename):
    """
    Convert all instantiated NEURON cells to SWC. 

    Each isolated tree is written to a separate .swc file.

    Instead use:
    - built in exporters in ModelView (to NeuroML)
    - https://github.com/JustasB/hoc2swc
    - http://neuronland.org/NLMorphologyConverter/NLMorphologyConverter.html
    """
    from hoc2swc import neuron2swc # this function exports all loaded cells to SWC
    neuron2swc(filename)


def uniform_morphology_to_SWC(sections, filename, encoding='ascii'):
    """
    Convert sections with uniform diameter to SWC file.

    Written for illustrative purposes.

    @pre    assumes 3d location info is stored from 0 to 1-end of Section

    @note   see SWC file specification at 
            http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    """
    morph_dicts = morphology_to_dict(sections)

    with io.open(filename, 'w', encoding=encoding) as outfile:
        for sec_id, sd in enumerate(morph_dicts):

            sec = sections[sec_id]
            parent_sec = parent(sec)
            parent_x = -1 if parent_sec is None else parent_loc(sec, parent_sec)
            orientation = sd['section_orientation']

            # Check if section has uniform diameter
            diams = sd['diam']
            if not all((d==diams[0] for d in diams)):
                ValueError("Encountered Section with non-uniform diameter: {}".format(sec))

            # Determine parent sample from connection point
            if parent_x == 1:
                parent_id = 2*sd['parent'] + 2
            elif parent_x == 0:
                parent_id = 2*sd['parent'] + 1
            elif parent_x == -1:
                parent_id = -1
            else:
                ValueError("Encountered non-terminal connection at Section {}.".format(sec))

            # Sample at start of Section
            sample_1 = {
                'segment_id': 2*sec_id + 1,
                'region_id': 0,
                'radius': sd['diam'][0] / 2.0,
                'x': sd['x'][0],
                'y': sd['y'][0],
                'z': sd['z'][0],
            }

            # Sample at end of Section
            sample_2 = {
                'segment_id': 2*sec_id + 2,
                'region_id': 0,
                'radius': sd['diam'][-1] / 2.0,
                'x': sd['x'][-1],
                'y': sd['y'][-1],
                'z': sd['z'][-1],
            }

            # Determine wich sample connects to parent
            if orientation == 0:
                sample_1['parent_id'] = parent_id
                sample_2['parent_id'] = sample_1['segment_id']
            elif orientation == 1:
                sample_2['parent_id'] = parent_id
                sample_1['parent_id'] = sample_2['segment_id']
            else:
                raise ValueError("Illegal orientation value {}".format(orientation))

            
            # Write samples to file         
            outfile.write(u"{segment_id:d} {region_id:d} {x:f} {y:f} {z:f} {radius:f} {parent_id:d}\n".format(**sample_1))

            outfile.write(u"{segment_id:d} {region_id:d} {x:f} {y:f} {z:f} {radius:f} {parent_id:d}\n".format(**sample_2))


def read_SWC_samples(file_path):
    """
    Read samples in SWC file.

    @return     samples : list[list[<7 sample elements>]]

                A sample consists of 7 elements, i.e. [sample_number (int), 
                structure_identifier (int), x (float), y (float), z (float),
                radius (float), parent_sample_number (int)]

    """
    sample_data_types = [int, int, float, float, float, float, int]
    samples = []

    with open(file_path, 'r') as swc_file:
        for line in swc_file:
            if line.startswith('#'):
                continue

            sample_data = line.split() # default using whitespace
            sample_parsed = [
                value_type(sample_data[i]) for i, value_type in enumerate(
                    sample_data_types)
            ]
            samples.append(sample_parsed)
            # sample_idx, sample_type, x, y, z, radius, parent_idx = sample_parsed

    return samples


def write_SWC_samples(samples, file_path, comment=None):
    """
    Write samples to SWC file.
    """
    sample_format = "{:.0f} {:.0f} {:f} {:f} {:f} {:f} {:.0f}\n"
    with open(file_path, 'w') as file:
        if comment:
            file.writelines(
                ['# ' + comment + '\n' for line in comment.splitlines()])
        for sample in samples:
            file.write(sample_format.format(*sample))


def save_json(sections, filename, encoding='utf-8'):
    """
    Save morphology to JSON
    """
    morph_dicts = morphology_to_dict(sections)
    
    # json_string = json.dumps(morph_dict, indent=2)

    if encoding == 'ascii':
        with open(filename, 'w') as f:
            json.dump(morph_dicts, f)

    else:
        with io.open(filename, 'w', encoding=encoding) as f:
            f.write(json.dumps(morph_dicts, ensure_ascii=False))


def load_json(morphfile):
    """
    Load morphology from JSON.
    """

    with open(morphfile, 'r') as f:
     secdata = json.load(f)

     seclist = []
     for sd in secdata:
        # make section
        sec = h.Section(name=sd['name'])
        seclist.append(sec)

        # make 3d morphology
        for x,y,z,d in zip(sd['x'], sd['y'], sd['z'], sd('diam')):
           h.pt3dadd(x, y, z, d, sec=sec)

           # connect children to parent compartments
           for sec,sd in zip(seclist,secdata):
              if sd['parent_loc'] >= 0:
               parent_sec = seclist[sd['parent']] # not parent_loc, grab from sec_list not sec 
               sec.connect(parent_sec(sd['parent_loc']), sd['section_orientation'])

               return seclist


def test_json_export():
    """
    Text exporting a simple example morphology to JSON format
    """

    s = [h.Section(name='s[%d]' % i) for i in xrange(13)]

    """
        Create the tree
       
              s0
        s1    s2         s3
        s4           s5      s6
        s7         s8 s9       s10
    """
    for p, c in [[0, 1], [0, 2], [0, 3], [1, 4], [4, 7], [3, 5], [3, 6], [5, 8], [5, 9], [6, 10]]:
        s[c].connect(s[p])
   
    print json.dumps(morphology_to_dict([s[3], s[5], s[8], s[0], s[1], s[4], s[7]]), indent=2)


def prepare_hoc_morphology_for_SWC_export(hoc_script_path, hoc_out_path):
    """
    Replace expressions in brackets (not containing variables) by their
    evaluated values.

    Usage
    -----

    - first replace all variable names in 'connect' 'access' and 'pt3dadd' statements
    - ensure there is only one statement per line
        - split 'create A, B, C' into separate statements
    """
    hoc_clean_morph = ''
    def match_evaluator(match):
        """ process a regex match and return the corrected line """
        assert match.lastindex == 1 # only one match
        expr = match.group(0)
        repl = str(eval(expr))
        print("{}\n>>>>>>\n{}".format(expr, repl))
        return repl

    pattern = r"\[([\d\*\+]+)\]" # expression within brackets e.g. [i*j+2]

    with open(hoc_script_path, 'r') as hoc_script:
        hoc_dirty_morph = hoc_script.read()

    hoc_clean_morph = re.sub(pattern, match_evaluator, hoc_dirty_morph)

    with open(hoc_out_path, 'w') as out_script:
        out_script.write(hoc_clean_morph)


def morphology_to_STEP_1D(secs, filepath):
    """
    Write morphology as one-dimensional STEP file.

    @param  secs : neuron.SectionList

            SectionList containing sections whose geometry will be exported. 
            It is important that this is a SectionList so that the CAS will be
            set correctly during iterations.
    """
    if not filepath.endswith('.stp'):
        filepath += '.stp'

    # library:      github.com/tpaviot/pythonocc-core
    # wrapper API:  api.pythonocc.org/py-modindex.html
    # c++ API:      www.opencascade.com/doc/occt-7.1.0/refman/html/toolkit_tkmath.html
    from OCC.gp import gp_Pnt # gp_Lin, gp_Ax1, gp_Dir # core geometry types
    from OCC.TColgp import TColgp_Array1OfPnt # collections
    from OCC.GeomAPI import GeomAPI_PointsToBSpline # geometry types
    from OCC.GeomAbs import GeomAbs_C0
    from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeEdge # geometry to shape
    from OCC.TopoDS import TopoDS_Compound, TopoDS_Builder # Compound shapes
    from OCC.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.IFSelect import IFSelect_RetDone # Return codes
    from OCC.Interface import Interface_Static_SetCVal
    
    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP203")
    c_edges = TopoDS_Compound() # group edges in a compound shape
    c_builder = TopoDS_Builder()
    c_builder.MakeCompound(c_edges)

    # Build polyline/curve using 3D location info
    for sec in secs:
        # Copy points
        num_verts = int(h.n3d(sec=sec))
        pts = TColgp_Array1OfPnt(1, num_verts)
        for i in xrange(num_verts):
            pts.SetValue(i+1,
                gp_Pnt(h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)))

        # Build curve
        crv_builder = GeomAPI_PointsToBSpline(pts, 1, 1, GeomAbs_C0, 1e-3) # No continuity, max degree 1
        crv = crv_builder.Curve() # this is a Handle/reference
        edge = BRepBuilderAPI_MakeEdge(crv).Edge()

        # Add to compound
        # step_writer.Transfer(edge, STEPControl_AsIs) # accepts TopoDS_Shape
        c_builder.Add(c_edges, edge)

    # Write CAD geometry to STEP file
    step_writer.Transfer(c_edges, STEPControl_AsIs) # accepts TopoDS_Shape
    status = step_writer.Write(filepath)
    if status != IFSelect_RetDone:
        raise Exception("Failed to write STEP file")


def morphologies_to_edges(section_lists, segment_centers=True,
                         scale=1.0, translation=None, transform=None,
                         flatten_cells=True):
    """
    Convert neuron morphologies to lists of vertices and edges.

    Each segment is represented by a degenerate face, i.e. a triangle
    with vertices [a b a].

    @pre    requires package plyfile, e.g. `pip install plyfile`

    @see    morphology_to_PLY()

    @return (vertices, edges) : tuple
            
            - If flatten_cells is True, vertices is an Nx3 numpy array containing
            all the compartment coordinates, and edges are pairs of indices
            into this array.
            
            - If flatten_cells is False, vertices is a list of Nx3 arrays, 
            one for each cell, and edges a list of index matrices into the 
            vertex array for each cell.
    """

    # Get 3D samples
    if segment_centers:
        samples_xyz, secs_num3d = morph_3d.get_segment_centers(section_lists, 
                                        samples_as_rows=True)
    else:
        samples_xyz, secs_num3d = morph_3d.get_section_samples(section_lists,
                                        include_diam=False)

    # Apply transformation before writing
    if (translation is not None) or (transform is not None) or (scale != 1.0):
        samples_mat = np.ones((len(samples_xyz), 4))
        samples_mat[:,:3] = samples_xyz
        A = np.array(transform) if transform else np.eye(4)
        if (translation is not None):
            A[:-1, 3] += translation
        samples_mat = np.dot(samples_mat, A.T)
        if scale != 1.0:
            samples_mat *= scale
        samples_xyz = samples_mat[:, :3]

    # Create vertices
    if flatten_cells:
        vertices = samples_xyz  # flat vertex list
    else:
        vertices = []           # vertex list per cell
    
    # Create edges
    secs_edges = []
    sample_offset = 0
    for i_sec, num_samples in enumerate(secs_num3d):

        # Add edge [a, b]
        if flatten_cells:
            edge_offset = sample_offset
        else:
            edge_offset = 0
        edges = [
            (i, i+1) for i in xrange(edge_offset, edge_offset + num_samples - 1)
        ]
        if flatten_cells:
            secs_edges.extend(edges)
        else:
            secs_edges.append(edges)
            vertices.append(samples_xyz[sample_offset:num_samples, :])

        sample_offset += num_samples

    # Correct return datatypes
    if flatten_cells:
        edges = np.array(secs_edges)
    else:
        edges = secs_edges
    return vertices, edges


def morphology_to_PLY(section_lists, filepath, segment_centers=True,
                      scale=1.0, rgb=(0.0, 0.0, 0.0), text=False,
                      translation=None, transform=None,
                      make_edges=True, make_faces=False):
    """
    Write neuron morphology to PLY file using degenerate faces.

    Each segment is represented by a degenerate face, i.e. a triangle
    with vertices [a b a].

    @pre    requires package plyfile, e.g. `pip install plyfile`

    @param  segment_centers : bool
            If true, write 3D locations of segment centers (nodes, i.e. centers
            of simulated compartments). This is useful for knowing the locations
            of compartments, and their current/voltage sources in 3D space.

    @param  scale : float
            Scale factor applied to coordinates after transform and translation
            is applied. Translation and transform not affected by scale.
    """
    from plyfile import PlyData, PlyElement

    # Get 3D samples
    if segment_centers:
        samples_xyz, secs_num3d = morph_3d.get_segment_centers(section_lists, 
                                        samples_as_rows=True)
    else:
        samples_xyz, secs_num3d = morph_3d.get_section_samples(section_lists,
                                        include_diam=False)

    # Apply transformation before writing
    if (translation is not None) or (transform is not None) or (scale != 1.0):
        samples_mat = np.ones((len(samples_xyz), 4))
        samples_mat[:,:3] = samples_xyz
        A = np.array(transform) if transform else np.eye(4)
        if (translation is not None):
            A[:-1, 3] += translation
        samples_mat = np.dot(samples_mat, A.T)
        if scale != 1.0:
            samples_mat *= scale
        samples_xyz = [tuple(row[:3]) for row in samples_mat]

    # Create vertices as PLY elements
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = np.array(samples_xyz, dtype=vertex_dtype) # argument must be list of tuple
    verts_element = PlyElement.describe(vertices, 'vertex')
    
    # Create all faces and edges
    secs_faces = []
    secs_edges = []

    sample_offset = 0
    for i_sec, num_samples in enumerate(secs_num3d):

        # Face is degenerate face [a, b, a]
        faces = [
            ([i, i+1, i], rgb[0], rgb[1], rgb[2])
                for i in xrange(sample_offset, sample_offset + num_samples - 1)
        ]
        secs_faces.extend(faces)

        # Add edge [a, b]
        edges = [
            (i, i+1, rgb[0], rgb[1], rgb[2])
                for i in xrange(sample_offset, sample_offset + num_samples - 1)
        ]
        secs_edges.extend(edges)

        sample_offset += num_samples


    # Concatenate all faces
    face_dtype = [
        ('vertex_indices', 'i4', (3,)), 
        ('red',   'u1'),
        ('green', 'u1'),
        ('blue',  'u1')
    ]
    faces = np.array(secs_faces, dtype=face_dtype)
    faces_element = PlyElement.describe(faces, 'face')

    # Concatenate all edges
    edge_dtype = [ # edge format: http://paulbourke.net/dataformats/ply/
        ('vertex1', 'i4'),
        ('vertex2', 'i4'), 
        ('red',   'u1'),
        ('green', 'u1'),
        ('blue',  'u1')
    ]
    edges = np.array(secs_edges, dtype=edge_dtype)
    edges_element = PlyElement.describe(edges, 'edge')

    # Write vertices and faces to PLY file
    elements = [verts_element]
    if make_faces:
        elements.append(faces_element)
    if make_edges:
        elements.append(edges_element)
    PlyData(elements, text=text).write(filepath)


def morphology_to_TXT(section_lists, filepath, segment_centers=True,
                      scale=1.0, rgb=(0.0, 0.0, 0.0),
                      translation=None, transform=None, precision_mm=1e-6):
    """
    Write neuron morphology to PLY file using degenerate faces.

    Each segment is represented by a degenerate face, i.e. a triangle
    with vertices [a b a].

    @pre    requires package plyfile, e.g. `pip install plyfile`

    @param  segment_centers : bool
            If true, write 3D locations of segment centers (nodes, i.e. centers
            of simulated compartments). This is useful for knowing the locations
            of compartments, and their current/voltage sources in 3D space.

    @param  scale : float
            Scale factor applied to coordinates after transform and translation
            is applied. Translation and transform not affected by scale.
    """

    # Get 3D samples
    if segment_centers:
        samples_xyz, secs_num3d = morph_3d.get_segment_centers(section_lists, 
                                        samples_as_rows=True)
    else:
        samples_xyz, secs_num3d = morph_3d.get_section_samples(section_lists,
                                        include_diam=False)

    # Apply transformation before writing
    if (translation is not None) or (transform is not None) or (scale != 1.0):
        samples_mat = np.ones((len(samples_xyz), 4))
        samples_mat[:,:3] = samples_xyz
        A = np.array(transform) if transform else np.eye(4)
        if (translation is not None):
            A[:-1, 3] += translation
        samples_mat = np.dot(samples_mat, A.T)
        if scale != 1.0:
            samples_mat *= scale
        samples_xyz = samples_mat[:, :3]
    else:
        samples_xyz = np.array(samples_xyz)

    # Set precision to 0.001 um
    num_significant = int(-np.log10(precision_mm) - 3 - np.log10(scale))
    fmt = '%.{:d}e'.format(num_significant)

    # Save to text file
    np.savetxt(filepath, samples_xyz, fmt=fmt)


if __name__ == '__main__':
    test_json_export()