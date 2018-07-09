from neuron import h
from neuron.rxd.morphology import parent, parent_loc
import json, io


def morphology_to_dict(sections):
    """
    Extract morphology info from given sections

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


def morphology_to_swc(sections, filename, encoding='utf-8'):
    """
    Convert any NEURON cell to SWC.
    """

    # orientation determines which sample point get a parent_id in another section
    # parent_loc determines the parent_id: which segment to refer to in other section

    raise NotImplementedError("Not yet implemented for non-uniform diameters over sections")


def uniform_to_swc(sections, filename, encoding='ascii'):
    """
    Convert sections with uniform diameter to SWC file.

    @note   see SWC file specification at 
            http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
    
    @pre    assumes 3d location info is stored from 0 to 1-end of Section
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


if __name__ == '__main__':
    test_json_export()