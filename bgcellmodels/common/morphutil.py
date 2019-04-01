from neuron import h
from neuron.rxd.morphology import parent, parent_loc
import json, io, re


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


def morphology_to_swc(sections, filename, encoding='utf-8'):
    """
    Convert any NEURON cell to SWC.

    Instead use:
    - built in exporters in ModelView (to NeuroML)
    - https://github.com/JustasB/hoc2swc
    - http://neuronland.org/NLMorphologyConverter/NLMorphologyConverter.html
    """
    raise NotImplementedError(morphology_to_swc.__doc__)


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

if __name__ == '__main__':
    test_json_export()