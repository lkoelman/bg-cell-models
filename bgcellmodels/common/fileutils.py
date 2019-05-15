# -*- coding: utf-8 -*-

"""
File handling utilities.
"""

import re
import json
import collections
from _ctypes import PyObj_FromPtr


def parse_json_file(filename, nonstrict=True, ordered=False):
    """
    Parse JSON file to dict.

    @param      nonstrict: bool

                If True, the json file can contain non strictly valid JSON
                including comments and extraneous commas. For the full list
                of allowed deviations, see function `validate_minify_json()`
    """
    with open(filename, 'r') as json_file:
        json_string = json_file.read()
        return parse_json_string(
                    json_string, nonstrict=nonstrict, ordered=ordered)


def parse_json_string(string, nonstrict=True, ordered=False):
    """
    Parse JSON string to dict.

    @param      nonstrict: bool

                If True, the json string can contain non strictly valid JSON
                including comments and extraneous commas. For the full list
                of allowed deviations, see function `validate_minify_json()`
    """
    if nonstrict:
        string = validate_minify_json(string)
    if ordered:
        object_pairs_hook = collections.OrderedDict
    else:
        object_pairs_hook = None
    return json.loads(string, object_pairs_hook=object_pairs_hook)


def load_json_nonstrict(filename):
    """
    (DEPRECATED)

    Same as json.load(filename) except the json file can contain
    non strictly valid JSON.

    For the things that are allowed in non-strict JSON, see function
    `validate_minify_json`.
    """
    return parse_json_file(filename, nonstrict=True)


def validate_minify_json(string):
    """
    Processes a JSON-like string into a valid, minified JSON string.

    Supports the following enhancements to JSON:
        - Trailing commas in lists/dicts: [1, 2, 3,]
        - Single-line Python/shell-style comments: # my comment
        - Single-line C/JavaScript-style comments: // my comment
        - Multi-line/in-line C/JavaScript-style comments: /* my comment */

    Examples:
      Input:  [1, 2, 3,]
      Output: [1,2,3]
      Input:  [1, 2, /* comment */ 3]
      Output: [1,2,3]
      Input:  [1, 2, 3] // comment
      Output: [1,2,3]

    @param      str
                A JSON or JSON-like string to process.

    @return     str
                Valid, minified JSON.

    @author     dana geier
    @email      dana@dana.is
    @url        https://github.com/okdana/jsonesque/
    @license    MIT
    """
    if not string:
        return ''

    round_one   = re.compile(r'"|(/\*)|(\*/)|(//)|#|\n|\r')
    round_two   = re.compile(r'"|,')
    end_slashes = re.compile(r'(\\)*$')

    # We add a new-line here so that trailing comments get caught at the end of
    # the string (e.g., '[1, 2, 3] // foo') — it'll be stripped out later
    string     += "\n"
    new_string  = ''
    length      = len(string)
    index       = 0

    in_string         = False
    in_comment_multi  = False
    in_comment_single = False

    # First round — remove comments
    for match in re.finditer(round_one, string):
        # Append everything up to the match, stripping white-space along the way
        if not (in_comment_multi or in_comment_single):
            tmp = string[index:match.start()]

            if not in_string:
                tmp = re.sub('[ \t\n\r]+', '', tmp)

            new_string += tmp

        index = match.end()
        val   = match.group()

        # Handle strings
        if val == '"' and not (in_comment_multi or in_comment_single):
            escaped = end_slashes.search(string, 0, match.start())

            # Start of string or un-escaped " character to end string
            if not in_string or escaped is None or len(escaped.group()) % 2 == 0:
                in_string = not in_string
            # Include " character in next iteration
            index -= 1

        # Handle comment beginnings and trailing commas
        elif not (in_string or in_comment_multi or in_comment_single):
            if val == '/*':
                in_comment_multi = True
            elif val == '//' or val == '#':
                in_comment_single = True

        # Handle multi-line comment endings
        elif val == '*/' and in_comment_multi and not (in_string or in_comment_single):
            in_comment_multi = False
            while index < length and string[index] in ' \t\n\r':
                index += 1

        # Handle single-line comment endings
        elif val in '\n\r' and in_comment_single and not (in_string or in_comment_multi):
            in_comment_single = False

        # Anything else — just append
        elif not (in_comment_multi or in_comment_single):
            new_string += val

    new_string += string[index:]
    string      = new_string
    new_string  = ''
    length      = len(string)
    index       = 0
    in_string   = False

    # Second round — remove trailing commas
    # @todo There's a more performant way to remove these, just too lazy rn
    for match in re.finditer(round_two, string):
        # Append everything up to the match
        new_string += string[index:match.start()]

        index = match.end()
        val   = match.group()

        # Handle strings
        if val == '"':
            escaped = end_slashes.search(string, 0, match.start())

            # Start of string or un-escaped " character to end string
            if not in_string or escaped is None or len(escaped.group()) % 2 == 0:
                in_string = not in_string
            # Include " character in next iteration
            index -= 1

        # Handle commas
        elif val == ',':
            if string[index] not in ']}':
                new_string += val

        # Anything else — just append
        else:
            new_string += val

    new_string += string[index:]
    return new_string


class NoIndent(object):
    """
    Wrapper for lists/dicts that should not be indented in JSON representation.
    """
    def __init__(self, value):
        self.value = value


class VariableIndentEncoder(json.JSONEncoder):
    """
    JSON encoder that lets you combine indentation with 'flat' (not indented)
    representations of objects.

    This makes the JSON file more readable for example when writing matrices
    as nested attributes.

    Credit to StackOverflow user martineau, https://stackoverflow.com/a/13252112

    EXAMPLE
    -------

    >>> properties = {
    >>>     'flat_dict': NoIndent([{"x":1,"y":7}, {"x":0,"y":4}, {"x":5,"y":3}]),
    >>>     'flat_list': NoIndent([[1,2,3],[2,3,1],[3,2,1]])
    >>> }
    >>>
    >>> json.dumps(properties, cls=VariableIndentEncoder indent=2, sort_keys=False)

    """
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(VariableIndentEncoder, self).__init__(**kwargs)


    def default(self, obj):
        # The default JSON representation of a NoIndent object is its id (memory address)
        if isinstance(obj, NoIndent):
            return self.FORMAT_SPEC.format(id(obj))
        else:
            return super(VariableIndentEncoder, self).default(obj)


    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(VariableIndentEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # Get id (memory address) of wrapper object and retrieve it
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr
