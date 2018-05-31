class dotdict(dict):
    """
    dot.notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Bunch(object):
    """
    Bunch or struct-like object for data storage using dot syntax
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    same as Python >= 3.5 math.isclose
    """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def getdictvals(d, *args, **kwargs):
    """
    Get dictionary values at given keys as tuple.

    @param      *args : *list(str)
                Expanded list of keys of arbitrary length

    @param      as_dict: bool (keyword argument)
                Return result as dict instead of tuple
    
    @return     vals : tuple
                Value associated with given keys
    """
    if kwargs.get("as_dict", False):
        return {k: d[k] for k in args}
    else:
        return tuple((d[k] for k in args))


def eval_context(**context):
    """
    Evaluate statement in given context.

    @param      statement : str
                First argument to eval()

    @param      do_format : bool
                If true, format(**locals) will be called on statement
                before passing it to eval().

    @param      globals : dict
                Second argument to eval(). If None, use 'caller_globals' if it
                is given, else use empty dict {}.

    @param      locals : dict
                Third argument to eval(). If None, use 'caller_locals' if it
                is given, else use empty dict {}.

    @return     result of eval() of statement with given locals and globals.
    """

    stmt = context["statement"]

    # None means copy 
    locals_dict = context.get("locals", None)
    if locals_dict is None:
        locals_dict = context.get("caller_locals", {}) # if default is None it will use the ones from this module

    globals_dict = context.get("globals", None)
    if globals_dict is None:
        globals_dict = context.get("caller_globals", {}) # if default is None it will use the ones from this module


    if context.get("do_format", False):
        stmt = stmt.format(**locals_dict)

    return eval(stmt, globals_dict, locals_dict)
