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