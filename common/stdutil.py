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