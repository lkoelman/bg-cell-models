"""
Logging utilities.

CREDITS

MultiProcessingHandler was copied from https://github.com/jruere/multiprocessing-logging

"""

from __future__ import absolute_import, division, unicode_literals

import logging
import multiprocessing
import sys
import threading
import traceback


# Create a very verbose log level named "ANAL"
DEBUG_ANAL_LVL = 5 # see https://docs.python.org/2/library/logging.html#logging-levels
logging.addLevelName(DEBUG_ANAL_LVL, "ANAL")

def anal_logfun(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_ANAL_LVL):
    	# Yes, logger takes its '*args' as 'args'.
        self._log(DEBUG_ANAL_LVL, message, args, **kws) 

logging.Logger.anal = anal_logfun


def setLogLevel(level, logger_names):
	"""
	Set log level of all loggers with given names to level.

	@param level	any log level accepted by logging.setLevel() or one of following:
					'verbose', 'quiet', 'silent', 'anal'
	"""
	if level=='verbose':
		level = logging.DEBUG
	elif level=='silent' or level=='quiet':
		level = logging.WARNING
	elif level=='anal':
		level = DEBUG_ANAL_LVL

	for logname in logger_names:
		slogger = logging.getLogger(logname)
		if slogger: slogger.setLevel(level)


def install_mp_handler(logger=None):
	"""
	Wraps the handlers in the given Logger with an MultiProcessingHandler.

	:param logger: whose handlers to wrap. By default, the root logger.

	USAGE

		import logutils
		logutils.install_mp_handler()
	
	"""
	if logger is None:
		logger = logging.getLogger()

	for i, orig_handler in enumerate(list(logger.handlers)):
		handler = MultiProcessingHandler(
			'mp-handler-{0}'.format(i), sub_handler=orig_handler)

		logger.removeHandler(orig_handler)
		logger.addHandler(handler)


class MultiProcessingHandler(logging.Handler):
	"""
	Usage: see install_mp_handler()
	"""

	def __init__(self, name, sub_handler=None):
		super(MultiProcessingHandler, self).__init__()

		if sub_handler is None:
			sub_handler = logging.StreamHandler()

		self.sub_handler = sub_handler
		self.queue = multiprocessing.Queue(-1)
		self.setLevel(self.sub_handler.level)
		self.setFormatter(self.sub_handler.formatter)
		# The thread handles receiving records asynchronously.
		t = threading.Thread(target=self.receive, name=name)
		t.daemon = True
		t.start()

	def setFormatter(self, fmt):
		logging.Handler.setFormatter(self, fmt)
		self.sub_handler.setFormatter(fmt)

	def receive(self):
		while True:
			try:
				record = self.queue.get()
				self.sub_handler.emit(record)
			except (KeyboardInterrupt, SystemExit):
				raise
			except EOFError:
				break
			except:
				traceback.print_exc(file=sys.stderr)

	def send(self, s):
		self.queue.put_nowait(s)

	def _format_record(self, record):
		# ensure that exc_info and args
		# have been stringified. Removes any chance of
		# unpickleable things inside and possibly reduces
		# message size sent over the pipe.
		if record.args:
			record.msg = record.msg % record.args
			record.args = None
		if record.exc_info:
			self.format(record)
			record.exc_info = None

		return record

	def emit(self, record):
		try:
			s = self._format_record(record)
			self.send(s)
		except (KeyboardInterrupt, SystemExit):
			raise
		except:
			self.handleError(record)

	def close(self):
		self.sub_handler.close()
		logging.Handler.close(self)
