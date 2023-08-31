import contextlib
import logging
import time


@contextlib.contextmanager
def log_duration(operation):
	logging.info(operation)
	start_time = time.time()
	yield
	duration = time.time() - start_time
	logging.debug(f"{operation} took {duration:.2f} seconds")
