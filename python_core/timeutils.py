import time
import sys
import logging
from importlib import reload

from functools import wraps

def timeit(get_time=False):
    def _timeit(method):
        @wraps(method)
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            walltime = te - ts
            logging.info('{} runtime: {:.0f}s'.format(method.__name__, (te - ts)))
            if get_time:
                return result, walltime
            else:
                return result
        return timed
    return _timeit

def update_progress(job_title, progress):
    length = 40 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()