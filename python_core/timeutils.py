import time
import sys
import logging
from importlib import reload
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        walltime = te - ts
        logging.info('{} runtime: {:.0f}s'.format(method.__name__, (te - ts)))
        return result, walltime
    return timed

def update_progress(job_title, progress):
    length = 40 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()