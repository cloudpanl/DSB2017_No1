# coding=utf-8
from __future__ import print_function

import contextlib
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

WORKERS = multiprocessing.cpu_count()


@contextlib.contextmanager
def multiThread(workers=None):
    workers = workers or WORKERS
    pool = ThreadPool(processes=workers)
    yield pool
    pool.close()
    pool.join()


@contextlib.contextmanager
def multiProcess(workers=None):
    workers = workers or WORKERS
    pool = Pool(processes=workers)
    yield pool
    pool.close()
    pool.join()
