from __future__ import print_function
import sys
import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            print('[%s] ' % self.name, end='')
            sys.stdout.flush()

    def __exit__(self, type, value, traceback):
        print('Elapsed: %s' % (time.time() - self.tstart))
