from __future__ import print_function
from urllib2 import urlopen
from urlparse import urljoin
import os
from shutil import copyfileobj


def maybe_download(base_url, filename, expected_size, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        in_stream = urlopen(urljoin(base_url, filename))
        with open(filename, 'wb') as out_file:
            copyfileobj(in_stream, out_file)
        print('Download Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_size:
        print('Found and verified', filename)
        return True
    else:
        print('Unable to verify size: {0} vs. expected {1}'.format(
            statinfo.st_size, expected_size))
        return False


maybe_download('http://thegreenplace.net/files/',
               filename='mnist.pkl.gz', expected_size=16168813)
