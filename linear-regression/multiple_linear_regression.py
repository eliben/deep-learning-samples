from __future__ import print_function
import csv
import matplotlib.pyplot as plt
import numpy as np

from timer import Timer

# TODO:
#
# - feature normalization
# - split to training + test set

def read_data(filename):
    """Read data from the given CSV file.

    Returns a 2D Numpy array of type np.float32, with a sample per row.
    """
    with open(filename, 'rb') as file:
        reader = csv.reader(file)
        reader.next() # skip header
        return np.array(list(reader), dtype=np.float32)


if __name__ == '__main__':
    with Timer('reading data'):
        data = read_data('CCPP-dataset/data.csv')
    print(data.shape)
    print(data[0])
    print(data[-1])
