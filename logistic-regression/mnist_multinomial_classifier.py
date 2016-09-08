from __future__ import print_function
import argparse
import numpy as np
import sys

from mnist_dataset import *
from regression_lib import *


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    args = argparser.parse_args()

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_mnist_data()

    print(X_train.shape)
