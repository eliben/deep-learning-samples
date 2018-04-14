# Use Python 2...
from __future__ import print_function
import sys


def clean_line(line):
    line = line.strip().lower()
    new_line = []
    for c in line:
        i = ord(c)
        if i >= 32 and i <= 126:
            new_line.append(c)
    return ''.join(new_line)


if __name__ == '__main__':
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        for line in f:
            cl = clean_line(line)
            if cl:
                print(cl)
