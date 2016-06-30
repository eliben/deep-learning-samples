from __future__ import print_function
import os, sys

if len(sys.argv) < 2:
    print('expecting dir as argument')
    sys.exit(1)

dirname = sys.argv[1]
files = os.listdir(dirname)
print(dirname, 'file count:', len(files))

totalsize = 0
for file in files:
    fullpath = os.path.join(dirname, file)
    size = os.stat(fullpath).st_size
    totalsize += size

print('totalsize:', totalsize)
