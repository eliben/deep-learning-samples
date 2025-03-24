#!/bin/bash

# Downloads the text8 datased.

set -eu
set -o pipefail

if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi
