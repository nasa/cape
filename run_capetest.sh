#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/001_cape/014_cntl" \
    --pdb \
    --junitxml=test/junit.xml \

# Save result
IERR=$?

# Create sphinx docs of results
#python3 -m testutils write-rst

# Return pytest's status
exit $IERR

