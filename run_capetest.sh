#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/902_pyfun/003_cigar" \
    --junitxml=test/junit.xml \
    --pdb \

