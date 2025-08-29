#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/001_cape/014_cntl" \
    --junitxml=test/junit.xml \
    --pdb \

