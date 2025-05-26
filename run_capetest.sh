#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/001_cape/" \
    "test/000_vendor/" \
    "test/007_pyfun/" \
    --junitxml=test/junit.xml \
    --pdb \

