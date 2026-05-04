#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/902_pyfun/002_ellipsoid" \
    --junitxml=test/junit.xml \
    --pdb \

