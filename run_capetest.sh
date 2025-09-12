#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/901_pycart" \
    "test/902_pyfun/001_bullet" \
    --junitxml=test/junit.xml \
    --pdb \

