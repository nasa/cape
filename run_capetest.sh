#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/901_pycart/003_isect" \
    --junitxml=test/junit.xml \
    --pdb \

