#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/007_pyfun/02_conditionals" \
    --junitxml=test/junit.xml \
    --pdb \

