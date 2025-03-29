#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/902_pyfun/004_hooks" \
    --junitxml=test/junit.xml \
    --pdb \

