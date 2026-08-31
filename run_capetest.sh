#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/006_pycart/02_casedata" \
    --junitxml=test/junit.xml \
    --pdb \

