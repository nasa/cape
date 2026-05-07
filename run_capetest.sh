#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/000_vendor/004_optdict" \
    --junitxml=test/junit.xml \
    --pdb \

