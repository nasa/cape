#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/903_pyover/001_bullet/" \
    --junitxml=test/junit.xml \
    --pdb \

