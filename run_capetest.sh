#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/001_cape/050_cli" \
    "test/901_pycart/" \
    --junitxml=test/junit.xml \
    --pdb \

