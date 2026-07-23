#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/901_pycart/001_bullet/051_cli" \
    --junitxml=test/junit.xml \
    --pdb \

