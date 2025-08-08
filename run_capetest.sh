#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/007_pyfun/05_databook" \
    --junitxml=test/junit.xml \
    --pdb \

