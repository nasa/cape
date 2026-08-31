#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/000_vendor/001_argread" \
    --junitxml=test/junit.xml \
    --pdb \

