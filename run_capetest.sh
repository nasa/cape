#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/901_pycart" \
    --junitxml=test/junit.xml \
    --pdb \
    --cov=$PKG \
    --cov-report html:test/htmlcov

