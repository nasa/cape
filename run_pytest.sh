#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/000_vendor" \
    "test/001_cape" \
    "test/002_attdb" \
    "test/005_cfdx" \
    --pdb \
    --junitxml=test/junit.xml \
    --cov=$PKG \
    --cov-report html:test/htmlcov

