#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/003_tnakit" \
    "test/001_cape/007_databook" \
    --junitxml=test/junit.xml \
    --pdb \
    --cov=$PKG \
    --cov-report html:test/htmlcov

