#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/902_pyfun" \
    --junitxml=test/junit.xml \
    --cov=$PKG \
    --cov-report html:test/htmlcov

