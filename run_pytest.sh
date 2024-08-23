#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/002_attdb/013_plot_contour" \
    "test/003_tnakit/001_subplot_col" \
    --pdb \
    --junitxml=test/junit.xml \
    --cov=$PKG \
    --cov-report html:test/htmlcov

