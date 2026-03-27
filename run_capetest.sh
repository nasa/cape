#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/002_dkit/013_plot_contour" \
    --junitxml=test/junit.xml \
    --pdb \

