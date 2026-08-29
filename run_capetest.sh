#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/001_cape/018_agentopts" \
    --junitxml=test/junit.xml \
    --pdb \

