#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    "test/001_cape/051_cli" \
    --junitxml=test/junit.xml \
    --pdb \

