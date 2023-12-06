#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    --ignore-glob 'test/[a-z]*' \
    --ignore-glob 'test/00[0-6]*' \
    --ignore-glob 'test/007_pyfun/0[1-3]*' \
    --ignore-glob 'test/00[8-9]*' \
    --ignore-glob 'test/90[1-3]*' \
    --junitxml=test/junit.xml \
    --cov=$PKG \
    --cov-report html:test/htmlcov

# Save result
IERR=$?

# Create sphinx docs of results
python3 -m testutils write-rst

# Return pytest's status
exit $IERR

