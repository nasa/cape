#!/bin/bash

# Package name
PKG="cape"

# Run tests
python3 -m pytest \
    --ignore-glob 'test/[a-z]*' \
    --ignore-glob 'test/0[5-9]*' \
    --pdb \
    --junitxml=test/junit.xml \
    --cov=$PKG \
    --cov-report html:test/htmlcov

# Save result
IERR=$?

# Track coverage report
rm test/htmlcov/.gitignore

# Create sphinx docs of results
python3 -m testutils write-rst

# Return pytest's status
exit $IERR

