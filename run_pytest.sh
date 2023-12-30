#!/bin/bash

# Package name
PKG="cape"

# Run tests
winpty py -m pytest \
    "test/001_cape/015_filecntl" \
    --junitxml=test/junit.xml \
    --cov=$PKG \
    --cov-report html:test/htmlcov

# Save result
IERR=$?

# Create sphinx docs of results
python3 -m testutils write-rst

# Return pytest's status
exit $IERR

