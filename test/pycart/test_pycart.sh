#!/bin/bash

# Function to run a test and check its status
function test {
    # Run the test
    echo "Testing '$1'..."
    python $1/test_$1.py > $1/test.out
    # Check the status.
    if [ -f $1/FAIL ]; then
        echo "  FAIL"
        exit 1
    elif [ -f $1/PASS ]; then
        echo "  PASS"
    else
        echo "  Undetermined status"
        exit 2
    fi
}

# Prepare the environment
. $MODULESHOME/init/bash
module load pycart
module load cart3d

# Run the test
test bullet

