#!/bin/bash

# Store current path
FPWD=$(pwd)
# Rise two levels
cd ..
cd ..
# Store current path as root level
CAPE=$(pwd)
# Return to original location
cd $FPWD

# Append paths
export PATH=$PATH:$CAPE/bin
export PYTHONPATH=$PYTHONPATH:$CAPE

# Run the CAPE test script
cape_TestCrawler.py

