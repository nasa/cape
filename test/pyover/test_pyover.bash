#!/bin/bash

# Load CAPE and OVERFLOW
. $HOMDULESHOME/init/bash
module load overflow
module load mpi
module load pycart

# Run the CAPE test script
cape_TestCrawler.py

