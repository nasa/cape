#!/bin/bash

# Load CAPE and FUN3D
. /usr/share/Modules/init/bash
module use -a /u/wk/ddalle/share/modulefiles
module load fun3d
module load cape

# Run the CAPE test script
cape_TestCrawler.py

