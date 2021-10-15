#!/bin/bash

# Initialize modules
. /usr/share/Modules/init/bash
module use -a /u/w/ddalle/share/modulefiles

# Environment
module load cape
module load cart3d
module load fun3d
module load mpi
module load overflow

# Go to appropriate folder
cd /u/wk/ddalle/usr/cape/test

# Run test crawler and document results
python3 test_all.py

