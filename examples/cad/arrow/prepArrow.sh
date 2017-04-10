#!/bin/bash

# Required modules
. $MODULESHOME/init/bash
module load pycart

# Convert to ASCII tri
pc_UH3D2Tri.py -i arrow.uh3d -c arrow.xml

