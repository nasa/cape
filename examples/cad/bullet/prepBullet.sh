#!/bin/bash

# Required modules
. $MODULESHOME/init/bash
module load pycart

# Convert to ASCII tri
pc_UH3D2Tri.py -i bullet.uh3d -c bullet.xml

