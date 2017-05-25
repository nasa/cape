#!/bin/bash

# Required modules
. $MODULESHOME/init/bash
module load pycart

# Convert to ASCII tri
pc_UH3D2Tri.py -i bullet.uh3d -c bullet.xml
pc_UH3D2Tri.py -i bullet.uh3d -lb4

# Generate curves
pc_StepTri2Crv.py bullet -o bullet.lb8.crv -lb8

