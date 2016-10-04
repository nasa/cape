#!/bin/bash

# Required modules
. $MODULESHOME/init/bash
module load pycart
module load aflr3

# Convert to ASCII tri
pc_UH3D2Tri.py -i bullet-far.uh3d -c bullet-far.xml

# Convert to surface
pc_Tri2Surf.py \
    -i  bullet-far.tri \
    -c  bullet-far.xml \
    -bc bullet-far.bc

# Create the mesh
aflr3 -i bullet-far.surf -o bullet-far.ugrid \
    angblisimx=175 -blc -bli 5 -blr 1.2 -grow2 \
    nqual=2

# Nice copy
ugc bullet-far.ugrid bullet-far.cgns
