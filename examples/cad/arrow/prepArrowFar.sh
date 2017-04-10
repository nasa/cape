#!/bin/bash

# Required modules
. $MODULESHOME/init/bash
module load pycart
module load aflr3

# Convert to ASCII tri
pc_UH3D2Tri.py -i arrow-far.uh3d -c arrow-far.xml

# Convert to surface
pc_Tri2Surf.py \
    -i  arrow-far.tri \
    -c  arrow-far.xml \
    -bc arrow-far.bc

# Create the mesh
aflr3 -i arrow-far.surf -o arrow-far.ugrid \
    angblisimx=175 -blc -bli 5 -blr 1.2 -grow2

# Nice copy
ugc arrow-far.ugrid arrow-far.cgns
