#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import sys library
import sys

# Import cape module
import cape.tri
import cape.plt

# Config file
CONFIGJSONFILE = "arrow.json"
# Prefix
OUTPUT_PREFIX = "arrow"
# XML file
XMLFILE = OUTPUT_PREFIX + ".xml"
# Source uh3d
SOURCE = "arrow.uh3d"
# Output tri file
TRIFILEOUT = "arrow.tri"


# Read uh3d triangulation
print("Reading source triangulation")
tri = cape.tri.Tri(uh3d=SOURCE, c=CONFIGJSONFILE)

# Write XML file
print("Writing ConfigXML file")
print("  %s" % XMLFILE)
tri.WriteConfigXML(XMLFILE)

# Map the AFLR3 boudnary conditions
print("Mapping AFLR3 boundary conditions")
tri.MapBCs_ConfigAFLR3()

# Write the AFLR3 boundary condition summary
print("Writing AFLR3 boundary conditions summary")
tri.config.WriteAFLR3BC(OUTPUT_PREFIX + ".bc")

# Map the FUN3D boundary conditions
print("Mapping FUN3D boundary conditions")
tri.config.WriteFun3DMapBC(OUTPUT_PREFIX + ".mapbc")

# Write surface
print("Writing surface TRI file")
# Number of triangles
ntrik = (tri.nTri - 10) // 1000 + 1
# File name
fname = OUTPUT_PREFIX + ("-tri%ik.lr4.tri" % ntrik)
# Status update
print(" Writing %s" % fname)
# Write it
tri.WriteTri_lr4(fname)

print("Writing combined surface PLT file")
print("  Creating PLT interface")
# Reread tri file
tri0 = cape.tri.Tri(fname, c=XMLFILE)
# Create PLTFile interface
plt = cape.plt.Plt(triq=tri0, c=XMLFILE)
# Number of triangles
ntrik = (tri.nTri - 10) // 1000 + 1
# File name
fname = OUTPUT_PREFIX + ("-tri%ik.plt" % ntrik)
# Status update
print("  Writing %s" % fname)
# Write it
plt.Write(fname)

