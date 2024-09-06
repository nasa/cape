#!/usr/bin/env python

# Triangulation
import cape.trifile

# Read the UH3D file
tri = cape.trifile.Tri("missile.uh3d", c="missile-inviscid.json")

# Read the farfield 
tri0 = cape.trifile.Tri("farfield.uh3d", c="missile-inviscid.json")

# Add them to a single triangulation
trifile.Add(tri0)

# Write an XML file
trifile.WriteConfigXML("missile-inviscid.xml")

# Output
trifile.Write("missile-inviscid.tri")

# Apply the BCs for AFLR3
trifile.MapBCs_ConfigAFLR3()

# Write the AFLR3 input file
trifile.WriteSurf("missile-inviscid.surf")

# Write the FUN3D BC file
trifile.config.WriteFun3DMapBC("missile-inviscid.mapbc")

