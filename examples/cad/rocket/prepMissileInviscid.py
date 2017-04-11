#!/usr/bin/env python

# Triangulation
import cape.tri

# Read the UH3D file
tri = cape.tri.Tri("missile.uh3d", c="missile-inviscid.json")

# Read the farfield 
tri0 = cape.tri.Tri("farfield.uh3d", c="missile-inviscid.json")

# Add them to a single triangulation
tri.Add(tri0)

# Write an XML file
tri.WriteConfigXML("missile-inviscid.xml")

# Output
tri.Write("missile-inviscid.tri")

# Apply the BCs for AFLR3
tri.MapBCs_ConfigAFLR3()

# Write the AFLR3 input file
tri.WriteSurf("missile-inviscid.surf")

# Write the FUN3D BC file
tri.config.WriteFun3DMapBC("missile-inviscid.mapbc")

