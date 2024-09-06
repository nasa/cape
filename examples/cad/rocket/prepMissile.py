#!/usr/bin/env python

# Triangulation
import cape.trifile

# Read the UH3D file
tri = cape.trifile.Tri("missile.uh3d", c="missile.json")

# Write an XML file
trifile.WriteConfigXML("missile.xml")

# Output
trifile.Write("missile.tri")

# Read just the engine
tri = cape.trifile.Tri("engine.uh3d", c="missile.json")

# Output
trifile.Write("engine.tri")
