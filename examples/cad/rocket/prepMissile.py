#!/usr/bin/env python

# Triangulation
import cape.tri

# Read the UH3D file
tri = cape.tri.Tri("missile.uh3d", c="missile.json")

# Write an XML file
tri.WriteConfigXML("missile.xml")

# Output
tri.Write("missile.tri")

# Read just the engine
tri = cape.tri.Tri("engine.uh3d", c="missile.json")

# Output
tri.Write("engine.tri")
