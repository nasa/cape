#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape.config

# Read JSON config
cfgj = cape.config.ConfigJSON("arrow.json")

# Write arrow2 XML config from JSON
cfgj.WriteXML("arrow2.xml", Name="bullet sample", Source="bullet.tri")

# Open arrow2 XML
with open("arrow2.xml", "r") as f:
    # Print lines of f
    for line in f:
        print(line)
