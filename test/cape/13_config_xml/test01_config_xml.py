#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape.config

# Basic Tests

# Open the JSON config
try:
    cfgj = cape.config.ConfigJSON("arrow.json")
except Exception:
    fail_msg("Failed to read 'arrow.json'")
    os.sys.exit(1)

# Open the XML config
try:
    cfgx = cape.config.Config("arrow.xml")
except Exception:
    fail_msg("Failed to read 'arrow.xml'")
    os.sys.exit(2)

# Write XML from the JSON
try:
    cfgj.WriteXML("arrow2.xml")
except Exception:
    fail_msg("Failed to write XML file from JSON config")
    os.sys.exit(3)
    
# Read new XML
try:
    cfg2 = cape.config.Config("arrow2.xml")
except Exception:
    fail_msg("Failed to read 'arrow2.xml'")
    os.sys.exit(4)
 

