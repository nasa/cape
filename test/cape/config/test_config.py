#!/usr/bin/env python

import cape.config
import os

# Function to create "FAIL" file
def fail_msg(msg):
    f = open('FAIL', 'w')
    f.write("%s\n" % msg)
    f.close()

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
    fail_msg("Failed to read 'arro2.xml'")
    os.sys.exit(4)
    
# Get component for fin3 results
f3x = cfgx.GetCompID('fin3')
f3j = cfgj.GetCompID('fin3')
f32 = cfg2.GetCompID('fin3')
# Check results
if f32 != f3j or f32 != f3x:
    fail_msg("Failed GetCompID() method for fin3")
    os.sys.exit(5)
    
# Check all components
for face in cfgj.faces:
    # Get values
    cx = cfgx.faces[face]
    cj = cfgj.faces[face]
    q = False
    # Check for list
    if type(cx).__name__ == 'list':
        cx.sort()
        cj.sort()
    # Compare
    if cx != cj:
        fail_msg("JSON and XML don't match for component '%s'" % face)
        q = True
        break
if q: os.sys.exit(6)

# Try to reset
try:
    # Renumber from 1 to *n*
    cfgj.ResetCompIDs()
    # Get the full list
    comps = cfgj.faces['bullet_total']
    # Check the result
    if comps != range(1,8):
        fail_msg("Unexpected component numbers after resetting")
        os.sys.exit(7)
except Exception:
    fail_msg("Failed to reset ConfigJSON component IDs")
    os.sys.exit(7)

# Passed.
open('PASS', 'w').close()
