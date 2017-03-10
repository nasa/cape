#!/usr/bin/env python

import pyCart
import cape.test
import numpy as np
import os, shutil

# Go to this folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Remove output file if necessary
if os.path.isfile('arrow.tri'): os.remove('arrow.tri')
# Remove FAIL and PASS file if necessary
if os.path.isfile('FAIL'): os.remove('FAIL')
if os.path.isfile('PASS'): os.remove('PASS')

# Run a system command
cape.test.callt("pc_UH3D2Tri.py -i arrow.uh3d -c arrow.xml",
    'Failed to convert UH3D file to Cart3D triangulation.')

# Test tri.WriteFast
try:
    # Status update
    print("\n> tri = pyCart.Tri('arrow.tri')")
    os.sys.stdout.flush()
    # Read the triangulation
    tri = pyCart.Tri('arrow.tri')
    # Status update
    print("> tri.WriteFast('arrow.i.tri')")
    os.sys.stdyou.flush()
    # Write using "Fast" method (C compiled)
    tri.WriteFast('arrow.i.tri')
except Exception:
    f = open('FAIL', 'w')
    f.write('Failed to use Tri.WriteFast')
    f.close()
    os.sys.exit(1)

try:
    # Status update
    print("\n> (testing list of component IDs)")
    os.sys.stdout.flush()
    # Get the comp IDs
    compID = np.unique(tri.CompID)
    # Check it
    if np.any(compID != np.array([1,2,3,11,12,13,14])):
        f = open('FAIL', 'w')
        f.write('Component ID list does not match')
        f.close()
        os.sys.exit(1)
except Exception:
    f = open('FAIL', 'w')
    f.write('Failed to process component IDs.')
    f.close()
    os.sys.exit(1)

# Passed.
open('PASS', 'w').close()

