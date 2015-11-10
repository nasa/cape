#!/usr/bin/env python

import pyCart
import os, shutil

# Go to this folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Clean up if necessary.
if os.path.isdir('poweroff'): shutil.rmtree('poweroff')
if os.path.isdir('data'):     shutil.rmtree('data')
# Remove FAIL and PASS file if necessary
if os.path.isfile('FAIL'): os.remove('FAIL')
if os.path.isfile('PASS'): os.remove('PASS')

# List cases
ierr = os.system('pycart -c')
# Check failure
if ierr:
    f = open('FAIL', 'w')
    f.write('Failed to list case statuses.')
    f.close()
    os.sys.exit(ierr)

# Run the first case
ierr = os.system('pycart -I 0')
# Check failure
if ierr:
    f = open('FAIL', 'w')
    f.write('Failed to run first case.')
    f.close()
    os.sys.exit(ierr)

# Calculate data book
ierr = os.system('pycart -I 0 --aero')
# Check failure
if ierr:
    f = open('FAIL', 'w')
    f.write('Failed to compute aero.')
    f.close()
    os.sys.exit(ierr)

try:
    # Read the databook
    cart3d = pyCart.Cart3d()
    cart3d.ReadDataBook()
    # Get the value.
    CA = cart3d.DataBook['bullet_no_base']['CA'][0]
    # Check it
    if abs(CA - 0.745) > 0.02:
        f = open('FAIL', 'w')
        f.write('Data book value did not match expected result.')
        f.close()
        os.sys.exit(1)
except Exception:
    f = open('FAIL', 'w')
    f.write('Failed to read the data book.')
    f.close()
    os.sys.exit(1)

# Passed.
open('PASS', 'w').close()

