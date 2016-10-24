#!/usr/bin/env python

import pyCart
import os, shutil, glob

# Go to this folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Clean up if necessary.
for fdir in glob.glob('poweroff*'):
    shutil.rmtree(fdir)
# Remove FAIL and PASS file if necessary
if os.path.isfile('FAIL'): os.remove('FAIL')
if os.path.isfile('PASS'): os.remove('PASS')

# List cases
ierr = os.system('pycart -c -f fins.json')
# Check failure
if ierr:
    f = open('FAIL', 'w')
    f.write('Failed to list case statuses.')
    f.close()
    os.sys.exit(ierr)

# Run the first case
ierr = os.system('pycart -f fins.json -I 8')
# Check failure
if ierr:
    f = open('FAIL', 'w')
    f.write('Failed to run first case.')
    f.close()
    os.sys.exit(ierr)

# Calculate data book
ierr = os.system('pycart -f fins.json --aero')
# Check failure
if ierr:
    f = open('FAIL', 'w')
    f.write('Failed to compute aero.')
    f.close()
    os.sys.exit(ierr)

try:
    # Read the databook
    cart3d = pyCart.Cart3d('fins.json')
    cart3d.ReadDataBook()
    # Sort
    cart3d.DataBook.UpdateTrajectory()
    # Get component
    DBc = cart3d.DataBook['fins']
    # Get the value.
    i = DBc.FindMatch(8)
    CN = DBc['CN'][i]
    # Check it
    if abs(CN - 0.0055) > 0.02:
        f = open('FAIL', 'w')
        f.write('Data book value did not match expected result.')
        f.close()
        os.sys.exit(1)
except Exception:
    f = open('FAIL', 'w')
    f.write('Failed to read/sort/process the data book.')
    f.close()
    os.sys.exit(1)

# Passed.
open('PASS', 'w').close()

