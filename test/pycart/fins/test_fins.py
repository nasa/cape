#!/usr/bin/env python

import pyCart
import cape.test
import os, shutil, glob

# Go to this folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Clean up if necessary.
for fdir in glob.glob('poweroff*'):
    shutil.rmtree(fdir)
# Remove FAIL and PASS file if necessary
if os.path.isfile('FAIL'): os.remove('FAIL')
if os.path.isfile('PASS'): os.remove('PASS')

# Show status
cape.test.callt('pycart -c -f fins.json', "Failed to show status.")

# Run the case 9
cape.test.callt('pycart -f fins.json -I 8', "Failed to run case 8.")

# Calculate data book (with cases not computed)
cape.test.callt('pycart -f fins.json --aero', "Failed to collect aero data.")

# API DataBook read
print("\n> cart3d.ReadDataBook()")
os.sys.stdout.flush()
try:
    # Read the databook
    cart3d = pyCart.Cart3d('fins.json')
    cart3d.ReadDataBook()
    # Sort
    cart3d.DataBook.UpdateRunMatrix()
    # Get component
    DBc = cart3d.DataBook['fins']
    # Get the value.
    i = DBc.FindMatch(8)
    CN = DBc['CN'][i]
    # Check it
    cape.test.test_val(CN, 0.0055, 0.02)
except Exception:
    f = open('FAIL', 'w')
    f.write('Failed to read/sort/process the data book.')
    f.close()
    os.sys.exit(1)

# Passed.
open('PASS', 'w').close()

