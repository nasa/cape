#!/usr/bin/env python

import pyCart
import cape.test
import os, shutil

# Go to this folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Clean up if necessary.
if os.path.isdir('poweroff'): shutil.rmtree('poweroff')
if os.path.isdir('data'):     shutil.rmtree('data')
# Remove FAIL and PASS file if necessary
if os.path.isfile('FAIL'): os.remove('FAIL')
if os.path.isfile('PASS'): os.remove('PASS')

# Show status
cape.test.callt('pycart -c', 'Failed to list case statuses.')

# Run one case
cape.test.callt('pycart -I 0', 'Failed to run case 0.')

# Assemble the data book
cape.test.callt('pycart -I 0 --aero', 'Failed to compute aero.')

# API DataBook read
print("\n> cart3d.ReadDataBook()")
os.sys.stdout.flush()
try:
    # Read the databook
    cart3d = pyCart.Cart3d()
    cart3d.ReadDataBook()
    # Get the value.
    CA = cart3d.DataBook['bullet_no_base']['CA'][0]
    # Test it
    cape.test.test_val(CA, 0.745, 0.02)
except Exception:
    f = open('FAIL', 'w')
    f.write('Failed to read the data book.')
    f.close()
    os.sys.exit(1)

# Passed.
open('PASS', 'w').close()

