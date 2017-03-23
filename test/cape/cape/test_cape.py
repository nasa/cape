#!/usr/bin/env python

import cape
import cape.test
import os, shutil
import numpy as np

# Go to this folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Function to create "FAIL" file
def fail_msg(msg, sts=1):
    f = open('FAIL', 'w')
    f.write("%s\n" % msg)
    f.close()
    os.sys.exit(sts)

# Remove FAIL and PASS file if necessary
if os.path.isfile('FAIL'): os.remove('FAIL')
if os.path.isfile('PASS'): os.remove('PASS')

# Show status
cape.test.callt('cape -c', 'Failed to list case statuses.')

# API read
print("\n> cntl = cape.Cntl()")
os.sys.stdout.flush()
try:
    # Read the interface
    cntl = cape.Cntl()
except Exception:
    fail_msg('Failed to read the cape API', 1)
    
# Test options API
print("\n> cntl.opts.get_PhaseIters()")
os.sys.stdout.flush()
try:
    # Access options
    cntl.opts.get_PhaseIters()
except Exception:
    fail_msg("Failed to access standard option", 2)
    
# Test filtering options
print("\n> cntl.x.GetIndices(filter='b2')")
os.sys.stdout.flush()
try:
    # Apply filter
    I = cntl.x.GetIndices(filter="b2")
    # Test the indices
    if np.any(cntl.x.beta[I] != 2):
        fail_msg("The 'cape --filter' test was not accurate", 4)
except Exception:
    fail_msg("Failed to execute 'cape --filter' test", 3)
    
# Test filtering options
print("\n> cntl.x.GetIndices(re='b2')")
os.sys.stdout.flush()
try:
    # Apply filter
    I = cntl.x.GetIndices(re="m.\.5.*b2")
    # Test the indices
    if np.any(cntl.x.beta[I] != 2):
        fail_msg("The 'cape --re' test did not match beta constraint", 6)
    elif np.any(cntl.x.Mach[I]%1 != 0.5):
        fail_msg("The 'cape --re' test did not match mach constraint", 7)
except Exception as e:
    fail_msg("Failed to execute 'cape --re' test", 5)
    
# Test filtering options
print("\n> cntl.x.GetIndices(glob='poweroff/m0*')")
os.sys.stdout.flush()
try:
    # Apply filter
    I = cntl.x.GetIndices(glob="poweroff/m0*")
    # Test the indices
    if np.any(cntl.x.Mach[I] >= 1.0):
        fail_msg("The 'cape --glob' test did not match mach constraint", 9)
except Exception:
    fail_msg("Failed to execute 'cape --glob' test", 8)
    
# Test filtering options
print("\n> cntl.x.GetIndices(cons=['alpha==0', 'Mach%1==0.8'])")
os.sys.stdout.flush()
try:
    # Apply filter
    I = cntl.x.GetIndices(cons=['beta==2', 'Mach%1==0.5'])
    # Test the indices
    if np.any(cntl.x.beta[I] != 2):
        fail_msg("The 'cape --cons' test did not match beta constraint", 11)
    elif np.any(cntl.x.Mach[I]%1 != 0.5):
        fail_msg("The 'cape --cons' test did not match mach constraint", 12)
except Exception:
    fail_msg("Failed to execute 'cape --cons' test", 10)

# Passed.
open('PASS', 'w').close()

