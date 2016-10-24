#!/usr/bin/env python

import cape.tri
import numpy as np
import os, shutil

# Go to this folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# List of files to remove
fglob = ['FAIL', 'PASS', 'over.o']
ftri = [
    'bullet.s.b4.tri', 'bullet.s.lb4.tri',
    'bullet.f.b4.tri', 'bullet.f.lb4.tri'
]
# Remove the files if necessary
for f in (fglob+ftri):
    if os.path.isfile(f): os.remove(f)

# Function to create 'FAIL' file
def fail_msg(msg):
    f = open('FAIL', 'w')
    f.write("%s\n" % msg)
    f.close()
    
# Function to get byte order of file
def test_props(fname, endian, tri):
    """Get TRI file properties from ``overConvert`` and compare to input
    
    :Call:
        >>> ierr = test_props(fname, endian, tri)
    :Inputs:
        *fname*: :class:`str`
            Name of tri file
        *endian*: ``"big"`` | ``"little"``
            Required byte order
        *tri*: :class:`cape.tri.Tri`
            Triangulation instance for testing node and tri count
    :Outputs:
        *ierr*: :class:`int`
            0: no errors
            1: wrong byte order
            2: wrong node count
            3: wrong face count
    :Versions:
        * 2016-10-24 ``@ddalle``: First version
    """
    try:
        # Run overConvert
        os.system('overConvert -i %s -v > over.o' % fname)
        # Get the lines from the file
        f = open('over.o')
        lines = f.readlines()
        f.close()
        # Get byte order
        words = lines[3].strip().split()
        bo = words[2].lower()
        # Get vertex and face count
        nNode = int(lines[6].split()[2])
        nTri  = int(lines[6].split()[5])
        # Tests
        if bo != endian:
            return 1
        elif nNode != tri.nNode:
            return 2
        elif nTri != tri.nTri:
            return 3
        else:
            return 0
    except Exception:
        return 5
    
# Read the file
tri = cape.tri.Tri(tri='bullet.tri')

# Write varietals
try:
    tri.WriteSlow_lb4('bullet.s.lb4.tri')
    tri.WriteFast_lb4('bullet.f.lb4.tri')
    tri.WriteSlow_b4('bullet.s.b4.tri')
    tri.WriteFast_b4('bullet.f.b4.tri')
except Exception:
    fail_msg('Failed during writing tri files')
    os.sys.exit(4)

# Test endianness
ierr = test_props('bullet.s.lb4.tri', 'little', tri)
if ierr:
    fail_msg('Failed "bullet.s.lb4.tri"')
    os.sys.exit(ierr)

ierr = test_props('bullet.f.lb4.tri', 'little', tri)
if ierr:
    fail_msg('Failed "bullet.f.lb4.tri"')
    os.sys.exit(ierr)
    
ierr = test_props('bullet.s.b4.tri', 'big', tri)
if ierr:
    fail_msg('Failed "bullet.s.b4.tri"')
    os.sys.exit(ierr)
    
ierr = test_props('bullet.f.b4.tri', 'big', tri)
if ierr:
    fail_msg('Failed "bullet.f.b4.tri"')
    os.sys.exit(ierr)

# Passed.
open('PASS', 'w').close()

