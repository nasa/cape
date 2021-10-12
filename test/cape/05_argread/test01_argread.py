#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import sys

# Import cape module
import cape


# Print dict in sorted order
def printdict(kw):
    sys.stdout.write("{")
    for j, k in enumerate(sorted(list(kw.keys()))):
        v = kw[k]
        if j > 0:
            sys.stdout.write(", ")
        sys.stdout.write("'%s': %s" % (k, repr(v)))
    sys.stdout.write("}\n")
    sys.stdout.flush()

# Basic Keys
# Test 1
# List of arguments
argv = ['cape', 'run', '-c', '--filter', 'b2']

# Process it
a, kw = cape.argread.readkeys(argv)
# Print results
print(a)
printdict(kw)

# Test 2
# List of arguments
argv = ['cape', 'run', '-c', '-filter', 'b2']

# Process it
a, kw = cape.argread.readkeys(argv)
# Print results
print(a)
printdict(kw)

# Test 3
# List of arguments
argv = ['cape', '--aero', 'body', '--aero', 'base', '--aero', 'fin', '-c']

# Process it
a, kw = cape.argread.readkeys(argv)
# Print results
print(a)
printdict(kw)

# Read as Flags
# Test 4
# List of arguments
argv = ['cape', 'run', '-c', '--filter', 'b2']

# Process it
a, kw = cape.argread.readflags(argv)
# Print results
print(a)
printdict(kw)

# Test 5
# List of arguments
argv = ['cape', 'run', '-c', '-fr', 'b2']

# Process it
a, kw = cape.argread.readflags(argv)
# Print results
print(a)
printdict(kw)

# Read flags like tar command
# Test 6
# List of arguments
argv = ['cape', 'run', '-c', '-fr', 'b2']

# Process it
a, kw = cape.argread.readflagstar(argv)
# Print results
print(a)
printdict(kw)
