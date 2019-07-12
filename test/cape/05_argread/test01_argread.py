#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape

# Basic Keys
# Test 1
# List of arguments
argv = ['cape', 'run', '-c', '--filter', 'b2']

# Process it
a, kw = cape.argread.readkeys(argv)
# Print results
print(a)
print(kw)

# Test 2
# List of arguments
argv = ['cape', 'run', '-c', '-filter', 'b2']

# Process it
a, kw = cape.argread.readkeys(argv)
# Print results
print(a)
print(kw)

# Test 3
# List of arguments
argv = ['cape', '--aero', 'body', '--aero', 'base', '--aero', 'fin', '-c']

# Process it
a, kw = cape.argread.readkeys(argv)
# Print results
print(a)
print(kw)

# Read as Flags
# Test 4
# List of arguments
argv = ['cape', 'run', '-c', '--filter', 'b2']

# Process it
a, kw = cape.argread.readflags(argv)
# Print results
print(a)
print(kw)

# Test 5
# List of arguments
argv = ['cape', 'run', '-c', '-fr', 'b2']

# Process it
a, kw = cape.argread.readflags(argv)
# Print results
print(a)
print(kw)

# Read flags like tar command
# Test 6
# List of arguments
argv = ['cape', 'run', '-c', '-fr', 'b2']

# Process it
a, kw = cape.argread.readflagstar(argv)
# Print results
print(a)
print(kw)