#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape

# Read settings
rc = cape.case.ReadCaseJSON()

# Form command
cmd1 = cape.cmd.intersect(rc)
# Alternate form
cmd2 = cape.cmd.intersect(T=True)

# Output
print(' '.join(cmd1))
print(' '.join(cmd2))
