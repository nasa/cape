#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape
import cape.cfdx.case
import cape.cfdx.cmd

# Read settings
rc = cape.cfdx.case.ReadCaseJSON()

# Form command
cmd1 = cape.cfdx.cmd.intersect(rc)
# Alternate form
cmd2 = cape.cfdx.cmd.intersect(T=True)

# Output
print(' '.join(cmd1))
print(' '.join(cmd2))
