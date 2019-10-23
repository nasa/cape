#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape
import cape.cfdx.cmd
import cape.cfdx.case

# Read settings
rc = cape.cfdx.case.ReadCaseJSON()

# Form command
cmd1 = cape.cfdx.cmd.aflr3(rc)
# Alternate form
cmd2 = cape.cfdx.cmd.aflr3(
    i="pyfun.surf",
    o="pyfun.lb8.ugrid",
    blr=10,
    flags={"someflag": 2},
    keys={"somekey": 'c'})

# Output
print(cmd1[-1])
print(' '.join(cmd2))
