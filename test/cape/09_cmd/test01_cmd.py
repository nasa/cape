#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape

# Read settings
rc = cape.case.ReadCaseJSON()

# Form command
cmd1 = cape.cmd.aflr3(rc)
# Alternate form
cmd2 = cape.cmd.aflr3(
    i="pyfun.surf",
    o="pyfun.lb8.ugrid",
    blr=10,
    flags={"someflag": 2},
    keys={"somekey": 'c'})

# Output
print(cmd1[-1])
print(' '.join(cmd2))