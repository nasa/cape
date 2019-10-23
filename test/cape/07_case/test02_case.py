#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape.cfdx.case

# Read conditions
rc = cape.cfdx.case.ReadCaseJSON()

# Show settings
print(rc.get_PhaseSequence())
print(rc.get_PhaseIters(1))
print(rc.get_PhaseIters(2))
print(rc.get_PhaseIters(3))
print(rc.get_qsub())
