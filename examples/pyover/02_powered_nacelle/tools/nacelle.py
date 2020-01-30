#!/usr/bin/env python

# System modules
import os
import copy
import numpy as np


# Initialization settings
def InitNAC1(cntl):
    """Modify complex settings that should always be applied

    For instance, sets *ArchiveFolder* according to user

    :Call:
        >>> InitNAC(cntl)
    :Inputs:
        *cntl*: :class:`pyOver.overflow.Overflow`
            OVERFLOW settings interface
    """

# Apply options based on the *Label* RunMatrix key
def ApplyLabel(cntl, i):
    """Modify settings for each case using value of *Label*

    This method is programmed to specify a different OVERFLOW input file based
    on the value of *Label* for a given case. This is used to run each of the
    three input files that come with the powered_nacelle test problem that
    comes with the OVERFLOW source code.

    :Call:
        >>> ApplyLabel(cntl, i)
    :Inputs:
        *cntl*: :class:`pyOver.overflow.Overflow`
            OVERFLOW settings interface
        *i*: :class:`int`
            Case number
    :Versions:
        * 2020-01-28 ``@serogers``: First version
    """

    # Get the specified label
    lbl = cntl.x['Label'][i]

    # Set the overflow input file as a function of the Label
    if 'test01' in lbl:
        cntl.opts['OverNamelist'] = 'common_powered/overflow_test01.inp'
    elif 'test02' in lbl:
        cntl.opts['OverNamelist'] = 'common_powered/overflow_test02.inp'
    elif 'test03' in lbl:
        cntl.opts['OverNamelist'] = 'common_powered/overflow_test03.inp'
# def ApplyLabel

