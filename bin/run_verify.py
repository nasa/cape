#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Run Cart3D/``verify``: ``run_verify.py``
==========================================

This executable Python file runs Cart3D's ``verify`` tool.

:Versions:
    * 2014-02-14 ``@ddalle``: Version 1.0
"""

# Import the module specifically for this task.
import cape.pycart.bin


# Simple function to call the main function of that module.
def run_verify():
    r"""Calls :func:`cape.pycart.bin.verify`
    
    :Call:
        >>> run_verify()
    :Versions:
        * 2015-02-13 ``@ddalle``: Version 1.0
    """
    cape.pycart.bin.verify()


# Check if run as a script.
if __name__ == "__main__":
    run_verify()
