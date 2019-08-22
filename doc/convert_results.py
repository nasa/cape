#!/usr/bin/env python
"""
Convert Sphinx doctest output to reST
======================================

This script converts the ``output.txt`` file in ``_build/doctest/`` to a nicely
formatted reST file called ``test/results.rst``.

    .. code-block:: console

        $ ./convert_results.py

:Versions:
    * 2018-03-27 ``@ddalle``: First version
"""

# Testing module
import cape.test

# Run it
if __name__ == "__main__":
    cape.test.convert_results()

