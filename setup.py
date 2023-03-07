#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library


# Third-party
from setuptools import setup

# Local imports
from cape import setup_py


# Compile and link
setup(**setup_py.SETUP_SETTINGS)
