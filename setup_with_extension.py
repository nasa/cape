#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party
from setuptools import setup

# Local imports
from cape import setup_py
from cape.setup_py.extensions import EXTENSIONS


# Compile and link
setup(ext_modules=EXTENSIONS, **setup_py.SETUP_SETTINGS)

