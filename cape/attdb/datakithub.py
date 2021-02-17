#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.attdb.datakithub`: Hub for importing DataKits by name
=================================================================

This module provides the class :class:`DataKitHub` that provides a tool
to simplify the importing of named "datakits" (from
:mod:`cape.attdb.rdb.DataKit`).  More specifically, it allows users to
create one or more naming conventions for databases/datakits and to
read that data with minimal low-level Python programming.

An instance of the :class:`DataKitHub` class is created by reading a
JSON file that contains the naming conventions, such as the following:

    .. code-block:: python

        from cape.attdb.datakithub import DataKitHub

        # Create an instance
        hub = DataKitHub("datakithub.json")

A simple ``datakithub.json`` file might contain the following:

    .. code-block:: javascript

        {
            "DB-ATT": {
                "repo": "/home/user/datakit/db",
                "type": "module",
                "module_attribute": "db",
                "module_regex": {
                    "DB-ATT-([0-9]+)": "dbatt.db%s",
                },
            }
        }

It will make more sense to explain this content after seeing an example.
Now we can use the :class:`DataKitHub` instance to read databases by
their title, such as ``"DB-ATT-1"`` or ``"DB-ATT-002"``, as long as they
start with ``"DB-ATT"`` or some other string defined in the JSON file.

    .. code-block:: python

        from cape.attdb.datakithub import DataKitHub

        # Create an instance
        hub = DataKitHub("/home/user/datakit/datakithub.json")

        # Read the database "DB-ATT-1"
        db1 = hub.read_db("DB-ATT-1")

        # Read the database "DB-ATT-002"
        db2 = hub.read_db("DB-ATT-002")

This is roughly the same as

    .. code-block:: python

        # Read the database "DB-ATT-1"
        import dbatt.db1
        db1 = dbatt.db1.db

        # Read the database "DB-ATT-002"
        import dbatt.db002
        db2 = dbatt.db002.db

but without having to deal with either ``sys.path`` or the *PYTHONPATH*
environment variable, which can be both tedious and difficult to make
work for multiple users on different types of computers.

Here is a description of the JSON parameters

    *repo*: :class:`str`
        Name of the folder containing the data or modules
    *type*: ``"module"``
        Type of datakit to read
    *module_attribute*: :class:`str` | ``None``
        Name of variable in imported module to use as datakit
    *module_regex*: :class:`dict`\ [:class:`str`]
        Rules for converting a regular expression to module names

"""

# Standard library modules
import os
import re
import sys

# CAPE modules
from cape.cfdx.options.util import loadJSONFile


# DataKit hub class
class DataKitHub(dict):
    # Initialization method
    def __init__(self, fjson="datakithub.json"):
        r"""Initialization method

        :Versions:
            * 2021-02-17 ``@ddalle``: Version 1.0
        """
        # Read the JSON file
        opts = loadJSONFile(fjson)
        # Save it...
        self.update(opts)

    # Find best matching category
    def get_group(self, name):
        r"""Find the first group that matches a datakit *name*

        :Call:
            grpname, grp = hub.get_group(name)
        :Inputs:
            *hub*: :class:`DataKitHub`
                Instance of datakit-reading hub
            *name*: :class:`str`
                Name of datakit to read
        :Outputs:
            *grpname*: :class:`str`
                Title of datakit reading group
            *grp*: :class:`dict`
                Options for that group
        :Versions:
            * 2021-02-17 ``@ddalle``: Version 1.0
        """
        # Loop through sections
        for grpname, grp in self.items():
            # Check if *name* matches
            if re.match(grpname, name):
                # Found a match!
                return grpname, grp

