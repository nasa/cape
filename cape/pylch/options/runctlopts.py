r"""
:mod:`cape.pylch.options.runctlopts.py`: Loci/CHEM run control options
======================================================================

This module provides special options for the ``"RunControl"`` section
for operating Loci/CHEM. This includes a :class:`RunControlOpts` class
with additional options and classes for the Loci/CHEM executables.
"""

# Local imports
from .chemopts import ChemOpts
from ...cfdx.options import runctlopts


# Class for ``chem`` executable options


# Customized RunControl options
class RunControlOpts(runctlopts.RunControlOpts):
    # Extra options
    _optlist = (
        "chem",
        "ProjectName",
    )

    # Aliases
    _optmap = {
        "JobName": "ProjectName",
        "Name": "ProjectName",
        "ProjectRootName": "ProjectName",
        "RootName": "ProjectName",
        "job_name": "ProjectName",
        "name": "ProjectName",
        "project_name": "ProjectName",
    }

    # Types
    _opttypes = {
        "ProjectName": str,
    }

    # Defaults
    _rc = {
        "ProjectName": "pylch",
    }

    # Descriptions
    _rst_descriptions = {
        "ProjectRootName": "base Loci/CHEM job name",
    }

    # Subsections
    _sec_cls = {
        "chem": ChemOpts,
    }


# Subsections
RunControlOpts.promote_sections()
# Add properties
RunControlOpts.add_property("ProjectName")
