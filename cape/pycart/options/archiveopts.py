"""
:mod:`cape.pycart.options.archiveopts`: Cart3D archiving options
===================================================================

This module provides Cart3D-specific modifications to the base
archiving options module.

:See also:
    * :mod:`cape.cfdx.options.archiveopts`
    * :class:`cape.cfdx.options.archiveopts.ArchiveOpts`
    * :mod:`cape.manage`
"""

# Local imports
from .util import rc0, INT_TYPES
from ...cfdx.options import archiveopts


# Globs and tarballs to tar most of the time
VizGlob = [
    {'Components.i.stats': 'Components.i.[0-9]*.stats'},
    {'Components.i': 'Components.i.[0-9]*'},
    {'cutPlanes': 'cutPlanes.[0-9]*'},
    {'pointSensors': 'pointSensors.[0-9]*'},
    {'lineSensors': 'lineSensors.[0-9]*'}
]
# Tar adapt
AdaptDict = [{"adapt??": 1}]
# Check files
CheckDict = [
    {"check.?????": 1},
    {"check.??????.td": 2}
]
# Visualization files; keep only recent
VizDict = [
    {"Components.i.*.plt": 1},
    {"Components.i.*.dat": 1},
    {"cutPlanes.*.plt": 1},
    {"cutPlanes.*.dat": 1}
]
# Run files
RunDict = [
    {"run_cart3d.??.pbs": 1},
    {"run.[0-9]*.*": 1},
    {"input.??.cntl": 1},
    {"aero.??.csh": 1}
]
# One-off files
RunFiles = [
    'input.c3d', 'Config.xml', 'jobID.dat',
    'results.dat', 'user_time.dat', 'forces.dat', 'moments.dat',
    'functional.dat', 'loadsCC.dat', 'loadsTRI.dat'
]


# Class for case management
class ArchiveOpts(archiveopts.ArchiveOpts):
    r"""Archiving options interface
    
    :Call:
        >>> opts = ArchiveOpts(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Version 1.0
        * 2016-03-01 ``@ddalle``: Version 1.1; custom settings
        * 2022-10-21 ``@ddalle``: Version 2.0; use :mod:`cape.optdict`
    """
    _optlist = {
        "TarAdapt",
        "TarViz",
        "nCheckPoint",
    }

    _opttypes = {
        "nCheckPoint": INT_TYPES,
    }

    _optvals = {
        "TarAdapt": ("", "tar", "gzip", "bz2", "zip"),
        "TarViz": ("", "tar", "gzip", "bz2", "zip"),
    }

    _rc = {
        "TarAdapt": "tar",
        "TarViz": "tar",
        "nCheckPoint": 2,
    }

    _rst_descriptions = {
        "TarAdapt": "archive format for adapt folders",
        "TarViz": "archive format for visualization files",
        "nCheckPoint": "number of check point files to keep",
    }

    # Initialization method
    def init_post(self):
        r"""Initialization hook for OVERFLOW archiving options

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`ArchiveOpts`
                Archiving options interface
        :Versions:
            * 2022-10-21 ``@ddalle``: Version 1.0
        """
        # Apply the template
        self.apply_ArchiveTemplate()


# Add full options
ArchiveOpts.add_properties(ArchiveOpts._optlist)


# Turn dictionary into Archive options
def auto_Archive(opts):
    r"""Automatically convert dict to pyover :class:`ArchiveOpts`
    
    :Call:
        >>> opts = auto_Archive(opts)
    :Inputs:
        *opts*: :class:`dict`
            Dict of either global, "RunControl" or "Archive" options
    :Outputs:
        *opts*: :class:`ArchiveOpts`
            Instance of archiving options
    :Versions:
        * 2016-02-29 ``@ddalle``: Version 1.0
        * 2022-10-21 ``@ddalle``: Version 2.0; generic call
    """
    return archiveopts.auto_Archive(opts, ArchiveOpts)

