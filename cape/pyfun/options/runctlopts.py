r"""
:mod:`cape.pyfun.options.runctlopts`: FUN3D un control options
=================================================================

Options interface for aspects of running a case of FUN3D.  The settings
are read from the ``"RunControl"`` of a JSON file, and the contents of
this section are written to ``case.json`` within each run folder.

The FUN3D-specific options include adaptation settings and command-line
arguments for both ``nodet`` and ``dual``.

:See Also:
    * :mod:`cape.cfdx.options.runctlopts`
    * :mod:`cape.cfdx.options.archiveopts`
    * :mod:`cape.pyfun.options.archiveopts`
"""

# Local imports
from .archiveopts import ArchiveOpts
from ...cfdx.options import runctlopts
from ...cfdx.options.util import ExecOpts
from ...optdict import BOOL_TYPES, INT_TYPES


# Class for `nodet` inputs
class NodetOpts(ExecOpts):
    r"""Class for ``nodet`` settings

    :Call:
        >>> opts = NodetOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`NodetOpts`
            CLI options for ``nodet`` or ``mpi_nodet``
    :Versions:
        * 2015-11-24 ``@ddalle``: Version 1.0 (``nodet``)
        * 2022-11-03 ``@ddalle``: Version 2.0; use ``optdict``
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "animation_freq",
        "plt_tecplot_output",
    }

    # Types
    _opttypes = {
        "animation_freq": INT_TYPES,
        "plt_tecplot_output": BOOL_TYPES,
    }

    # Defaults
    _rc = {
        "animation_freq": -1,
        "plt_tecplot_output": False,
    }

    # Descriptions
    _rst_descriptions = {
        "animation_freq": "animation frequency for ``nodet``",
        "plt_tecplot_output": "option to write ``.plt`` files",
    }


# Add properties
NodetOpts.add_properties(NodetOpts._optlist, prefix="nodet_")


# Class for ``dual`` inputs
class DualOpts(ExecOpts):
    r"""Class for FUN3D ``dual`` settings

    :Call:
        >>> opts = DualOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`DualOpts`
            CLI options for ``dual`` from FUN3D
    :Versions:
        * 2015-11-24 ``@ddalle``: Version 1.0 (``dual``)
        * 2022-11-03 ``@ddalle``: Version 2.0; use ``optdict``
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "adapt",
        "outer_loop_krylov",
        "rad",
    }

    # Types
    _opttypes = {
        "adapt": BOOL_TYPES,
        "outer_loop_krylov": BOOL_TYPES,
        "rad": BOOL_TYPES,
    }

    # Defaults
    _rc = {
        "adapt": True,
        "outer_loop_krylov": True,
        "rad": True,
    }

    # Descriptions
    _rst_descriptions = {
        "adapt": "whether to adapt when running FUN3D ``dual``",
        "outer_loop_krylov": "option to use Krylov method in outer loop",
        "rad": "option to use residual adjoint dot product",
    }


# Add properties
DualOpts.add_properties(DualOpts._optlist, prefix="dual_")


# Class for Report settings
class RunControlOpts(runctlopts.RunControlOpts):
    r"""FUN3D-specific "RunControl" options interface

    :Call:
        >>> opts = RunControl(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of "RunControl" settings
    :Outputs:
        *opts*: :class:`Options`
            Options interface
    :Versions:
        * 2015-09-28 ``@ddalle``: Version 1.0
        * 2022-10-24 ``@ddalle``: Version 2.0
    """
   # =====================
   # Class attributes
   # =====================
   # <
    # Names of allowed settings
    _optlist = {
        "AdaptPhase",
        "Adaptive",
        "Dual",
        "DualPhase",
        "KeepRestarts",
        "WarmStartProject",
        "dual",
        "nIterAdjoint",
        "nodet",
    }

    # Option types
    _opttypes = {
        "AdaptPhase": BOOL_TYPES,
        "Adaptive": BOOL_TYPES,
        "Dual": BOOL_TYPES,
        "DualPhase": BOOL_TYPES,
        "KeepRestarts": BOOL_TYPES,
        "WarmStartProject": str,
        "nIterAdjoint": INT_TYPES,
    }

    # Default values
    _rc = {
        "AdaptPhase": True,
        "Adaptive": False,
        "Dual": False,
        "DualPhase": True,
        "KeepRestarts": False,
        "nIterAdjoint": 200,
        "mpicmd": "mpiexec",
    }

    # Descriptions
    _rst_descriptions = {
        "AdaptPhase": "whether or not to adapt mesh at end of phase",
        "Adaptive": "whether or not to run adaptively",
        "Dual": "whether or not to run all adaptations with adjoint",
        "DualPhase": "whether or not to run phase in dual mode",
        "KeepRestarts": "whether or not to keep restart files",
        "WarmStartProject": "project name in WarmStart source folder",
        "nIterAdjoint": "number of iterations for adjoint solver",
    }

    # Additional sections
    _sec_cls = {
        "Archive": ArchiveOpts,
        "dual": DualOpts,
        "nodet": NodetOpts,
    }

    # Disallow other attributes
    __slots__ = tuple()
   # >

   # ==============
   # Local settings
   # ==============
   # <
    # Get current adaptation number
    def get_AdaptationNumber(self, j=None):
        r"""Get the adaptation number for a given phase

        :Call:
            >>> nadapt = opts.get_AdaptationNumber(j=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *nadapt*: :class:`int` | ``None``
                Number of adaptations prior to phase *j*
        :Versions:
            * 2015-12-31 ``@ddalle``: Version 1.0
            * 2022-10-24 ``@ddalle``: Version 1.1; use *PhaseSequence*
        """
        # Check for adaptive case
        if not (self.get_Adaptive() or self.get_Dual()):
            return None
        elif j is None:
            # No phases
            return None
        # Initialize adaptation number
        nadapt = 0
        # Loop through prior phases
        for k in self.get_PhaseSequence():
            # Exit if we've passed phase *j*
            if k >= j:
                break
            # Check if it's an adaptation phase
            if self.get_nIter(k) > 0 and self.get_AdaptPhase(k):
                nadapt += 1
        # Output
        return nadapt
   # >


# Create properties
RunControlOpts.add_properties(RunControlOpts._rst_descriptions)
# Upgrade subsections
RunControlOpts.promote_sections()
