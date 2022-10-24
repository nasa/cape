r"""
:mod:`cape.cfdx.options.rctlopts`: Primary case control options
==================================================================

This module provides a class :class:`RunControlOpts` that defines a CAPE
job's basic phasing and submittal settings. For example, it contains
settings for whether a case is submitted as a PBS job, Slurm job, or run
in the current shell.

It contains the critical options *PhaseSequence* and *PhaseIters*, which
define the number of times number of iterations for each phase for which
the CFD solver will be run.

This portion of the settings interface also includes command-line
options for the various executables that are used in the execution
sequence. It specifies command-line options to grid generation tools
like ``aflr3`` or other preprocessing tools like ``intersect``. The
solver-specific subclasses of :class:`RunControlOpts` also contain the
command-line options to solver commands such as ``flowCart`` for Cart3D.
"""

# Local imports
from . import intersect
from . import ulimit
from .aflr3opts import AFLR3Opts
from .archiveopts import ArchiveOpts
from .util import odict, getel, setel
from ...optdict import ARRAY_TYPES, BOOL_TYPES, INT_TYPES, OptionsDict


# Environment class
class Environ(odict):
    r"""Class for environment variables
    
    :Call:
        >>> opts = Environ(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of environment variables
    :Outputs:
        *opts*: :class:`cape.options.runControl.Environ`
            System environment variable options interface
    :Versions:
        * 2015-11-10 ``@ddalle``: Version 1.0
    """
    
    # Get an environment variable by name
    def get_Environ(self, key, i=0):
        """Get an environment variable setting by name
        
        :Call:
            >>> val = opts.get_Environ(key, i=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *key*: :class:`str`
                Name of the environment variable
            *i*: :class:`int` or ``None``
                Phase number
        :Outputs:
            *val*: :class:`str`
                Value to set the environment variable to
        :Versions:
            * 2015-11-10 ``@ddalle``: Version 1.0
        """
        # Check for the key.
        if key not in self:
            raise KeyError(
                "Environment variable '%s' is not set in JSON file" % key)
        # Get the setting or list of settings
        V = self[key]
        # Select the value for run sequence *i*
        return str(getel(V, i))
        
    # Set an environment variable by name
    def set_Environ(self, key, val, i=None):
        r"""Set an environment variable setting by name
        
        :Call:
            >>> val = opts.get_Environ(key, i=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *key*: :class:`str`
                Name of the environment variable
            *val*: :class:`str`
                Value to set the environment variable to
            *i*: :class:`int` or ``None``
                Phase number
        :Versions:
            * 2015-11-10 ``@ddalle``: Version 1.0
        """
        # Initialize the key if necessary.
        self.setdefault(key, "")
        # Set the value by run sequence.
        self[key] = setel(self[key], str(val), i)
# class Environ


# Class for iteration & mode control settings and command-line inputs
class RunControlOpts(OptionsDict):
    r"""Dictionary-based interface for generic code run control
    
    :Call:
        >>> opts = RunControlOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of run control options
    :Outputs:
        *opts*: :class:`RunControlOpts`
            Basic control options interface
    :Versions:
        * 2014-12-01 ``@ddalle``: Version 1.0
    """
   # ================
   # Class attributes
   # ================
   # <
    # Accepted options
    _optlist = {
        "Archive",
        "Continue",
        "Environ",
        "MPI",
        "PhaseSequence",
        "PhaseIters",
        "PreMesh",
        "Resubmit",
        "Verbose",
        "aflr3",
        "intersect",
        "mpicmd",
        "nIter",
        "nProc",
        "qsub",
        "ulimit",
        "verify",
    }
    
    # Option types
    _opttypes = {
        "Continue": BOOL_TYPES,
        "MPI": BOOL_TYPES,
        "PreMesh": BOOL_TYPES,
        "PhaseIters": INT_TYPES,
        "PhaseSequence": INT_TYPES,
        "Resubmit": BOOL_TYPES,
        "Verbose": BOOL_TYPES,
        "mpicmd": str,
        "nIter": INT_TYPES,
        "nProc": INT_TYPES,
        "qsub": BOOL_TYPES,
        "slurm": BOOL_TYPES,
    }

    # Allowed values
    _optvals = {
        "mpicmd": ("mpiexec", "mpirun"),
    }

    # Defaults
    _rc = {
        "Continue": True,
        "MPI": False,
        "PreMesh": False,
        "Resubmit": False,
        "Verbose": False,
        "qsub": True,
        "slurm": False,
    }

    # Local parameter descriptions
    _rst_descriptions = {
        "Continue": "whether restarts of same phase can use same job",
        "MPI": "whether or not to run MPI in phase",
        "PhaseIters": "check-point iterations for phase *j*",
        "PhaseSequence": "list of phase indices to run",
        "PreMesh": "whether or not to generate volume mesh before submitting",
        "Resubmit": "whether or not to submit new job at end of phase *j*",
        "Verbose": '"RunControl" verbosity flag',
        "mpicmd": "MPI executable name",
        "nIter": "number of iterations to run in phase *j*",
        "nProc": "number of cores/threads to run",
        "qsub": "wheter or not to submit jobs with PBS",
        "sbatch": "wheter or not to submit jobs with Slurm",
    }

    # Sections
    _sec_cls = {
        "Archive": ArchiveOpts,
        "Environ": Environ,
        "aflr3": AFLR3Opts,
        "intersect": intersect.intersect,
        "ulimit": ulimit.ulimit,
        "verify": intersect.verify,
    }
   # >
   
   # =====
   # AFLR3
   # =====
   # <
    # Whether or not to use AFLR3
    def get_aflr3(self):
        r"""Return whether or not to run AFLR3 to create mesh
        
        :Call:
            >>> q = opts.get_aflr3()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *q*: ``True`` | {``False``}
                Whether or not there are nontrivial AFLR3 settings
        :Versions:
            * 2016-04-05 ``@ddalle``: Version 1.0
            * 2022-10-14 ``@ddalle``: Version 1.1; use :func:`bool`
        """
        # Initialize if necessary
        self.init_section(AFLR3Opts, "aflr3")
        # Get the value and type
        v = self.get('aflr3')
        # Get the flag and convert to True or False
        return bool(v.get('run'))
   # >
    
   # =========
   # intersect
   # =========
   # <
    # Whether or not to use intersect
    def get_intersect(self):
        r"""Return whether or not to run ``intersect`` on triangulations
        
        :Call:
            >>> q = opts.get_intersect()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *q*: ``True`` | {``False``}
                Whether or not to run ``intersect``
        :Versions:
            * 2016-04-05 ``@ddalle``: Version 1.0
            * 2022-10-23 ``@ddalle``: Version 1.1; use :func:`bool`
        """
        # Initialize if necessary
        self.init_section(intersect.intersect, "intersect")
        # Get the value and type
        v = self.get("intersect")
        # Get the flag and convert to True or False
        return bool(v.get('run'))
   # >
   
   # ======
   # verify
   # ======
   # <
    # Whether or not to use verify
    def get_verify(self):
        r"""Return whether or not to run ``verify`` on triangulations
        
        :Call:
            >>> q = opts.get_verify()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *q*: ``True`` | {``False``}
                Whether or not to run ``verify``
        :Versions:
            * 2016-04-05 ``@ddalle``: Version 1.0
            * 2022-10-23 ``@ddalle``: Version 1.1; use :func:`bool`
        """
        # Initialize if necessary
        self.init_section(intersect.verify, "verify")
        # Get the value and type
        v = self.get("verify")
        # Get the flag and convert to True or False
        return bool(v.get('run'))
   # >
    
   # ===============
   # Local Functions
   # ===============
   # <
    # Number of phases
    def get_nSeq(self):
        r"""Return the number of phases in the sequence
        
        :Call:
            >>> nSeq = opts.get_nSeq()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *nSeq*: :class:`int`
                Number of input sets in the sequence
        :Versions:
            * 2014-10-02 ``@ddalle``: Version 1.0
            * 2015-02-02 ``@ddalle``: Version 1.1; add *nPhase* override
            * 2022-10-23 ``@ddalle``: Version 2.0; ``OptionsDict``
        """
        # Get the input sequence.
        PhaseSeq = self.get_PhaseSequence()
        # Check if it's a list.
        if isinstance(PhaseSeq, ARRAY_TYPES):
            # Use the length.
            return max(PhaseSeq) + 1
        else:
            # Something is messed up.
            return 1
            
    # Minimum required number of iterations
    def get_LastIter(self):
        r"""Return the minimum number of iterations for case to be done
        
        :Call:
            >>> nIter = opts.get_LastIter()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *nIter*: :class:`int`
                Number of required iterations for case
        :Versions:
            * 2014-10-02 ``@ddalle``: Version 1.0
        """
        return self.get_PhaseIters(self.get_PhaseSequence(-1))
   # >


# Create properties
RunControlOpts.add_properties(RunControlOpts._rst_descriptions)
# Upgrade subsections
RunControlOpts.promote_sections()
