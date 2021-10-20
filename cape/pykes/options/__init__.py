"""

This module provides tools to read, access, modify, and write settings for
:mod:`cape.pyfun`.  The class is based off of the built-in :class:`dict` class, so
its default behavior, such as ``opts['Namelist']`` or 
``opts.get('Namelist')`` are also present.  In addition, many convenience
methods, such as ``opts.get_project_rootname()``, are also provided.

In addition, this module controls default values of each pyFun
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    1. Values directly specified in the input file, :file:`pyFun.json`
    
    2. Values specified in the default control file,
       ``$PYFUN/settings/pyFun.default.json``
    
    3. Hard-coded defaults from this module

"""

# Local imports
from . import util
from .runcontrol import RunControl
from ...cfdx import options
from ...cfdx.options.pbs         import PBS
from ...cfdx.options.DataBook    import DataBook
from ...cfdx.options.Report      import Report
from ...cfdx.options.Mesh        import Mesh
from ...cfdx.options.Config      import Config
from ...cfdx.options.slurm       import Slurm


# Class definition
class Options(options.Options):
    r"""Options interface for :mod:`cape.pykes`
    
    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Additional manual options
    :Versions:
        * 2021-10-18 ``@ddalle``: Version 1.0
    """
    # Initialization method
    def __init__(self, fname=None, **kw):
        r"""Initialization method"""
        # Check for an input file
        if fname:
            # Read JSON file into a dict
            d = util.loadJSONFile(fname)
        else:
            # No dict
            d = {}
        # Read the defaults.
        defs = util.getPyKesDefaults()
        # Apply the defaults.
        d = util.applyDefaults(d, defs)
        # Store the data in *this* instance
        self.update(d)
        self.update(kw)
        # Initialize sections
        self.init_section(PBS)
        self.init_section(PBS, "BatchPBS", parent="PBS")
        self.init_section(PBS, "PostPBS", parent="PBS")
        # Upgrade important groups to their own classes.
        self.init_section(Slurm)
        self.init_section(DataBook)
        self.init_section(Report)
        self.init_section(RunControl)
        self.init_section(Mesh)
        self.init_section(Config)
        # Add extra folders to path.
        self.AddPythonPath()
    
   # ==============
   # Global Options
   # ==============
   # <
    # Get 
    # Method to get the namelist template
    def get_JobXML(self, j=None):
        r"""Return the name of the main Kestrel XML input file
        
        :Call:
            >>> fname = opts.get_JobXML(j=None)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *j*: :class:`int` or ``None``
                Phase index
        :Outputs:
            *fname*: :class:`str`
                Name of Kestrel XML template file
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        return self.get_key("JobXML", j)
        
    # Method to set the namelist template
    def set_JobXML(self, fname):
        r"""Set the name of the main Kestrel XML input file
        
        :Call:
            >>> opts.set_JobXML(fname)
        :Inputs:
            *opts*: :class:`Options`
                Options interface
            *fname*: :class:`str`
                Name of Kestrel XML template file
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        self["JobXML"] = fname
   # >


util.promote_subsec(Options, RunControl)

