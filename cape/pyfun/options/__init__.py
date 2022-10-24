r"""
:mod:`cape.pyfun.options`: FUN3D options interface module
==========================================================

This module provides the options interface for :mod:`cape.pyfun`. Many
settings are inherited from :mod:`cape.cfdx.options`, and there are
some additional options specific to FUN3D for pyfun.

:See also:
    * :mod:`cape.cfdx.options`

"""

# Import options-specific utilities (loads :mod:`os`, too)
from .util import rc0


# Local imports
from . import util
from .runctlopts import RunControlOpts
from .DataBook    import DataBook
from .Report      import Report
from .fun3dnml    import Fun3DNml
from .Mesh        import Mesh
from .Config      import Config
from .Functional  import Functional
from ...cfdx import options


# Class definition
class Options(options.Options):
    r"""Options interface for :mod:`cape.pyfun`
    
    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary to be transformed into :class:`pyCart.options.Options`
    :Versions:
        * 2014-07-28 ``@ddalle``: Version 1.0
    """
   # ================
   # Class attributes
   # ================
   # <
    # Known option types
    _opttypes = {
    }

    # Option default list depth
    _optlistdepth = {
    }

    # Defaults
    _rc = {
    }

    # Descriptions for methods
    _rst_descriptions = {
    }

    # Section classes
    _sec_cls = {
        "Config": Config,
        "DataBook": DataBook,
        "DualFun3D": Fun3DNml,
        "Fun3D": Fun3DNml,
        "Functional": Functional,
        "Mesh": Mesh,
        "MovingBodyInput": Fun3DNml,
        "Report": Report,
        "RunControl": RunControlOpts,
    }
   # >

   # =============
   # Configuration
   # =============
   # <
    # Initialization hook
    def init_post(self):
        r"""Initialization hook for :class:`Options`

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`Options`
                Options interface
        :Versions:
            * 2022-10-23 ``@ddalle``: Version 1.0
        """
        # Read the defaults
        defs = util.getPyFunDefaults()
        # Apply the defaults.
        self = util.applyDefaults(self, defs)
        # Add extra folders to path.
        self.AddPythonPath()
   # >

   # ==============
   # Global Options
   # ==============
   # <
    # Method to get the namelist template
    def get_FUN3DNamelist(self, j=None):
        """Return the name of the master :file:`fun3d.nml` file
        
        :Call:
            >>> fname = opts.get_FUN3DNamelist(j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *j*: :class:`int` or ``None``
                Run sequence index
        :Outputs:
            *fname*: :class:`str`
                Name of FUN3D namelist template file
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        return self.get_key('Namelist', j)
        
    # Method to set the namelist template
    def set_FUN3DNamelist(self, fname):
        """Set the name of the master :file:`fun3d.nml` file
        
        :Call:
            >>> opts.set_FUN3DNamelist(fname)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of FUN3D namelist template file
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        self['Namelist'] = fname
    
    # Method to determine if groups have common meshes.
    def get_GroupMesh(self):
        """Determine whether or not groups have common meshes
        
        :Call:
            >>> qGM = opts.get_GroupMesh()
        :Inputs:
            *opts* :class:`pyFun.options.Options`
                Options interface
        :Outputs:
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
        """
        # Safely get the trajectory.
        x = self.get('RunMatrix', {})
        return x.get('GroupMesh', rc0('GroupMesh'))
        
    # Method to specify that meshes do or do not use the same mesh
    def set_GroupMesh(self, qGM=rc0('GroupMesh')):
        """Specify that groups do or do not use common meshes
        
        :Call:
            >>> opts.get_GroupMesh(qGM)
        :Inputs:
            *opts* :class:`pyFun.options.Options`
                Options interface
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
        """
        self['RunMatrix']['GroupMesh'] = qGM
   # >


# Add methods from subsections
Options.promote_sections()
