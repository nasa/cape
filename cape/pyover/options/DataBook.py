"""
This module contains the interface for data book options for pyOver and 
OVERFLOW. The classes in this module are subclassed as
    
    * :class:`pyOver.options.DataBook.DataBook` ->
      :class:`cape.cfdx.options.DataBook.DataBook`
      
    * :class:`pyOver.options.DataBook.DBTarget` ->
      :class:`cape.cfdx.options.DataBook.DBTarget`

The OVERFLOW-specific options for these classes are limited, but a few methods
are modified in order to change default data book component types and the
columns of data available for each.  In particular, may special options for
``usurp`` or ``mixsur`` are specified, which is needed to extract a surface
triangulation from an OVERFLOW solution.

"""


# Import options-specific utilities
from .util import rc0, getel, odict


# Import base class
import cape.cfdx.options

# Class for DataBook options
class DataBook(cape.cfdx.options.DataBook):
    """Dictionary-based interface for DataBook specifications
    
    :Call:
        >>> opts = DataBook(**kw)
    :Outputs:
        *opts*: :class:`pyOver.options.dataBook.DataBook`
    :Versions:
        * 2015-12-29 ``@ddalle``: Subclassed from CAPE
    """
    
    # Get MIXSUR file
    def get_DataBook_mixsur(self, comp):
        """Get the ``mixsur`` or ``overint`` input file for a databook component
        
        :Call:
            >>> fname = opts.get_DataBook_mixsur(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fname*: :class:`str`
                Name of ``mixsur`` input file template
        :Versions:
            * 2016-12-29 ``@ddalle``: First version
        """
        # Global data book setting
        db_mixsuri = self.get("mixsur", self.get("overint", "mixsur.i"))
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("mixsur", copts.get("overint", db_mixsuri))
    
    # Get USURP file
    def get_DataBook_usurp(self, comp):
        """Get the ``mixsur`` input file for use with ``usurp``
        
        :Call:
            >>> fname = opts.get_DataBook_usurp(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fname*: :class:`str`
                Name of ``mixsur`` input file template
        :Versions:
            * 2017-04-06 ``@ddalle``: First version
        """
        # Global data book setting
        db_mixsuri = self.get("usurp", "")
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("usurp", db_mixsuri)
        
    # Get SPLITMQ file
    def get_DataBook_splitmq(self, comp):
        """Get the ``splitmq`` input file for a databook component
        
        :Call:
            >>> fname = opts.get_DataBook_splitmq(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fname*: :class:`str`
                Name of ``splitmq`` input file template
        :Versions:
            * 2016-12-29 ``@ddalle``: First version
        """
        # Global data book setting
        db_splitmqi = self.get("splitmq", "splitmq.i")
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("splitmq", db_splitmqi)
        
    # Get path to FOMO files, to avoid running ``mixsur``
    def get_DataBook_fomo(self, comp):
        """Get path to ``mixsur`` output files
        
        If each of the following files is found, there is no need to run
        ``mixsur``, and files are linked instead.
        
            * ``grid.i.tri``
            * ``grid.bnd``
            * ``grid.ib``
            * ``grid.ibi``
            * ``grid.map``
            * ``grid.nsf``
            * ``grid.ptv``
            * ``mixsur.fmp``
            
        :Call:
            >>> fdir = opts.get_DataBook_fomo(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fdir*: {``None``} | :class:`str`
                Path to ``mixsur`` output files
        :Versions:
            * 2017-01-11 ``@ddalle``: First version
        """
        # Global data book setting
        db_fdir = self.get("fomo", None)
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("fomo", db_fdir)
        
        
    # Get *q* file for line loads
    def get_DataBook_QIn(self, comp):
        """Get the input ``q`` file for a databook component
        
        :Call:
            >>> fq = opts.get_DataBook_QIn(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fq*: {``"q.pyover.p3d"``} | :class:`str`
                Name of input Overflow solution file
        :Versions:
            * 2017-01-10 ``@ddalle``: First version
        """
        # Global data book setting
        db_fqo = self.get("QIn", "q.pyover.p3d")
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("QIn", db_fqo)
        
    # Get *q* file for line loads
    def get_DataBook_QOut(self, comp):
        """Get the preprocessed ``q`` file for a databook component
        
        :Call:
            >>> fq = opts.get_DataBook_QOut(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fq*: {``None``} | ``"q.pyover.srf"`` | :class:`str`
                Name of output Overflow solution file
        :Versions:
            * 2017-01-10 ``@ddalle``: First version
        """
        # Global data book setting
        db_fqi = self.get("QOut", None)
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("QOut", db_fqi)
        
    # Get *q* surface file for line loads
    def get_DataBook_QSurf(self, comp):
        """Get the preprocessed ``q.srf`` file name for a databook component
        
        :Call:
            >>> fq = opts.get_DataBook_QSurf(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fq*: ``None`` | {``"q.pyover.srf"``} | :class:`str`
                Name of output Overflow surface solution file
        :Versions:
            * 2017-01-10 ``@ddalle``: First version
        """
        # Global data book setting
        db_fq = self.get("QSurf", "q.pyover.srf")
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("QSurf", db_fq)
        
    # Get *x* file for line loads
    def get_DataBook_XIn(self, comp):
        """Get the input ``x`` file for a databook component
        
        :Call:
            >>> fx = opts.get_DataBook_XIn(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fx*: {``"x.pyover.p3d"``} | :class:`str`
                Name of input Overflow grid file
        :Versions:
            * 2017-01-10 ``@ddalle``: First version
        """
        # Global data book setting
        db_fxi = self.get("XIn", "x.pyover.p3d")
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("XIn", db_fxi)
        
    # Get *x* file for line loads
    def get_DataBook_XOut(self, comp):
        """Get the input ``x`` file for a databook component
        
        :Call:
            >>> fx = opts.get_DataBook_XOut(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fx*: {``None``} | ``"x.pyover.srf"`` | :class:`str`
                Name of output Overflow grid file
        :Versions:
            * 2017-01-10 ``@ddalle``: First version
        """
        # Global data book setting
        db_fxo = self.get("XOut", None)
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("XOut", db_fxo)
        
    # Get *x.srf* file for line loads
    def get_DataBook_XSurf(self, comp):
        """Get the input ``x.srf`` file for a databook component
        
        :Call:
            >>> fx = opts.get_DataBook_XSurf(comp)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load data book component
        :Outputs:
            *fx*: ``None`` | {``"x.pyover.srf"``} | :class:`str`
                Name of output Overflow grid file
        :Versions:
            * 2017-01-10 ``@ddalle``: First version
        """
        # Global data book setting
        db_fx = self.get("XSurf", "x.pyover.srf")
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("XSurf", db_fx)
# class DataBook

        
# Class for target data
class DBTarget(cape.cfdx.options.DBTarget):
    """Dictionary-based interface for databook targets"""
    
    pass


