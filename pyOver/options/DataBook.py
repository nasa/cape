"""Interface for OVERFLOW data book configuration"""


# Import options-specific utilities
from util import rc0, getel, odict


# Import base class
import cape.options

# Class for DataBook options
class DataBook(cape.options.DataBook):
    """Dictionary-based interface for DataBook specifications
    
    :Call:
        >>> opts = DataBook(**kw)
    :Outputs:
        *opts*: :class:`pyOver.options.DataBook
    :Versions:
        * 2015-12-29 ``@ddalle``: Subclassed from CAPE
    """
    
    # Get MIXSUR file
    def get_DataBook_mixsur(self, comp):
        """Get the ``mixsur`` input file for a databook component
        
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
        db_mixsuri = self.get("mixsur", "mixsur.i")
        # Get component options
        copts = self.get(comp, {})
        # Get the component-specific value
        return copts.get("mixsur", db_mixsuri)
        
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
class DBTarget(cape.options.DBTarget):
    """Dictionary-based interface for databook targets"""
    
    pass


