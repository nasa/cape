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
        
# class DataBook

        
# Class for target data
class DBTarget(cape.options.DBTarget):
    """Dictionary-based interface for databook targets"""
    
    pass


