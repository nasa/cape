"""Interface for OVERFLOW meshing"""

# Import options-specific utilities
from .util import rc0, odict


# Class for FUN3D mesh settings
class Mesh(odict):
    """Dictionary-based interface for OVERFLOW meshing options"""
    
    # Mesh filenames
    def get_MeshFiles(self, config=None):
        """Return the original mesh file names
        
        :Call:
            >>> fname = opts.get_MeshFiles(i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *config*: :class:`str`
                Name of configuration to use (optional)
        :Outputs:
            *fname*: :class:`str` | :class:`list` (:class:`str`)
                Mesh file name or list of files
        :Versions:
            * 2015-12-29 ``@ddalle``: First version
        """
        return self.get_MeshCopyFiles(config) + self.get_MeshLinkFiles(config)
        
    # Mesh filenames to copy
    def get_MeshCopyFiles(self, config=None):
        """Return the names of mesh files to copy
        
        :Call:
            >>> fmsh = opts.get_MeshCopyFiles()
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *config*: :class:`str`
                Name of configuration to use (optional)
        :Outputs:
            *fmsh*: :class:`list` (:class:`str`)
                List of mesh file names to be copied to each case folder
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get mesh type in order to inform defaults
        ftyp = self.get_MeshType(config)
        # Get value, referencing defaults
        if ftyp.lower() == "dcf":
            # Get value with default DCF options
            fmsh = self.get_key("CopyFiles", rck="CopyFilesDCF")
        elif ftyp.lower() == "peg5":
            # Get value with default Pegasus 5 options
            fmsh = self.get_key("CopyFiles", rck="CopyFilesPeg5")
        # Check type
        if type(fmsh).__name__.endswith('dict'):
            # Select the config dir
            return fmsh[config]
        else:
            # Return the config dir regardless of *config*
            return fmsh
        # Ensure list
        if fmsh is None:
            # No files
            return []
        elif type(fmsh).__name__ not in ['list', 'ndarray']:
            # Single file
            return [fmsh]
        else:
            # Return the list
            return fmsh
    
    # Mesh filenames to copy
    def get_MeshLinkFiles(self, config=None):
        """Return the names of mesh files to link
        
        :Call:
            >>> fmsh = opts.get_MeshLinkFiles()
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *config*: :class:`str`
                Name of configuration to use (optional)
        :Outputs:
            *fmsh*: :class:`list` (:class:`str`)
                List of mesh file names to be copied to each case folder
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get mesh type in order to inform defaults
        ftyp = self.get_MeshType(config)
        # Get value, referencing defaults
        if ftyp.lower() == "dcf":
            # Get value with default DCF options
            fmsh = self.get_key("LinkFiles", rck="LinkFilesDCF")
        elif ftyp.lower() == "peg5":
            # Get value with default Pegasus 5 options
            fmsh = self.get_key("LinkFiles", rck="LinkFilesPeg5")
        # Check type
        if type(fmsh).__name__.endswith('dict'):
            # Select the config dir
            return fmsh[config]
        else:
            # Return the config dir regardless of *config*
            return fmsh
        # Ensure list
        if fmsh is None:
            # No files
            return []
        elif type(fmsh).__name__ not in ['list', 'ndarray']:
            # Single file
            return [fmsh]
        else:
            # Return the list
            return fmsh
            
    # Config dir
    def get_ConfigDir(self, config=None):
        """Get configuration directory containing mesh files
        
        :Call:
            >>> fdir = opts.get_ConfigDir()
            >>> fdir = opts.get_ConfigDir(config)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *config*: :class:`str`
                Name of configuration to use (optional)
        :Outputs:
            *fdir*: :class:`str`
                Configuration directory
        :Versions:
            * 2016-02-02 ``@ddalle``: First version
        """
        # Get value
        fcfg = self.get_key("ConfigDir")
        # Check type
        if type(fcfg).__name__.endswith('dict'):
            # Check for an input config
            if config is None:
                # Configuration needed
                raise KeyError(
                    "Multiple mesh configurations defined but none selected")
            elif config not in fcfg:
                # Unrecognized
                raise KeyError("Configuration '%s' not defined" % config)
            # Select the config dir
            return fcfg[config]
        else:
            # Return the config dir regardless of *config*
            return fcfg
            
    # Config dir
    def get_MeshType(self, config=None):
        """Get configuration directory containing mesh files
        
        :Call:
            >>> ftyp = opts.get_ConfigDir()
            >>> ftyp = opts.get_ConfigDir(config)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
            *config*: :class:`str`
                Name of configuration to use (optional)
        :Outputs:
            *ftyp*: :class:`str` | {dcf} | peg5
        :Versions:
            * 2016-02-02 ``@ddalle``: First version
        """
        # Get value
        ftyp = self.get_key("Type", rck="MeshType")
        # Check type
        if type(ftyp).__name__.endswith('dict'):
            # Check for an input config
            if config is None:
                # Configuration needed
                raise KeyError(
                    "Multiple mesh configurations defined but none selected")
            elif config not in ftyp:
                # Unrecognized
                raise KeyError("Configuration '%s' not defined" % config)
            # Select the config dir
            return ftyp[config]
        else:
            # Return the config dir regardless of *config*
            return ftyp
# class Mesh

