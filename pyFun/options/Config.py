"""Interface for configuration control: :mod:`pyCart.options.Config`"""


# Import options-specific utilities
from .util import rc0

# Import base class
import cape.options.Config

# Class for PBS settings
class Config(cape.options.Config):
    """
    Configuration options for Fun3D
    
    :Call:
        >>> opts = Config(**kw)
    :Versions:
        * 2015-10-20 ``@ddalle``: First version
    """
    
    # Get inputs for a particular component
    def get_ConfigInput(self, comp):
        """Return the input for a particular component
        
        :Call:
            >>> inp = opts.get_ConfigInput(comp)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
        :Outputs:
            *inp*: :class:`str` | :class:`list` (:class:`int`)
                List of BCs in this component
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        # Get the inputs.
        conf_inp = self.get("Inputs", {})
        # Get the definitions
        return conf_inp.get(comp)
        
    # Set inputs for a particular component
    def set_ConfigInput(self, comp, inp):
        """Set the input for a particular component
        
        :Call:
            >>> opts.set_ConfigInput(comp, nip)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *inp*: :class:`str` | :class:`list` (:class:`int`)
                List of BCs in this component
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        # Ensure the field exists.
        self.setdefault("Inputs", {})
        # Set the value.
        self["Inputs"][comp] = inp
        
    # Boundary point groups
    def get_BoundaryPointGroups(self):
        """Get list of ``"boundary_point"`` geometries
        
        If *Config>BoundaryPointGroups* does not exist, this reads the
        *Config>BoundaryPoints* option and sorts the keys alphabetically.
        
        :Call:
            >>> BP = opts.get_BoundaryPointGroups()
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
        :Outputs:
            *BP*: :class:`list` (:class:`str`)
                List of boundary point groups
        :Versions:
            * 2017-09-01 ``@ddalle``: First version
        """
        # Get the BoundaryPointGroups parameter
        BP = self.get("BoundaryPointGroups")
        # Check if None
        if BP is None:
            # Get the "BoundaryPoints"
            PS = self.get("BoundaryPoints", {})
            # Get the keys
            BP = PS.keys()
            # Sort them.
            BP.sort()
        # Output
        return BP
        
    # Set boundary point groups
    def set_BoundaryPointGroups(self, BP=[]):
        """Set list of ``"boundary_point"`` geometries
        
        :Call:
            >>> pts.set_BoundaryPointGroups(BP=[])
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *BP*: {``[]``} | :class:`list` (:class:`str`)
                List of boundary point groups
        :Versions:
            * 2017-09-01 ``@ddalle``: First version
        """
        # Set it.
        self["BoundaryPointGroups"] = BP
    
    # Boundary points (snapped to boundary)
    def get_BoundaryPoints(self, name=None):
        """Get points for ``boundary_point`` sampling geometry *name*
        
        This corresponds to the namelist parameter
        
            * *sampling_parameters>type_of_geometry*\ (k) = "boundary_points"
            
        It snaps point sensors to the surface.  It requires the namelist
        parameters *number_of_points* and *points* and is modified by
        *snap_output_xyz* and *dist_tolerance*
        
        :Call:
            >>> PS = opts.get_BoundaryPoints(name=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *name*: {``None``} | :class:`str`
                Name of boundary point group (geometry) to process
        :Outputs:
            *PS*: :class:`list` (:class:`list`) | :class:`dict`
                List of points in boundary point group
        :Versions:
            * 2017-09-01 ``@ddalle``: First version
        """
        # Extract list
        PS = self.get("BoundaryPoints", {})
        # Get the boundary point for this group
        if name is None:
            # Return whole dictionary
            return self.expand_Point(PS)
        else:
            # Return individual group
            return self.expand_Point(PS.get(name, []))
            
    # Set boundary points
    def set_BoundaryPoints(self, PS, name=None):
        """Set points for ``boundary_point`` sampling geometry *name*
        
        :Call:
            >>> opts.set_BoundaryPoints(PS, name=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *PS*: :class:`list` (:class:`list`) | :class:`dict`
                List of points in boundary point group
            *name*: {``None``} | :class:`str`
                Name of boundary point group (geometry) to process
        :Versions:
            * 2017-09-01 ``@ddalle``: First version
        """
        # Check for single input
        if name is None:
            # Set the entire property
            self["BoundaryPoints"] = PS
        else:
            # Ensure "BoundaryPoints" exists
            self.setdefault("BoundaryPoints", {})
            # Set the group
            self["BoundaryPoints"][name] = PS
        
# class Config

