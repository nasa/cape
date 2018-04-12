"""
:mod:`pyFun.options.Config`: pyFun configurations options
==========================================================

This module provides options for defining some aspects of the surface
configuration for a FUN3D run.  It can point to a surface configuration file
such as :file:`Config.xml` or :file:`Config.json` that reads an instance of
:class:`cape.config.Config` or :class:`cape.config.ConfigJSON`, respectively.
This surface configuration file is useful for grouping individual components
into families using a format very similar to how they are defined for Cart3D.

The ``"Config"`` section also defines which components are requested by FUN3D
for iterative force & moment history reporting.  For the moment histories, this
section also specifies the moment reference points (moment center in FUN3D
nomenclature) for each component.

This is the section in which the user specifies which components to track
forces and/or moments on, and in addition it defines a moment reference point
for each component.

The reference area (``"RefArea"``) and reference length (``"RefLength"``)
parameters are also defined in this section.  FUN3D does have two separate
reference lengths, so there is also a ``"RefSpan"`` parameter.

Many parameters are inherited from the :class:`cape.config.Config` class, so
readers are referred to that module for defining points by name along with
several other methods.

Like other solvers, the ``"Config"`` section is also used to define the
coordinates of named points.  These can be specified as point sensors to be
reported on directly by FUN3D and/or define named points for other sections of
the JSON file.

:See Also:
    * :mod:`cape.options.Config`
    * :mod:`cape.config`
    * :mod:`pyFun.namelist`
"""


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
   # ------------------
   # Component Mapping
   # ------------------
   # [
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
   # ]
   
   # ------------
   # Other Files
   # ------------
   # [
    # Get template file
    def get_RubberDataFile(self, j=None):
        """Get the ``rubber.data`` file name
        
        :Call:
            >>> fname = opts.get_RubberFile(j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *fname*: :class:`str`
                Name of file template
        :Versions:
            * 2016-04-27 ``@ddalle``: First version
            * 2018-04-11 ``@ddalle``: Moved to *Config* section
        """
        return self.get_key('RubberDataFile', j)
        
    # Get template file
    def get_TDataFile(self, j=None):
        """Get the ``tdata`` file name
        
        :Call:
            >>> fname = opts.get_TDataFile(j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *fname*: :class:`str`
                Name of file template
        :Versions:
            * 2018-04-11 ``@ddalle``: First version
        """
        return self.get_key('TDataFile', j)
    
    # Get thermo data file
    def get_SpeciesThermoDataFile(self, j=None):
        """Get the ``species_thermo_data`` file, if any
        
        :Call:
            >>> fname = opts.get_SpeciesThermoDataFile(j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *fname*: :class:`str`
                Name of file template
        :Versions:
            * 2018-04-12 ``@ddalle``: First version
        """
        return self.get_key('SpeciesThermoDataFile', j)
    
    # Get kinetic data file
    def get_KinetciDataFile(self, j=None):
        """Get the ``kinetic_data`` file, if any
        
        :Call:
            >>> fname = opts.get_KineticDataFile(j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *fname*: :class:`str`
                Name of file template
        :Versions:
            * 2018-04-12 ``@ddalle``: First version
        """
        return self.get_key("KineticDataFile", j)
   # ]
        
   # ----------
   # Points
   # ----------
   # [
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
   # ]
   
# class Config

