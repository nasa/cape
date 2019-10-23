"""
:mod:`cape.pycart.options.Config`: pyCart Configuration Options
===============================================================

This module provides options for defining some aspects of the surface
configuration for a Cart3D run.  In addition to specifying a template
:file:`Config.xml` file for naming individual components or groups of
components, the ``"Config"`` section also contains user-defined points and a
set of parameters that are written to the Cart3D input file :file:`input.cntl`.

This is the section in which the user specifies which components to track
forces and/or moments on, and in addition it defines a moment reference point
for each component.

The reference area (``"RefArea"``) and reference length (``"RefLength"``)
parameters are also defined in this section.  Cart3D does not have two separate
reference lengths, so there is no ``"RefSpan"`` parameter.

Most parameters are inherited from the :class:`cape.config.Config` class, so
readers are referred to that module for defining points by name along with
several other methods.

The ``"Xslices"``, ``"Yslices"``, and ``"Zslices"`` parameters are used by
Cart3D to define the coordinates at which Tecplot slices are extracted.  These
are written to the output file :file:`cutPlanes.plt` or :file:`cutPlanes.dat`.
Furthermore, these coordinates can be tied to user-defined points that may vary
for each case in a run matrix.

The following example has a user-defined coordinate for a point that is on a
component called ``"Fin2"``.  Assuming there is a trajectory key that rotates
``"Fin2"``, the *y*-slice and *z*-slice coordinates will automatically be
updated according to the fin position.

    .. code-block:: javascript
    
        "Config": {
            // Define a point at the tip of a fin
            "Points": {
                "Fin2": [7.25, -0.53, 0.00]
            },
            // Write a *y* and *z* slice through the tip of the fin
            "Yslices": ["Fin2"],
            "Zslices": ["Fin2"]
        }
        
Cart3D also contains the capability for point sensors, which record the state
variables at a point, and line sensors, which provide the state at several
points along a line.  The line sensors are particularly useful for extracting
a sonic boom signature.  Both point sensors and line sensors can also be used
as part of the definition of an objective function for mesh refinement.  In
addition, the sensors can be used to extract not only the conditions at the
final iteration but also the history of relevant conditions at each iteration.

:See Also:
    * :mod:`cape.cfdx.options.Config`
    * :mod:`cape.config`
    * :mod:`cape.pycart.inputCntl`
"""


# Import options-specific utilities
from .util import rc0

# Import base class
import cape.cfdx.options.Config

# Class for PBS settings
class Config(cape.cfdx.options.Config):
    """Component configuration options for Cart3D"""
    
    # Get the list of forces to report
    def get_ClicForces(self, i=None):
        """Get force or list of forces requested to track using `clic`
        
        :Call:
            >>> comp = opts.get_ClicForces(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: ``None`` | :class:`int` | :class:`list`(:class:`int`)
                Index of component in list to return
        :Outputs:
            *comp*: :class:`str` | :class:`int` | :class:`list`
                Component or list of components to track with `clic`
        :Versions:
            * 2014-12-08 ``@ddalle``: First version
        """
        # Get the values.
        comp = self.get_key('Force', i)
        # Check to make sure it's a list.
        if (i is None) and (type(comp).__name__!="list"):
            # Create simple list.
            comp = [comp]
        # Output
        return comp
        
    # Set a clic force
    def set_ClicForces(self, comp='entire', i=None):
        """Set force or list of forces by index to track using `clic`
        
        :Call:
            >>> opts.set_ClicForces(comp, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` | :class:`int` | :class:`list`
                Component or list of components to track with `clic`
            *i*: ``None`` | :class:`int` | :class:`list`(:class:`int`)
                Index of component in list to return
        :Versions:
            * 2014-12-08 ``@ddalle``: First version
        """
        self.set_key('Force', comp, i)
    
    # Add an additional clic force
    def add_ClicForce(self, comp):
        """Add a component to track using `clic`
        
        :Call:
            >>> opts.add_ClicForce(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` | :class:`int`
                Component or list of components
        :Versions:
            * 2014-12-08 ``@ddalle``: First version
        """
        # Get the current list.
        comps = self.get_ClicForce()
        # Set the input value as an addendum to the list.
        self.set_key('Force', comp, len(comps))
    
    
    # Get cut plane extraction coordinate(s)
    def get_Xslices(self, i=None):
        """Return the list of Xslices for extracting solution cut planes
        
        :Call:
            >>> x = opts.get_Xslices()
            >>> x = opts.get_Xslices(i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Outputs:
            *x*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the values.
        Xslices = self.get_key('Xslices', i)
        # Check for required list.
        if (i is None) and (type(Xslices).__name__ not in ['list','array']):
            # Convert scalar to list
            Xslices = [Xslices]
        # Current output type
        typx = type(Xslices).__name__
        # Output
        if typx in ['list','array']:
            # Loop through points.
            for j in range(len(Xslices)):
                # Type
                typj = type(Xslices[j]).__name__
                # Check type.
                if typj.startswith('str') or typj == 'unicode':
                    # Convert the point
                    Xslices[j] = self.get_Point(Xslices[j])[0]
        elif typx.startswith('str') or typx=='unicode':
            # Convert the point.
            Xslices = self.get_Point(Xslices)[0]
        # Output
        return Xslices
        
    # Set cut plane extraction coordinate(s)
    def set_Xslices(self, x, i=None):
        """Return the list of Xslices for extracting solution cut planes
        
        :Call:
            >>> x = opts.set_Xslices(x)
            >>> x = opts.set_Xslices(x, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *x*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('Xslices', x, i)
        
    # Add an additional cut plane
    def add_Xslice(self, x):
        """Add a cutting plane to the current list
        
        :Call:
            >>> opts.add_Xslice(x)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *x*: :class:`float`
                Cut plane coordinate
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the current list.
        Xslices = self.get_key('Xslices')
        # Set the input value as an addendum to the list.
        self.set_key('Xslices', x, len(Xslices))
        
        
    # Get cut plane extraction coordinate(s)
    def get_Yslices(self, i=None):
        """Return the list of Yslices for extracting solution cut planes
        
        :Call:
            >>> y = opts.get_Yslices()
            >>> y = opts.get_Yslices(i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Outputs:
            *y*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        Yslices = self.get_key('Yslices', i)
        # Check for required list.
        if (i is None) and (type(Yslices).__name__ not in ['list','array']):
            # Convert scalar to list
            Yslices = [Yslices]
        # Current output type
        typy = type(Yslices).__name__
        # Output
        if typy in ['list','array']:
            # Loop through points.
            for j in range(len(Yslices)):
                # Type
                typj = type(Yslices[j]).__name__
                # Check type.
                if typj.startswith('str') or typj == 'unicode':
                    # Convert the point
                    Yslices[j] = self.get_Point(Yslices[j])[1]
        elif typy.startswith('str') or typy=='unicode':
            # Convert the point.
            Yslices = self.get_Point(Yslices)[1]
        # Output
        return Yslices
        
    # Set cut plane extraction coordinate(s)
    def set_Yslices(self, y, i=None):
        """Return the list of Yslices for extracting solution cut planes
        
        :Call:
            >>> y = opts.set_Yslices(x)
            >>> y = opts.set_Yslices(x, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *y*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('Yslices', y, i)
        
    # Add an additional cut plane
    def add_Yslice(self, y):
        """Add a cutting plane to the current list
        
        :Call:
            >>> opts.add_Yslice(x)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *y*: :class:`float`
                Cut plane coordinate
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the current list.
        Yslices = self.get_key('Yslices')
        # Set the input value as an addendum to the list.
        self.set_key('Yslices', y, len(Yslices))
        
        
    # Get cut plane extraction coordinate(s)
    def get_Zslices(self, i=None):
        """Return the list of Zslices for extracting solution cut planes
        
        :Call:
            >>> z = opts.get_Zslices()
            >>> z = opts.get_Zslices(i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Outputs:
            *z*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        Zslices = self.get_key('Zslices', i)
        # Check for required list.
        if (i is None) and (type(Zslices).__name__ not in ['list','array']):
            # Convert scalar to list
            Zslices = [Zslices]
        # Current output type
        typz = type(Zslices).__name__
        # Output
        if typz in ['list','array']:
            # Loop through points.
            for j in range(len(Zslices)):
                # Type
                typj = type(Zslices[j]).__name__
                # Check type.
                if typj.startswith('str') or typj == 'unicode':
                    # Convert the point
                    Zslices[j] = self.get_Point(Zslices[j])[2]
        elif typx.startswith('str') or typx=='unicode':
            # Convert the point.
            Zslices = self.get_Point(Zslices)[2]
        # Output
        return Zslices
        
    # Set cut plane extraction coordinate(s)
    def set_Zslices(self, z, i=None):
        """Return the list of Zslices for extracting solution cut planes
        
        :Call:
            >>> x = opts.set_Zslices(z)
            >>> x = opts.set_Zslices(z, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *z*: :class:`float` or :class:`list`(:class:`float`)
                Cut plane coordinate(s)
            *i*: :class:`int`
                Index of cut plane coordinate to extract
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('Zslices', z, i)
        
    # Add an additional cut plane
    def add_Zslice(self, z):
        """Add a cutting plane to the current list
        
        :Call:
            >>> opts.add_Zslice(z)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *z*: :class:`float`
                Cut plane coordinate
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the current list.
        Zslices = self.get_key('Zslices')
        # Set the input value as an addendum to the list.
        self.set_key('Zslices', z, len(Zslices))
        
    
    # Get line sensors
    def get_LineSensors(self, name=None):
        """Get dictionary of line sensors
        
        :Call:
            >>> LS = opts.get_LineSensors(name=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *name*: :class:`str`
                Name of individual line sensor to extract
        :Outputs:
            *LS*: :class:`dict` (:class:`dict`)
                Line sensor or dictionary of line sensors
        :Versions:
            * 2015-05-06 ``@ddalle``: First version
        """
        # Extract list.
        LS = self.get('LineSensors', {})
        # Check for an individual name.
        if name is not None:
            # Get individual line
            LS = LS.get(name)
        else:
            # Expand all
            LSE = {}
            # Loop through keys
            for k in LS:
                # Expand that line sensor
                LSE[k] = self.get_LineSensors(k)
            # Output
            return LSE
        # Length of LineSensor definition
        n = len(LS)
        # Check for special case where all coordinates are specified
        if n == 6:
            # Split points
            return [LS[:3], LS[3:]]
        elif n == 0:
            # No points
            return LS
        # Expand points
        return [self.expand_Point(LS[0]), self.expand_Point(LS[1])]
            
            
    # Set line sensors
    def set_LineSensors(self, LS={}, name=None, X=[]):
        """Set dictionary of line sensors or individual line sensor
        
        :Call:
            >>> opts.set_LineSensors(LS={})
            >>> opts.set_LineSensors(name=None, X=[])
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *name*: :class:`str`
                Name of individual line sensor to set
            *X*: :class:`list` (:class:`double`, *len*\ =6)
                List of start x,y,z and end x,y,z
            *LS*: :class:`dict` (:class:`dict`)
                Line sensor or dictionary of line sensors
        :Versions:
            * 2015-05-06 ``@ddalle``: First version
        """
        # Check version of the call.
        if name is not None:
            # Initialize line sensors if necessary.
            self.setdefault('LineSensors', {})
            # Set the line sensor.
            self['LineSensors'][name] = X
        else:
            # Set the full list.
            self['LineSensors'] = LS
            
    # Add an additional line sensor.
    def add_LineSensor(self, name, X):
        """Add an additional line sensor
        
        :Call:
            >>> opts.add_LineSensor(name, X)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *name*: :class:`str`
                Name of individual line sensor to extract
            *X*: :class:`list` (:class:`double`, *len*\ =6)
                List of start x,y,z and end x,y,z
        :Versions:
            * 2015-05-06 ``@ddalle``: First version
        """
        # Initialize line sensors if necessary.
        self.setdefault('LineSensors', {})
        # Set the line sensor.
        self['LineSensors'][name] = X
        
        
    # Get point sensors
    def get_PointSensors(self, name=None):
        """Get dictionary of point sensors
        
        :Call:
            >>> PS = opts.get_PointSensors(name=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *name*: :class:`str`
                Name of individual line sensor to extract
        :Outputs:
            *PS*: :class:`dict` (:class:`dict`)
                Point sensor or dictionary of point sensors
        :Versions:
            * 2015-05-07 ``@ddalle``: First version
        """
        # Extract list.
        PS = self.get('PointSensors', {})
        # Check for an individual name.
        if name is not None:
            # Return individual line
            return self.expand_Point(PS.get(name))
        else:
            # Return the whole list.
            return self.expand_Point(PS)
            
    # Set point sensors
    def set_PointSensors(self, PS={}, name=None, X=[]):
        """Set dictionary of point sensors or individual point sensor
        
        :Call:
            >>> opts.set_PointSensors(LS={})
            >>> opts.set_PointSensors(name=None, X=[])
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *name*: :class:`str`
                Name of individual line sensor to set
            *X*: :class:`list` (:class:`double`, *len*\ =3)
                List of point x,y,z coordinates
            *PS*: :class:`dict` (:class:`dict`)
                Point sensor or dictionary of point sensors
        :Versions:
            * 2015-05-07 ``@ddalle``: First version
        """
        # Check version of the call.
        if name is not None:
            # Initialize point sensors if necessary.
            self.setdefault('PointSensors', {})
            # Set the point sensor.
            self['PointSensors'][name] = X
        else:
            # Set the full list.
            self['PointSensors'] = PS
            
    # Add an additional point sensor.
    def add_PointSensor(self, name, X):
        """Add an additional point sensor
        
        :Call:
            >>> opts.add_PointSensor(name, X)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *name*: :class:`str`
                Name of individual line sensor to extract
            *X*: :class:`list` (:class:`double`, *len*\ =3)
                List of point x,y,z sensors
        :Versions:
            * 2015-05-07 ``@ddalle``: First version
        """
        # Initialize point sensors if necessary.
        self.setdefault('PointSensors', {})
        # Set the point sensor.
        self['PointSensors'][name] = X
            
            
