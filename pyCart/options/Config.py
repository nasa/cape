"""Interface for configuration control: :mod:`pyCart.options.Config`"""


# Import options-specific utilities
from util import rc0

# Import base class
import cape.options.Config

# Class for PBS settings
class Config(cape.options.Config):
    """
    Configuration options for Cart3D
    
    :Call:
        >>> opts = Config(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed from CAPE
    """
        
    
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
            *comp*: :class:`str` or :class:`int`
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
            # Return individual line
            return self.expand_Point(LS.get(name))
        else:
            # Return the whole list.
            return self.expand_Point(LS)
            
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
            
            
