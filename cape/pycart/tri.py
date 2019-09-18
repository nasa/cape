"""
:mod:`pyCart.tri`: Cart3D Triangulation Module
==============================================

This module provides the utilities for interacting with Cart3D triangulations,
including annotated triangulations (including ``.triq`` files).  Triangulations
can also be read from the UH3D format (which is similar to the Cart3D ``.tri``
format but with face names for component IDs).

Triangulations can be read from a variety of formats, which are tabulated
below.

    +-------------+------------------------------------------+
    | Extension   | Description                              |
    +=============+==========================================+
    | ``.tri``    | Cart3D tri format, can be ASCII, Fortran |
    |             | unformatted (big- or little-endian), or  |
    |             | C-style stream                           |
    +-------------+------------------------------------------+
    | ``.triq``   | Annotated Cart3D tri format, any format. |
    |             | This format contains a number of state   |
    |             | variables at each point in addition to   |
    |             | the geometry information                 |
    +-------------+------------------------------------------+
    | ``.uh3d``   | UH3D format, similar to ``.tri`` but     |
    |             | names for faces                          |
    +-------------+------------------------------------------+
    | ``.surf``   | AFLR3 surface mesh; can include quads    |
    +-------------+------------------------------------------+
    | ``.unv``    | IDEAS format; very strange               |
    +-------------+------------------------------------------+
    
In addition to the main functions inherited from :class:`cape.tri.TriBase`, the
:class:`pyCart.tri.Triq` class has a few methods that are particular to the
state variables saved in Cart3D.

Cart3D also utilizes the GMP format to associate names with the faces of
component ID numbers.  This is an XML format (standard name for the file is
:file:`Config.xml`) that associates a name for each of the component IDs in a
triangulation.  In addition, groups of component IDs can be given a name by
defining "Parent" components in the XML file.  Reading a triangulation with
such a configuration can be done with the following commands.

    .. code-block:: python
    
        import cape.pycart.tri
        tri = pyCart.tri.Tri("Components.i.tri", c="Config.xml")
        
Note that Cart3D's interpretation of the XML file is very strict.  There cannot
be any component IDs mentioned in the XML file that are not present in the TRI
file.  In addition, there is no way for a component to have multiple parents.
For example, it is not possible to have a group that contains fins 1 and 2
while another group contains fins 1 and 3.

The module consists of individual classes that are built off of a base
triangulation class :class:`cape.tri.TriBase`.  Methods that are written for
the TriBase class apply to all other classes as well.

:See Also:
    * :mod:`cape.tri`
    * :mod:`cape.config`
    * :mod:`pyCart.config`
"""

# Required modules
# File system and operating system management
import os, shutil

# Attempt to load the compiled helper module.
try:
    from . import _pycart as pc
except ImportError:
    pass

# Import base triangulation
import cape.tri

# Rotation functions
from cape.geom import np, RotatePoints, TranslatePoints


# Regular triangulation class
class Tri(cape.tri.Tri):
    """pyCart triangulation class
    
    This class provides an interface for a basic triangulation without
    surface data.  It can be created either by reading an ASCII file or
    specifying the data directly.
    
    When no component numbers are specified, the object created will label
    all triangles ``1``.
    
    :Call:
        >>> tri = pyCart.Tri(fname=fname)
        >>> tri = pyCart.Tri(uh3d=uh3d)
        >>> tri = pyCart.Tri(Nodes=Nodes, Tris=Tris, CompID=CompID)
    :Inputs:
        *fname*: :class:`str`
            Name of triangulation file to read (Cart3D format)
        *uh3d*: :class:`str`
            Name of triangulation file (UH3D format)
        *nNode*: :class:`int`
            Number of nodes in triangulation
        *Nodes*: :class:`np.ndarray` (:class:`float`), (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *nTri*: :class:`int`
            Number of triangles in triangulation
        *Tris*: :class:`np.ndarray` (:class:`int`), (*nTri*, 3)
            Indices of triangle vertex nodes
        *CompID*: :class:`np.ndarray` (:class:`int`), (*nTri*)
            Component number for each triangle
    :Data members:
        *tri.nNode*: :class:`int`
            Number of nodes in triangulation
        *tri.Nodes*: :class:`np.ndarray` (:class:`float`), (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *tri.nTri*: :class:`int`
            Number of triangles in triangulation
        *tri.Tris*: :class:`np.ndarray` (:class:`int`), (*nTri*, 3)
            Indices of triangle vertex nodes
        *tri.CompID*: :class:`np.ndarray` (:class:`int`), (*nTri*)
            Component number for each triangle
    """
    pass
# class Tri

# Regular triangulation class
class Triq(cape.tri.Triq):
    """pyCart triangulation class
    
    This class provides an interface for a basic triangulation without
    surface data.  It can be created either by reading an ASCII file or
    specifying the data directly.
    
    When no component numbers are specified, the object created will label
    all triangles ``1``.
    
    :Call:
        >>> triq = pyCart.Triq(fname=fname)
        >>> triq = pyCart.Triq(Nodes=Nodes, Tris=Tris, CompID=CompID, q=q)
    :Inputs:
        *fname*: :class:`str`
            Name of triangulation file to read (Cart3D format)
        *nNode*: :class:`int`
            Number of nodes in triangulation
        *Nodes*: :class:`np.ndarray` (:class:`float`), (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *nTri*: :class:`int`
            Number of triangles in triangulation
        *Tris*: :class:`np.ndarray` (:class:`int`), (*nTri*, 3)
            Indices of triangle vertex nodes
        *CompID*: :class:`np.ndarray`, (*nTri*)
            Component number for each triangle
        *nq*: :class:`int`
            Number of state variables at each node
        *q*: :class:`np.ndarray` (:class:`float`), (*nNode*, *nq*)
            State vector at each node
    :Data members:
        *triq.nNode*: :class:`int`
            Number of nodes in triangulation
        *triq.Nodes*: :class:`np.ndarray` (:class:`float`), (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *triq.nTri*: :class:`int`
            Number of triangles in triangulation
        *triq.Tris*: :class:`np.ndarray` (:class:`int`), (*nTri*, 3)
            Indices of triangle vertex nodes
        *triq.CompID*: :class:`np.ndarray` (:class:`int`), (*nTri*)
            Component number for each triangle
        *triq.nq*: :class:`int`
            Number of state variables at each node
        *triq.q*: :class:`np.ndarray` (:class:`float`), (*nNode*, *nq*)
            State vector at each node
        *triq.n*: :class:`int`
            Number of files averaged in this triangulation (used for weight)
    """
    
    
    # Apply an angular velocity perturbation
    def ApplyAngularVelocity(self, xcg, w, Lref=1.0, M=None):
        """Perturb surface pressures with a normalized rotation rate
        
        This is based on a method from Stalnaker:
        
            * Stalnaker, J. F. "Rapid Computation of Dynamic Stability
              Derivatives." *42nd AIAA Aerospace Sciences Meeting and Exhibit.*
              2004. AIAA Paper 2004-210. doi:10.2514/6.2004-210
        
        :Call:
            >>> triq.ApplyAngularVelocity(xcg, w, Lref=1.0, M=None)
        :Inputs:
            *triq*: :class:`pyCart.tri.Triq`
                Triangulation instance
            *xcg*: :class:`np.ndarray` | :class:`list`
                Center of gravity or point about which body rotates
            *w*: :class:`np.ndarray` | :class:`list`
                Angular velocity divided by *Lref* times freestream soundspeed
            *Lref*: :class:`float`
                Reference length used for reduced angular velocity
            *M*: :class:`float`
                Freestream Mach number for updating *Cp*
        :Versions:
            * 2016-01-23 ``@ddalle``: First version
        """
        # Calculate area-averaged node normals
        self.GetNodeNormals()
        # Make center of gravity vector
        Xcg = np.repeat([xcg], self.nNode, axis=0)
        # Calculate angular velocities at each node
        V = np.cross(self.Nodes-Xcg, w)
        # Calculate dot product of surface normals and nodal velocities
        Vn = np.sum(V*self.NodeNormals, 1)
        # Local speed of sound (with minimum pressure 0.001)
        an = np.sqrt(self.q[:,1] / (1.4*np.fmax(self.q[:,5], 0.001)))
        # Divide by local speed of sound
        Mn = Vn / an
        # Calculate pressure ratio
        pr = np.fmax(1-0.2*Mn, 0.0) ** 7.0
        # Save the updated state
        self.q[:,5] *= pr
        # Update the pressure coefficient
        if M is not None:
            self.q[:,0] = (self.q[:,5] - 1/1.4) / (0.5*M*M)
            
    # Apply an angular velocity perturbation
    def ApplyAngularVelocityLinear(self, xcg, w, Lref=1.0, M=None):
        """Perturb surface pressures with a normalized rotation rate
        
        This is based on a method from Stalnaker:
        
            * Stalnaker, J. F. "Rapid Computation of Dynamic Stability
              Derivatives." *42nd AIAA Aerospace Sciences Meeting and Exhibit.*
              2004. AIAA Paper 2004-210. doi:10.2514/6.2004-210
        
        :Call:
            >>> triq.ApplyAngularVelocityLinear(xcg, w, Lref=1.0, M=None)
        :Inputs:
            *triq*: :class:`pyCart.tri.Triq`
                Triangulation instance
            *xcg*: :class:`np.ndarray` | :class:`list`
                Center of gravity or point about which body rotates
            *w*: :class:`np.ndarray` | :class:`list`
                Angular velocity divided by *Lref* times freestream soundspeed
            *Lref*: :class:`float`
                Reference length used for reduced angular velocity
            *M*: :class:`float`
                Freestream Mach number for updating *Cp*
        :Versions:
            * 2016-01-23 ``@ddalle``: First version
        """
        # Calculate area-averaged node normals
        self.GetNodeNormals()
        # Make center of gravity vector
        Xcg = np.repeat([xcg], self.nNode, axis=0)
        # Calculate angular velocities at each node
        V = np.cross(self.Nodes-Xcg, w*Lref)
        # Calculate dot product of surface normals and nodal velocities
        Vn = np.sum(V*self.NodeNormals, 1)
        # Calculate pressure ratio
        dp = - np.sqrt(1.4*self.q[:,1]*self.q[:,5])*Vn
        # Save the updated state
        self.q[:,5] = np.fmax(self.q[:,5]+dp, 0.0)
        # Update the pressure coefficient
        if M is not None:
            self.q[:,0] = (self.q[:,5] - 1/1.4) / (0.5*M*M)
# class Triq

