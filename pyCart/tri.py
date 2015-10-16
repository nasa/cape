"""
Cart3D triangulation module: :mod:`pyCart.tri`
==============================================

This module provides the utilities for interacting with Cart3D triangulations,
including annotated triangulations (including ``.triq`` files).  Triangulations
can also be read from the UH3D format.

The module consists of individual classes that are built off of a base
triangulation class :class:`pyCart.tri.TriBase`.  Methods that are written for
the TriBase class apply to all other classes as well.
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
from cape.geom import RotatePoints, TranslatePoints


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
    pass


