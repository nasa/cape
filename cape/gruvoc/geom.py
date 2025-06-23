r"""
:mod:`gruvoc.geom`: Geometry manipulation tools for ``gruvoc``
================================================================

This module provides a series of functions that perform basic geometry
operations, such as :func:`rotate_points` to rotate an array of points
about a vector specified as two end points.
"""

# Standard library
from typing import Union

# Third-party
import numpy as np

# Local imports
from .errors import assert_nd, assert_shape


# Constants
DEG = np.pi / 180.0

# Type annotations
Array = Union[list, tuple, np.ndarray]
Float = Union[float, np.float32, np.float64]


def rotate_points(
        X: Array,
        v1: Array,
        v2: Array,
        theta: Float) -> np.ndarray:
    r"""Rotate an array of points about a vector defined by two points

    :Call:
        >>> Y = rotate_points(X, v1, v2, theta)
    :Inputs:
        *X*: :class:`np.ndarray`\ [:class:`float`]
            Coordinates of *N* points to rotate; shape is (*N*, 3)
        *v1*: :class:`np.ndarray`\ [:class:`float`]
            Coordinates of start point of rotation vector; shape (3,)
        *v2*: :class:`np.ndarray`\ [:class:`float`]
            Coordinates of end point of rotation vector; shape (3,)
        *theta*: :class:`float`
            Angle by which to rotate *X*, in degrees
    :Outputs:
        *Y*: :class:`np.ndarray`\ [:class:`float`]
            Rotated points
    """
    # Ensure list of points
    X = np.array(X, ndmin=2)
    # Convert to vector
    v1 = np.array(v1)
    v2 = np.array(v2)
    # Check shape (Nx3)
    assert_nd(X, 2, "points array")
    assert_shape(X, 3, 1, name="points array")
    # Check shape (3,)
    assert_nd(v1, 1, "start of rotation vector")
    assert_nd(v2, 1, "end of rotation vector")
    assert_shape(v1, 3, 0, "start of rotation vector")
    assert_shape(v2, 3, 0, "end of rotation vector")
    # Function w/o checks
    return _rotate_points(X, v1, v2, theta)


def translate_points(
        X: Array,
        v: Array) -> np.ndarray:
    r"""Translate an array of points

    :Call:
        >>> Y = translate_points(X, v)
    :Inputs:
        *X*: :class:`np.ndarray`\ [:class:`float`]
            Coordinates of *N* points to translate; shape is (*N*, 3)
        *v*: :class:`np.ndarray`\ [:class:`float`]
            Vector by which to translate each point; shape (3,)
    :Outputs:
        *Y*: :class:`np.ndarray`\ [:class:`float`]
            Translated points
    """
    # Ensure list of points
    X = np.array(X, ndmin=2)
    # Convert to vector
    v1 = np.array(v)
    # Check shape (Nx3)
    assert_nd(X, 2, "points array")
    assert_shape(X, 3, 1, name="points array")
    # Check shape (3,)
    assert_nd(v1, 1, "translation vector")
    assert_shape(v1, 3, 0, "translation vector")
    # Function w/o checks
    return _rotate_points(X, v)


def _rotate_points(
        X: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        theta: Float) -> np.ndarray:
    # Extract the coordinates and shift origin.
    x = X[:, 0] - v1[0]
    y = X[:, 1] - v1[1]
    z = X[:, 2] - v1[2]
    # Make the rotation vector
    v = (v2-v1) / np.linalg.linalg.norm(v2-v1)
    # Dot product of points with rotation vector
    k1 = v[0]*x + v[1]*y + v[2]*z
    # Trig functions
    c_th = np.cos(theta*DEG)
    s_th = np.sin(theta*DEG)
    # Initialize output
    Y = np.zeros_like(X)
    # Apply Rodrigues' rotation formula to get the rotated coordinates.
    Y[:, 0] = x*c_th+(v[1]*z-v[2]*y)*s_th+v[0]*k1*(1-c_th)+v1[0]
    Y[:, 1] = y*c_th+(v[2]*x-v[0]*z)*s_th+v[1]*k1*(1-c_th)+v1[1]
    Y[:, 2] = z*c_th+(v[0]*y-v[1]*x)*s_th+v[2]*k1*(1-c_th)+v1[2]
    # Output
    return Y


def _translate_points(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""Translate points, w/o input checks"""
    # Initialize output
    Y = np.zeros_like(X)
    # Translate each coordiante
    Y[:, 0] = X[:, 0] + v[0]
    Y[:, 1] = X[:, 1] + v[1]
    Y[:, 2] = X[:, 2] + v[2]
    # Output
    return Y

