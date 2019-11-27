#ifndef _PC_NUMPY_H
#define _PC_NUMPY_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pycart_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

// Macros to extract data from a NumPy array
#define np3d(X, i, j, k) *((double *) PyArray_GETPTR3(X, i, j, k))
#define np3f(X, i, j, k) *((float *)  PyArray_GETPTR3(X, i, j, k))
#define np3i(X, i, j, k) *((int *)    PyArray_GETPTR3(X, i, j, k))
#define np3s(X, i, j, k) *((short *)  PyArray_GETPTR3(X, i, j, k))
#define np2d(X, i, j) *((double *) PyArray_GETPTR2(X, i, j))
#define np2f(X, i, j) *((float *)  PyArray_GETPTR2(X, i, j))
#define np2i(X, i, j) *((int *)    PyArray_GETPTR2(X, i, j))
#define np2s(X, i, j) *((short *)  PyArray_GETPTR2(X, i, j))
#define np1d(X, i) *((double *) PyArray_GETPTR1(X, i))
#define np1f(X, i) *((float *)  PyArray_GETPTR1(X, i))
#define np1i(X, i) *((int *)    PyArray_GETPTR1(X, i))
#define np1s(X, i) *((short *)  PyArray_GETPTR1(X, i))

#endif // _PC_NUMPY_H
