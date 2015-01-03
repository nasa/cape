#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdio.h>

// Functions to extract data from a NumPy array
#define np2d(X, i, j) *((double *) PyArray_GETPTR2(X, i, j))
#define np2i(X, i, j) *((int *)    PyArray_GETPTR2(X, i, j))

// Function to write Components.i.tri file
PyObject *
pc_WriteTri(PyObject *self, PyObject *args)
{
    int i, ierr;
    int nNode, nTri;
    FILE *fid;
    PyArrayObject *P;
    PyArrayObject *T;
    
    // Process the inputs.
    if (!PyArg_ParseTuple(args, "OO", &P, &T)) {
        // Check for failure.
        PyErr_SetString(PyExc_RuntimeError, \
            "Could not process inputs to :func:`pc.WriteTri`");
        return NULL;
    }
    
    // Check for two-dimensional Mx3 array.
    if (PyArray_NDIM(P) != 2 || PyArray_DIM(P, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal coordinates must be Nx3 array.");
        return NULL;
    }
    // Read number of nodes.
    nNode = (int) PyArray_DIM(P, 0);
    
    // Check for two-dimensional Nx3 array.
    if (PyArray_NDIM(T) != 2 || PyArray_DIM(T, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal indices must be Nx3 array.");
        return NULL;
    }
    // Read number of nodes.
    nTri = (int) PyArray_DIM(T, 0);
    
    // Open output file for writing (wipe out if it exists.)
    fid = fopen("Components.i.tri", "w");
    
    // Write the number of nodes and tris.
    fprintf(fid, "%12i%12i\n", nNode, nTri);
    
    // Loop through nodal indices.
    for (i=0; i<nNode; i++) {
        // Write a single node.
        fprintf(fid, "%+15.8E %+15.8E %+15.8E\n", \
            np2d(P,i,0), np2d(P,i,1), np2d(P,i,2));
    }
    // Loop through triangles.
    for (i=0; i<nTri; i++) {
        // Write a single triangle.
        fprintf(fid, "%i %i %i\n", \
            np2i(T,i,0), np2i(T,i,1), np2i(T,i,2));
    }
    
    // Close the file.
    ierr = fclose(fid);
    if (ierr) {
        // Failure on close?
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.i.tri'");
        return NULL;
    }
    
    // Return None.
    Py_INCREF(Py_None);
    return Py_None;
}


// Function to write the component IDs
PyObject *
pc_WriteCompID(PyObject *self, PyObject *args)
{
    int ierr;
    
    // Return a number.
    return Py_BuildValue("n", 0);
}
