#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _cape_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <byteswap.h>

// Local includes
#include "capec_io.h"
#include "capec_NumPy.h"


// Function to write nodes
int
capec_WriteTriNodes(FILE *fid, PyArrayObject *P)
{
    int i;
    int n, nNode, nd;
    
    // Number of values written.
    n = 0;
    
    // Check for two-dimensional Mx3 array.
    if (PyArray_NDIM(P) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal coordinates must be two-dimensional array.");
        return 2;
    }
    // Read number of nodes and dimensionality
    nNode = (int) PyArray_DIM(P, 0);
    nd = (int) PyArray_DIM(P, 1);
    
    // Loop through nodal indices.
    if (nd == 2) {
        // Two-dimensional nodes
        for (i=0; i<nNode; i++) {
            // Write a single node.
            fprintf(fid, "%+15.8E %+15.8E\n", \
                np2d(P,i,0), np2d(P,i,1));
            // Increase the count.
            n += 1;
        }
    }
    else {
        // Three-dimensional nodes
        for (i=0; i<nNode; i++) {
            // Write a single node.
            fprintf(fid, "%+15.8E %+15.8E %+15.8E\n", \
                np2d(P,i,0), np2d(P,i,1), np2d(P,i,2));
            // Increase the count.
            n += 1;
        }
    }
    
    // Check count.
    if (n != nNode) {
        return 1;
    }
    
    // Good output
    return 0;
}


// Function to write SURF file nodes
int
capec_WriteSurfNodes(FILE *fid, PyArrayObject *P, \
    PyArrayObject *blds, PyArrayObject *bldel)
{
    int i;
    int n, nNode, nd;
    
    // Number of values written
    n = 0;
    
    // Check for two-dimensional nNode x 3 array
    if (PyArray_NDIM(P) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal coordinates must be two-dimensional array.");
        return 2;
    }
    // Read number of nodes 
    nNode = (int) PyArray_DIM(P, 0);
    nd = (int) PyArray_DIM(P, 1);
    // Check the other inputs
    if (PyArray_NDIM(bldel) != 1 || PyArray_DIM(bldel,0) != nNode) {
        PyErr_SetString(PyExc_ValueError, \
            "BL depths must be 1D array with one value per node.");
        return 2;
    }
    if (PyArray_NDIM(blds) != 1 || PyArray_DIM(blds,0) != nNode) {
        PyErr_SetString(PyExc_ValueError, \
            "BL spacing must be 1D array with one value per node.");
        return 2;
    }
    
    // Loop through nodal indices
    if (nd == 2) {
        // Two-dimensional nodes
        for (i=0; i<nNode; i++) {
            // Write a single node
            fprintf(fid, "%+15.8E %+15.8E %.4E %.4E\n", \
                np2d(P,i,0), np2d(P,i,1), \
                np1d(blds,i), np1d(bldel,i));
            // Increase the count.
            n += 1;
        }
    }
    else {
        // Three-dimensional nodes
        for (i=0; i<nNode; i++) {
            // Write a single node
            fprintf(fid, "%+15.8E %+15.8E %+15.8E %.4E %.4E\n", \
                np2d(P,i,0), np2d(P,i,1), np2d(P,i,2), \
                np1d(blds,i), np1d(bldel,i));
            // Increase the count.
            n += 1;
        }
    }
    
    // Check count.
    if (n != nNode) {
        return 1;
    }
    
    // Good output
    return 0;
}


// Function to write triangles
int
capec_WriteTriTris(FILE *fid, PyArrayObject *T)
{
    int i;
    int n, nTri;
    
    // Number of values written.
    n = 0;
    
    // Check for two-dimensional Nx3 array.
    if (PyArray_NDIM(T) != 2 || PyArray_DIM(T, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal indices must be Nx3 array.");
        return 2;
    }
    // Read number of triangles.
    nTri = (int) PyArray_DIM(T, 0);
    
    // Loop through triangles.
    for (i=0; i<nTri; i++) {
        // Write a single triangle.
        fprintf(fid, "%i %i %i\n", \
            np2i(T,i,0), np2i(T,i,1), np2i(T,i,2));
        // Increase the count.
        n += 1;
    }
    
    // Check count.
    if (n != nTri) {
        return 1;
    }
    
    // Good output
    return 0;
}


// Function to write AFLR3 SURF triangles, component IDs, and BCs
int
capec_WriteSurfTris(FILE *fid, PyArrayObject *T,
    PyArrayObject *C, PyArrayObject *BC)
{
    int i;
    int n, nTri;
    
    // Number of values written.
    n = 0;
    
    // Check for two-dimensional Nx3 array.
    if (PyArray_NDIM(T) != 2 || PyArray_DIM(T, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, \
            "Triangle nodes must be Nx3 array.");
        return 2;
    }
    // Read number of triangles.
    nTri = (int) PyArray_DIM(T, 0);
    // Check for one-dimensional Mx1 array.
    if (PyArray_NDIM(C) != 1 || PyArray_DIM(C,0) != nTri) {
        PyErr_SetString(PyExc_ValueError, \
            "Face labels must be 1D array with one per tri.");
        return 2;
    }
    // Check for one-dimensional Mx1 array.
    if (PyArray_NDIM(BC) != 1 || PyArray_DIM(BC,0) != nTri) {
        PyErr_SetString(PyExc_ValueError, \
            "Boundary condition tags must be 1D array with one per tri.");
        return 2;
    }
    
    // Loop through triangles
    for (i=0; i<nTri; i++) {
        // Write triangle nodes
        fprintf(fid, "%i %i %i ", np2i(T,i,0), np2i(T,i,1), np2i(T,i,2));
        // Write component ID, reconnect flag (0), and BC
        fprintf(fid, "%i 0 %i\n", np1i(C,i), np1i(BC,i));
        // Increase count.
        n += 1;
    }
    
    // Check count.
    if (n != nTri) {
        return 1;
    }
    
    // Good output
    return 0;
}

// Function to write AFLR3 SURF triangles, component IDs, and BCs
int
capec_WriteSurfQuads(FILE *fid, PyArrayObject *Q,
    PyArrayObject *C, PyArrayObject *BC)
{
    int i;
    int n, nQuad;
    
    // Number of values written.
    n = 0;
    
    // Check for two-dimensional Nx3 array.
    if (PyArray_NDIM(Q) != 2 || PyArray_DIM(Q, 1) != 4) {
        PyErr_SetString(PyExc_ValueError, \
            "Quad nodes must be Nx4 array.");
        return 2;
    }
    // Read number of triangles.
    nQuad = (int) PyArray_DIM(Q, 0);
    // Check for one-dimensional Mx1 array.
    if (PyArray_NDIM(C) != 1 || PyArray_DIM(C,0) != nQuad) {
        PyErr_SetString(PyExc_ValueError, \
            "Face labels must be 1D array with one per tri.");
        return 2;
    }
    // Check for one-dimensional Mx1 array.
    if (PyArray_NDIM(BC) != 1 || PyArray_DIM(BC,0) != nQuad) {
        PyErr_SetString(PyExc_ValueError, \
            "Boundary condition tags must be 1D array with one per tri.");
        return 2;
    }
    
    // Loop through triangles
    for (i=0; i<nQuad; i++) {
        // Write triangle nodes
        fprintf(fid, "%i %i %i %i ",
            np2i(Q,i,0), np2i(Q,i,1), np2i(Q,i,2), np2i(Q,i,3));
        // Write component ID, reconnect flag (0), and BC
        fprintf(fid, "%i 0 %i\n", np1i(C,i), np1i(BC,i));
        // Increase count.
        n += 1;
    }
    
    // Check count.
    if (n != nQuad) {
        return 1;
    }
    
    // Good output
    return 0;
}


// Function to component IDs
int
capec_WriteTriCompID(FILE *fid, PyArrayObject *C)
{
    int i;
    int n, nTri;
    
    // Number of values written.
    n = 0;
    
    // Check for two-dimensional Mx1 array.
    if (PyArray_NDIM(C) != 1) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal coordinates must be one-dimensional array.");
        return 2;
    }
    // Read number of triangles.
    nTri = (int) PyArray_DIM(C, 0);
    
    // Loop through triangles.
    for (i=0; i<nTri; i++) {
        // Write a single triangle.
        fprintf(fid, "%i\n", np1i(C,i));
        // Increase count.
        n += 1;
    }
    
    // Check count.
    if (n != nTri) {
        return 1;
    }
    
    // Good output
    return 0;
}


// Function to write states
int
capec_WriteTriState(FILE *fid, PyArrayObject *Q)
{
    int i, j;
    int n, nNode, nq;
    
    // Number of values written.
    n = 0;
    
    // Check for two-dimensional Nx3 array.
    if (PyArray_NDIM(Q) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "State must be a two-dimensional array.");
        return 2;
    }
    // Read number of triangles.
    nNode = (int) PyArray_DIM(Q, 0);
    // Read number of states.
    nq = (int) PyArray_DIM(Q, 1);
    
    // Loop through triangles.
    for (i=0; i<nNode; i++) {
        // Write a the first entry (Cp).
        fprintf(fid, "%.6f\n", np2d(Q,i,0));
        // Loop through remaining state variables.
        for (j=1; j<nq; j++) {
            // Write the value.
            fprintf(fid, " %.6f", np2d(Q,i,j));
        }
        // End the line.
        fprintf(fid, "\n");
        // Increase the count.
        n += 1;
    }
    
    // Check count.
    if (n != nNode) {
        return 1;
    }
    
    // Good output
    return 0;
}