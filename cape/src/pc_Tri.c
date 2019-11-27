#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pycart_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <byteswap.h>

#include "pc_io.h"
#include "pc_NumPy.h"


// Function to write nodes
int
pc_WriteTriNodes(FILE *fid, PyArrayObject *P)
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
pc_WriteSurfNodes(FILE *fid, PyArrayObject *P, \
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
pc_WriteTriTris(FILE *fid, PyArrayObject *T)
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
pc_WriteSurfTris(FILE *fid, PyArrayObject *T,
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
pc_WriteSurfQuads(FILE *fid, PyArrayObject *Q,
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
pc_WriteTriCompID(FILE *fid, PyArrayObject *C)
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
pc_WriteTriState(FILE *fid, PyArrayObject *Q)
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

// Function to write Components.pyCart.tri file
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
    
    // Check for two-dimensional node array.
    if (PyArray_NDIM(P) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal coordinates must be Nx3 or Nx2 array.");
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
    // Read number of triangles.
    nTri = (int) PyArray_DIM(T, 0);
    
    // Open output file for writing (wipe out if it exists.)
    fid = fopen("Components.pyCart.tri", "w");
    
    // Write the number of nodes and tris.
    fprintf(fid, "%12i%12i\n", nNode, nTri);
    
    // Write the nodes.
    ierr = pc_WriteTriNodes(fid, P);
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure writing nodes to `Components.pyCart.tri'");
    }
    // Write the tris.
    ierr = pc_WriteTriTris(fid, T);
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure writing tris to `Components.pyCart.tri'");
    }
    
    // Close the file.
    ierr = fclose(fid);
    if (ierr) {
        // Failure on close?
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.pyCart.tri'");
        return NULL;
    }
    
    // Return None.
    Py_INCREF(Py_None);
    return Py_None;
}

// Function to write binary tri, single-precision big-endian
PyObject *
pc_WriteTri_b4(PyObject *self, PyObject *args)
{
    int i, ierr=1;
    int nNode, nTri, nb;
    FILE *fid;
    PyArrayObject *P;
    PyArrayObject *T;
    PyArrayObject *C;
    
    // Process the inputs.
    if (!PyArg_ParseTuple(args, "OOO", &P, &T, &C)) {
        // Check for failure.
        PyErr_SetString(PyExc_RuntimeError, \
            "Could not process inputs to :func:`pc.WriteTri_b4`");
        return NULL;
    }
    
    // Read number of nodes and triangles
    nNode = (int) PyArray_DIM(P, 0);
    nTri  = (int) PyArray_DIM(T, 0);
    // Number of bytes in header record
    nb = 2*sizeof(int);
    
    // Open Output file for writing
    fid = fopen("Components.pyCart.tri", "wb");
    
    // Write header record
    pc_Write_b4_i(fid, nb);
    pc_Write_b4_i(fid, nNode);
    pc_Write_b4_i(fid, nTri);
    pc_Write_b4_i(fid, nb);
    
    // Write the nodes, tris, and CompIDs
    ierr = ierr & pc_WriteRecord_b4_f2(fid, P);
    ierr = ierr & pc_WriteRecord_b4_i2(fid, T);
    ierr = ierr & pc_WriteRecord_b4_i1(fid, C);
    
    // Check for errors; error message set elsewhere
    if (ierr) {return NULL; }
    
    // Close the file.
    ierr = fclose(fid);
    // Check for errors.
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.pyCart.tri'");
        return NULL;
    }
    
    // Return None.
    Py_INCREF(Py_None);
    return Py_None;
}

// Function to write binary tri, single-precision little-endian
PyObject *
pc_WriteTri_lb4(PyObject *self, PyObject *args)
{
    int i, ierr=1;
    int nNode, nTri, nb;
    FILE *fid;
    PyArrayObject *P;
    PyArrayObject *T;
    PyArrayObject *C;
    
    // Process the inputs.
    if (!PyArg_ParseTuple(args, "OOO", &P, &T, &C)) {
        // Check for failure.
        PyErr_SetString(PyExc_RuntimeError, \
            "Could not process inputs to :func:`pc.WriteTri_b4`");
        return NULL;
    }
    
    // Read number of nodes and triangles
    nNode = (int) PyArray_DIM(P, 0);
    nTri  = (int) PyArray_DIM(T, 0);
    // Number of bytes in header record
    nb = 2*sizeof(int);
    
    // Open Output file for writing
    fid = fopen("Components.pyCart.tri", "wb");
    
    // Write header record
    pc_Write_lb4_i(fid, nb);
    pc_Write_lb4_i(fid, nNode);
    pc_Write_lb4_i(fid, nTri);
    pc_Write_lb4_i(fid, nb);
    
    // Write the nodes, tris, and CompIDs
    ierr = ierr & pc_WriteRecord_lb4_f2(fid, P);
    ierr = ierr & pc_WriteRecord_lb4_i2(fid, T);
    ierr = ierr & pc_WriteRecord_lb4_i1(fid, C);
    
    // Check for errors; error message set elsewhere
    if (ierr) {return NULL; }
    
    // Close the file.
    ierr = fclose(fid);
    // Check for errors.
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.pyCart.tri'");
        return NULL;
    }
    
    // Return None.
    Py_INCREF(Py_None);
    return Py_None;
}

// Function to write binary tri, double-precision big-endian
PyObject *
pc_WriteTri_b8(PyObject *self, PyObject *args)
{
    int i, ierr=1;
    int nNode, nTri, nb;
    FILE *fid;
    PyArrayObject *P;
    PyArrayObject *T;
    PyArrayObject *C;
    
    // Process the inputs.
    if (!PyArg_ParseTuple(args, "OOO", &P, &T, &C)) {
        // Check for failure.
        PyErr_SetString(PyExc_RuntimeError, \
            "Could not process inputs to :func:`pc.WriteTri_b4`");
        return NULL;
    }
    
    // Read number of nodes and triangles
    nNode = (int) PyArray_DIM(P, 0);
    nTri  = (int) PyArray_DIM(T, 0);
    // Number of bytes in header record
    nb = 2*sizeof(int);
    
    // Open Output file for writing
    fid = fopen("Components.pyCart.tri", "wb");
    
    // Write header record
    pc_Write_b4_i(fid, nb);
    pc_Write_b4_i(fid, nNode);
    pc_Write_b4_i(fid, nTri);
    pc_Write_b4_i(fid, nb);
    
    // Write the nodes, tris, and CompIDs
    ierr = ierr & pc_WriteRecord_b8_f2(fid, P);
    ierr = ierr & pc_WriteRecord_b4_i2(fid, T);
    ierr = ierr & pc_WriteRecord_b4_i1(fid, C);
    
    // Check for errors; error message set elsewhere
    if (ierr) {return NULL; }
    
    // Close the file.
    ierr = fclose(fid);
    // Check for errors.
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.pyCart.tri'");
        return NULL;
    }
    
    // Return None.
    Py_INCREF(Py_None);
    return Py_None;
}

// Function to write binary tri, double-precision little-endian
PyObject *
pc_WriteTri_lb8(PyObject *self, PyObject *args)
{
    int i, ierr=1;
    int nNode, nTri, nb;
    FILE *fid;
    PyArrayObject *P;
    PyArrayObject *T;
    PyArrayObject *C;
    
    // Process the inputs.
    if (!PyArg_ParseTuple(args, "OOO", &P, &T, &C)) {
        // Check for failure.
        PyErr_SetString(PyExc_RuntimeError, \
            "Could not process inputs to :func:`pc.WriteTri_b4`");
        return NULL;
    }
    
    // Read number of nodes and triangles
    nNode = (int) PyArray_DIM(P, 0);
    nTri  = (int) PyArray_DIM(T, 0);
    // Number of bytes in header record
    nb = 2*sizeof(int);
    
    // Open Output file for writing
    fid = fopen("Components.pyCart.tri", "wb");
    
    // Write header record
    pc_Write_lb4_i(fid, nb);
    pc_Write_lb4_i(fid, nNode);
    pc_Write_lb4_i(fid, nTri);
    pc_Write_lb4_i(fid, nb);
    
    // Write the nodes, tris, and CompIDs
    ierr = ierr & pc_WriteRecord_lb8_f2(fid, P);
    ierr = ierr & pc_WriteRecord_lb4_i2(fid, T);
    ierr = ierr & pc_WriteRecord_lb4_i1(fid, C);
    
    // Check for errors; error message set elsewhere
    if (ierr) {return NULL; }
    
    // Close the file.
    ierr = fclose(fid);
    // Check for errors.
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.pyCart.tri'");
        return NULL;
    }
    
    // Return None.
    Py_INCREF(Py_None);
    return Py_None;
}



// Function to write AFLR3 surface file
PyObject *
pc_WriteSurf(PyObject *self, PyObject *args)
{
    int i, ierr;
    int nNode, nTri, nQuad;
    FILE *fid;
    PyArrayObject *P;
    PyArrayObject *T;
    PyArrayObject *CT;
    PyArrayObject *BCT;
    PyArrayObject *Q;
    PyArrayObject *CQ;
    PyArrayObject *BCQ;
    PyArrayObject *blds;
    PyArrayObject *bldel;
    
    // Process the inputs
    if (!PyArg_ParseTuple(args, "OOOOOOOOO", &P, &blds, &bldel,
        &T, &CT, &BCT, &Q, &CQ, &BCQ)){
        // Check for failure.
        PyErr_SetString(PyExc_RuntimeError, \
            "Could not process inputs to :func:`pc.WriteSurf`");
        return NULL;
    }
    // Check for two-dimensional node array.
    if (PyArray_NDIM(P) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal coordinates must be Nx3 or Nx2 array.");
        return NULL;
    }
    // Read number of nodes.
    nNode = (int) PyArray_DIM(P, 0);
    
    // Check for two-dimensional Nx3 array.
    if (PyArray_NDIM(T) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "Triangle nodes must be Nx3 array.");
        return NULL;
    }
    // Read number of triangles.
    nTri = (int) PyArray_DIM(T, 0);
    
    // Check for two-dimensional Nx3 array.
    if (PyArray_NDIM(Q) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "Quad nodes must be Nx4 array.");
        return NULL;
    }
    // Read number of triangles.
    nQuad = (int) PyArray_DIM(Q, 0);
    
    // Open output file for writing (wipe out if it exists.)
    fid = fopen("Components.pyCart.surf", "w");
    
    // Write the number of nodes and tris.
    fprintf(fid, "%12i%12i%12i\n", nTri, nQuad, nNode);
    
    // Write the nodes.
    ierr = pc_WriteSurfNodes(fid, P, blds, bldel);
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure writing nodes to `Components.pyCart.surf'");
    }
    // Write the tris.
    if (nTri > 0) {
        ierr = pc_WriteSurfTris(fid, T, CT, BCT);
        if (ierr) {
            PyErr_SetString(PyExc_IOError, \
                "Failure writing tris to `Components.pyCart.surf'");
        }
    }
    // Write the quads.
    if (nQuad > 0) {
        ierr = pc_WriteSurfQuads(fid, Q, CQ, BCQ);
        if (ierr) {
            PyErr_SetString(PyExc_IOError, \
                "Failure writing quads to `Components.pyCart.surf'");
        }
    }
    
    // Close the file.
    ierr = fclose(fid);
    if (ierr) {
        // Failure on close?
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.pyCart.tri'");
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
    int i, ierr;
    int nTri;
    FILE *fid;
    PyArrayObject *C;
    
    // Process the inputs.
    if (!PyArg_ParseTuple(args, "O", &C)) {
        // Check for failure.
        PyErr_SetString(PyExc_RuntimeError, \
            "Could not process inputs to :func:`pc.WriteCompID`");
        return NULL;
    }
    
    // Open output file for writing (wipe out if it exists.)
    fid = fopen("Components.pyCart.tri", "a");
    
    // Write the nodes.
    ierr = pc_WriteTriCompID(fid, C);
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure writing component IDs to `Components.pyCart.tri'");
    }
    
    // Close the file.
    ierr = fclose(fid);
    if (ierr) {
        // Failure on close?
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.pyCart.tri'");
        return NULL;
    }
    
    // Return None.
    Py_INCREF(Py_None);
    return Py_None;
}


// Function to write Components.pyCart.tri file
PyObject *
pc_WriteTriQ(PyObject *self, PyObject *args)
{
    int i, ierr;
    int nNode, nTri, nq;
    FILE *fid;
    PyArrayObject *P;
    PyArrayObject *T;
    PyArrayObject *C;
    PyArrayObject *Q;
    
    // Process the inputs.
    if (!PyArg_ParseTuple(args, "OOOO", &P, &T, &C, &Q)) {
        // Check for failure.
        PyErr_SetString(PyExc_RuntimeError, \
            "Could not process inputs to :func:`pc.WriteTri`");
        return NULL;
    }
    
    // Check for two-dimensional node array.
    if (PyArray_NDIM(P) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal coordinates must be Nx3 or Nx2 array.");
        return NULL;
    }
    // Read number of nodes.
    nNode = (int) PyArray_DIM(P, 0);
    
    // Check for two-dimensional triangle index array.
    if (PyArray_NDIM(T) != 2 || PyArray_DIM(T, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal indices must be Nx3 array.");
        return NULL;
    }
    // Read number of triangles.
    nTri = (int) PyArray_DIM(T, 0);
    
    // Check for one-dimensional CompIDs
    if (PyArray_NDIM(C) != 1) {
        PyErr_SetString(PyExc_ValueError, \
            "Component IDs must be one-dimensional array.");
        return NULL;
    }
    
    // Check for two-dimensional q array
    if (PyArray_NDIM(Q) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "State array must be two-dimensional.");
        return NULL;
    }
    // Read number of states.
    nq = (int) PyArray_DIM(Q, 1);
    
    // Open output file for writing (wipe out if it exists.)
    fid = fopen("Components.pyCart.tri", "w");
    
    // Write the number of nodes and tris.
    fprintf(fid, "%12i%12i%4i\n", nNode, nTri, nq);
    
    // Write the nodes.
    ierr = pc_WriteTriNodes(fid, P);
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure writing nodes to `Components.pyCart.tri'");
    }
    // Write the tris.
    ierr = pc_WriteTriTris(fid, T);
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure writing tris to `Components.pyCart.tri'");
    }
    // Write the ComponentIDs.
    ierr = pc_WriteTriCompID(fid, C);
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure writing CompIDs to `Components.pyCart.tri'");
    }
    // Write the tris.
    ierr = pc_WriteTriState(fid, Q);
    if (ierr) {
        PyErr_SetString(PyExc_IOError, \
            "Failure writing state to `Components.pyCart.tri'");
    }
    
    // Close the file.
    ierr = fclose(fid);
    if (ierr) {
        // Failure on close?
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.pyCart.tri'");
        return NULL;
    }
    
    // Return None.
    Py_INCREF(Py_None);
    return Py_None;
}


// Function to write Components.pyCart.stl file
PyObject *
pc_WriteTriSTL(PyObject *self, PyObject *args)
{
    int i, ierr;
    int i0, i1, i2;
    int nNode, nTri;
    FILE *fid;
    PyArrayObject *P, *T, *N;
    
    // Process the inputs
    if (!PyArg_ParseTuple(args, "OOO", &P, &T, &N)) {
        // Check for failure.
        PyErr_SetString(PyExc_RuntimeError, \
            "Could not process inputs to :func:`pc.WriteTriSTL`");
        return NULL;
    }
    
    // Check for two-dimensional node array.
    if (PyArray_NDIM(P) != 2) {
        PyErr_SetString(PyExc_ValueError, \
            "Nodal coordinates must be Nx3 or Nx2 array.");
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
    // Read number of triangles.
    nTri = (int) PyArray_DIM(T, 0);
    
    // Check for third two-dimensional Nx3 array.
    if (PyArray_NDIM(T) != 2 || PyArray_DIM(T,1) != 3) {
        PyErr_SetString(PyExc_ValueError, \
            "Normal vectors must be Nx3 array.");
        return NULL;
    }
    
    // Open output file for writing (wipe out if it exists.)
    fid = fopen("Components.pyCart.stl", "w");
    
    // Write the header.
    fprintf(fid, "solid\n");
    
    // Loop through the triangles
    for (i=1; i<=nTri; i++) {
        // Write a single triangle.
        fprintf(fid, "   facet normal   %5.2f %5.2f %5.2f\n", \
            np2d(N,i,0), np2d(N,i,1), np2d(N,i,2));
        // Extract node numbers
        i0 = np2i(T,i,0);
        i1 = np2i(T,i,1);
        i2 = np2i(T,i,2);
        // Write the vertices
        fprintf(fid, "      outer loop\n");
        fprintf(fid, "         vertex   %5.2f %5.2f %5.2f\n", \
            np2d(P,i0,0), np2d(P,i0,1), np2d(P,i0,2));
        fprintf(fid, "         vertex   %5.2f %5.2f %5.2f\n", \
            np2d(P,i1,0), np2d(P,i1,1), np2d(P,i1,2));
        fprintf(fid, "         vertex   %5.2f %5.2f %5.2f\n", \
            np2d(P,i2,0), np2d(P,i2,1), np2d(P,i2,2));
        // Triangle footer
        fprintf(fid, "      endloop\n");
        fprintf(fid, "   endfacet\n");
    }
    
    // Close the file.
    ierr = fclose(fid);
    if (ierr) {
        // Failure on close?
        PyErr_SetString(PyExc_IOError, \
            "Failure on closing file 'Components.pyCart.stl'");
        return NULL;
    }
    
    // Return None.
    Py_INCREF(Py_None);
    return Py_None;
}
