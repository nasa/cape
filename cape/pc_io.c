#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pycart_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <byteswap.h>

// Macros to extract data from a NumPy array
#define np2d(X, i, j) *((double *) PyArray_GETPTR2(X, i, j))
#define np2f(X, i, j) *((float *)  PyArray_GETPTR2(X, i, j))
#define np2i(X, i, j) *((int *)    PyArray_GETPTR2(X, i, j))
#define np1d(X, i)    *((double *) PyArray_GETPTR1(X, i))
#define np1f(X, i)    *((float *)  PyArray_GETPTR1(X, i))
#define np1i(X, i)    *((int *)    PyArray_GETPTR1(X, i))

// Function to test if system is little-endian
int is_le(void)
{
    int x = 1;
    return *(char*)&x;
}

// Function get size of array (PyArray_SIZE has some issues)
int np_size(PyArrayObject *P)
{
    int j, m, nj;
    int n = 1;
    
    // Get number of dimensions
    m = (int) PyArray_NDIM(P);
    // Loop through dimensions
    for (j=0; j<m; j++) {
        // Multiply the total size
        n *= (int) PyArray_DIM(P, j);
    }
    
    // Output
    return n;
}

// Function to swap a single
float swap_single(const float f)
{
    float v;
    char *F = ( char* ) & f;
    char *V = ( char* ) & v;
    
    // swap the bytes
    V[0] = F[3];
    V[1] = F[2];
    V[2] = F[1];
    V[3] = F[0];
    
    // Output
    return v;
}

// Function to swap a single
double swap_double(const double f)
{
    double v;
    char *F = ( char* ) & f;
    char *V = ( char* ) & v;
    
    // swap the bytes
    V[0] = F[7];
    V[1] = F[6];
    V[2] = F[5];
    V[3] = F[4];
    V[4] = F[3];
    V[5] = F[2];
    V[6] = F[1];
    V[7] = F[0];
    
    // Output
    return v;
}

// Write record of single-precision big-endian integers
int write_record_b4_1i(FILE *fid, PyArrayObject *P)
{
    int i, j;
    int n, nd, nb;
    
    // Check for little-endian system
    int le = is_le();
    
    // Number of dimensions
    nd = (int) PyArray_NDIM(P);
    // Check dims
    if (nd != 1) {
        PyErr_SetString(PyExc_ValueError, \
            "Object must be a 1-dimensional array.");
        return 2;
    }
    // Number of points
    n = (int) PyArray_DIM(P, 0);
    
    // Number of bytes for record marker
    nb = n * sizeof(int);
    // Check for little-endian system
    if (le) {nb = __bswap_32(nb); }
    
    // Record marker
    fwrite(&nb, sizeof(int), 1, fid);
    // Loop through elements
    for (i=0; i<n; i++) {
        // Get value
        if (le) {
            j = __bswap_32(np1i(P,i));
        } else {
            j = np1i(P,i);
        }
        // Write it
        fwrite(&j, sizeof(int), 1, fid);
    }
    // End-of-record marker
    fwrite(&nb, sizeof(int), 1, fid);
    
    return 0;
}
    