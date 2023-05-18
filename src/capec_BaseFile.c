#include <Python.h>
#include <stdio.h>
#include <string.h>

// Local includes
#include "capec_NumPy.h"
#include "capec_PyTypes.h"
#include "capec_BaseFile.h"

// Read to end of line
void
capec_FileAdvanceEOL(FILE *fp)
{
    // Buffer
    char buff[100];
    char *pbuff;
    int n = 0;
    int nc;
    
    // Read white space (80 chars at a time)
    nc = 1;
    while (fgets(buff, sizeof buff, fp) != NULL)
    {
        // Count
        n++;
        // Check if line ends with \n
        nc = strlen(buff);
        printf("Label 0061: n=%i, c='%c'\n", n, buff[nc - 1]);
        if (buff[nc - 1] == '\n') break;
    }
    // Read newline character
    //getc(fp);
}

// Advance past any whitespace (not newline) characters
void
capcec_FileAdvanceWhiteSpace(FILE *fp)
{
    // Buffer
    char buff[80];
    
    // Read white space (80 chars at a time)
    while (fscanf(fp, "%80[ \t\r]", buff) == 1)
    {
        
    }
}

// Process data type code
int capeDTYPE_FromString(const char *dtype)
{
    // Indices
    int i;
    
    // Loop through options
    for (i=0; i<capeDTYPE_Last; ++i)
    {
        // Check comparison
        if (strncmp(capeDTYPE_NAMES[i], dtype, 12) == 0) {
            // Successful match
            return i;
        }
    }
    
    // No match found
    return -1;
}


// Process data type code from Python string
int capeDTYPE_FromPyString(PyObject *o)
{
    // String handler
    char *dtype;
    
    // Check string
    if (!capePyString_Check(o)) {
        PyErr_SetString(PyExc_TypeError, "DType object is not a string");
        return -1;
    }
    
    // Otherwise get string from *o*
    dtype = capePyString_AsString(o);
    // Check it
    if (dtype == NULL) {
        // Error already set
        return -1;
    }
    
    // Otherwise use handler above
    return capeDTYPE_FromString((const char *) dtype);
}

// Add data types to module
int capec_AddDTypes(PyObject *m)
{
    // Error check flag
    int ierr;
    // Index
    long i;
    // Name of current dtype
    char* name;
    char  attr_name[80];
    // List of data type names
    PyObject *dtype;
    PyObject *dtypes;
    
    // Create new list
    dtypes = PyList_New((Py_ssize_t) capeDTYPE_Last);
    // Check it
    if (dtypes == NULL) {
        return -1;
    }
    
    // Loop through names
    for (i=0; i<capeDTYPE_Last; i++)
    {
        // Get current name
        name = capeDTYPE_NAMES[i];
        // Convert the name to a Python object
        dtype = capePyString_FromString((const char*) name);
        // Check for failure
        if (dtype == NULL) {
            return -1;
        }
        // Set the name to the list
        ierr = PyList_SetItem(dtypes, (Py_ssize_t) i, dtype);
        // Check for failure
        if (ierr)
            return ierr;
        // Full name for enumeration
        ierr = sprintf(attr_name, "capeDTYPE_%s", name);
        if (ierr < 0)
            return -1;
        // Add integer code to the module
        ierr = PyModule_AddObject(m, (const char*) attr_name,
            capePyInt_FromLong(i));
        // Check for errors
        if (ierr)
            return ierr;
    }
    
    // Save a reference to the newly created list
    Py_INCREF(dtypes);
    // Add the list of names
    return PyModule_AddObject(m, "capeDTYPE_NAMES", dtypes);
}

// New list or NumPy array by type
PyObject *
capeFILE_NewCol1D(int dtype, size_t n)
{
    // Handle for array
    PyObject *V;
    // Dimensions handle
    npy_intp dims[1] = {(npy_intp) n};
    
    
    // Filter dtype
    if (dtype < 0) {
        PyErr_SetString(PyExc_ValueError, "Negative DTYPE value");
        return NULL;
    } else if (dtype == capeDTYPE_float64) {
        // Create vector of doubles
        V = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    } else if (dtype == capeDTYPE_int32) {
        // Create vector of ints (longs)
        V = PyArray_SimpleNew(1, dims, NPY_INT32);
    } else if (dtype == capeDTYPE_str) {
        // Create a list
        V = PyList_New((Py_ssize_t) n);
    } else if (dtype == capeDTYPE_float16) {
        // Create vector of half floats
        PyErr_Format(PyExc_NotImplementedError,
            "Reading DTYPE '%s' not implemented", capeDTYPE_NAMES[dtype]);
        return NULL;
    } else if (dtype == capeDTYPE_float32) {
        // Create vector of singles
        V = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    } else if (dtype == capeDTYPE_float128) {
        // Create vector of long doubles
        V = PyArray_SimpleNew(1, dims, NPY_FLOAT128);
    } else if (dtype == capeDTYPE_int8) {
        // Create vector of shorts
        V = PyArray_SimpleNew(1, dims, NPY_INT8);
    } else if (dtype == capeDTYPE_int16) {
        // Create vector of ints (regular)
        V = PyArray_SimpleNew(1, dims, NPY_INT16);
    } else if (dtype == capeDTYPE_int64) {
        // Create vector of long doubles
        V = PyArray_SimpleNew(1, dims, NPY_INT64);
    } else if (dtype == capeDTYPE_uint8) {
        // Create vector of shorts
        V = PyArray_SimpleNew(1, dims, NPY_UINT8);
    } else if (dtype == capeDTYPE_uint16) {
        // Create vector of shorts
        V = PyArray_SimpleNew(1, dims, NPY_UINT16);
    } else if (dtype == capeDTYPE_uint32) {
        // Create vector of longs
        V = PyArray_SimpleNew(1, dims, NPY_UINT32);
    } else if (dtype == capeDTYPE_uint64) {
        // Create vector of long longs
        V = PyArray_SimpleNew(1, dims, NPY_UINT64);
    } else {
        PyErr_Format(PyExc_ValueError, "Invalid DTYPE value '%i'", dtype);
        return NULL;
    }
    
    // Check for errors
    if (V == NULL) {
        PyErr_Format(PyExc_SystemError,
            "Failed to allocate object for DTYPE %i, length %li",
            dtype, (long) n);
        return NULL;
    }
    
    // Output
    return V;
}

    
