#include <Python.h>
#include <stdio.h>
#include <math.h>

// Local includes
#include "capec_Memory.h"
#include "capec_BaseFile.h"
#include "capec_CSVFile.h"


// Read through file to count data lines
PyObject *
cape_CSVFileCountLines(PyObject *self, PyObject *args)
{
   // --- Declarations ---
    // Line counts
    PyObject *n;
    long nline = 0;
    // File handle
    PyObject *f;
    FILE *fp;
    // File position
    long pos;
    // Strings
    char c;
 
   // --- Inputs ---
    // Parse inputs
    if (!PyArg_ParseTuple(args, "O", &f)) {
        // Failed to read
        PyErr_SetString(PyExc_ValueError, "Failed to parse inputs");
        return NULL;
    }
    
    // Check type
    if (!PyFile_Check(f)) {
        // Not a file
        PyErr_SetString(PyExc_TypeError, "Input is not a file handle");
        return NULL;
    }
    // Convert to native C
    fp = PyFile_AsFile(f);
    // Increment use count
    PyFile_IncUseCount((PyFileObject *) f);
    
   // --- Read ---
    // Get line count
    nline = capec_CSVFileCountLines(fp);
    
   // -- Cleanup ---
    // Decrease use count
    PyFile_DecUseCount((PyFileObject *) f);
    
    // Convert to integer
    n = PyInt_FromLong(nline);
    
    // Output
    return n;
}


// Read CSV file
PyObject *
cape_CSVFileReadData(PyObject *self, PyObject *args)
{
   // --- Declarations ---
    // Error flag
    int ierr;
    // Iterators
    Py_ssize_t i;
    // Data file interface
    PyObject *db;
    PyObject *cols;
    PyObject *col;
    PyObject *j;
    PyObject *dtypes;
    // Column attributes
    Py_ssize_t ncol;
    int *DTYPES;
    long DTYPE;
    // File handle
    PyObject *f;
    FILE *fp;
    
   // --- Inputs ---
    // Parse inputs
    if (!PyArg_ParseTuple(args, "OO", &db, &f)) {
        // Failed to parse
        PyErr_SetString(PyExc_ValueError, "Failed to parse inputs");
        return NULL;
    }
    
    // Check type of *db*: must be dict
    if (!PyDict_Check(db)) {
        // Not a dictionary
        PyErr_SetString(PyExc_TypeError,
            "CSV file object is not an instance of 'dict' class");
        return NULL;
    }
    
    // Get columns
    cols = PyObject_GetAttrString(db, "cols");
    // Check *db.cols* for appropriate types 
    if (cols == NULL) {
        // No columns attribute at all
        PyErr_SetString(PyExc_AttributeError,
            "CSV file object has no 'cols' attribute");
        return NULL;
    } else if (!PyList_Check(cols)) {
        // *db.cols* is not a list
        PyErr_SetString(PyExc_TypeError,
            "CSV file 'cols' attribute is not a list");
        return NULL;
    }
    
    // Get number of columns
    ncol = PyList_GET_SIZE(cols);
    // Check each column
    for (i=0; i<ncol; ++i) {
        // Get column name
        col = PyList_GET_ITEM(cols, i);
        // Check type
        if (!PyString_Check(col)) {
            // *db.cols[i]* is not a string
            // ... watch out for unicode situation
            PyErr_Format(PyExc_TypeError,
                "Column %i is not a string", (int) i);
            return NULL;
        }
    }
    
    // Data types
    dtypes = PyObject_GetAttrString(db, "_c_dtypes");
    // Check *db._c_dtypes*
    if (dtypes == NULL) {
        // No special attribute required to reduce C code
        PyErr_SetString(PyExc_AttributeError,
            "CSV file object has no '_c_dtypes' attribute; "
            "call 'db.get_c_dtypes()' first.");
        return NULL;
    } else if (PyList_GET_SIZE(dtypes) != ncol) {
        // Mismatching length
        PyErr_Format(PyExc_ValueError,
            "_c_dtypes has length %i, but found %i cols",
            (int) PyList_GET_SIZE(dtypes), (int) ncol);
        return NULL;
    }
    
    // Allocate data types integer
    ierr = capec_New1D((void **) &DTYPES, (size_t) ncol, sizeof(int));
    // Check for errors
    if (ierr) {
        PyErr_SetString(PyExc_ValueError,
            "Failed to allocate C DTYPES array");
        return NULL;
    }
    // Loop through entries
    for (i=0; i<ncol; ++i) {
        // Get type
        j = PyList_GET_ITEM(dtypes, i);
        // Check that it's an integer
        if (!PyInt_Check(j)) {
            PyErr_Format(PyExc_TypeError,
                "_c_dtypes[%i] is not an int", (int) i);
            return NULL;
        }
        // Convert to integer
        DTYPE = PyInt_AS_LONG(j);
        // Save it
        DTYPES[i] = (int) DTYPE;
    }
    
    
    // Output
    Py_RETURN_NONE;
}
