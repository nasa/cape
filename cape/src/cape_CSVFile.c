#include <Python.h>
#include <stdio.h>
#include <math.h>

// Local includes
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
    // Column attributes
    Py_ssize_t ncol;
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
        // not a list
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
    }
    
    
    
    
    
    // Output
    Py_RETURN_NONE;
}
