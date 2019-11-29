#include <Python.h>
#include <stdio.h>
#include <string.h>

// Local includes
#include "capec_BaseFile.h"

// Read to end of line
void
capec_FileAdvanceEOL(FILE *fp)
{
    // Buffer
    char buff[80];
    
    // Read white space (80 chars at a time)
    while (fscanf(fp, "%80[^\n]", buff) == 1)
    {
        
    }
    // Read newline character
    getc(fp);
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
    if (!PyString_Check(o)) {
        PyErr_SetString(PyExc_TypeError, "DType object is not a string");
        return -1;
    }
    
    // Otherwise get string from *o*
    dtype = PyString_AsString(o);
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
        dtype = PyString_FromString((const char*) name);
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
            PyInt_FromLong(i));
        // Check for errors
        if (ierr)
            return ierr;
    }
    
    // Save a reference to the newly created list
    Py_INCREF(dtypes);
    // Add the list of names
    return PyModule_AddObject(m, "capeDTYPE_NAMES", dtypes);
}       
    
