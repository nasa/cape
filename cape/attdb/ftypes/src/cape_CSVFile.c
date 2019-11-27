#include <Python.h>
#include <stdio.h>
#include <math.h>

// Local includes
//#include "cape_CSVFile.h"


// ODE solver for 3DOF orbital EOMs
PyObject *
cape_CSVFileCountLines(PyObject *self, PyObject *args)
{
   // --- Declarations ---
    // Error flag
    int ierr;
    // Line counts
    PyObject *n;
    long nline = 0;
    // File handle
    PyObject *f;
    FILE *fp;
    long pos;
 
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
    // Remember current location
    pos = ftell(fp);
    
    
    
    // Return to original location
    fseek(fp, pos, SEEK_SET);
    // Decrease use count
    PyFile_DecUseCount((PyFileObject *) f);
    
    // Convert to integer
    n = PyInt_FromLong(nline);
    
    // Output
    return n;
}
