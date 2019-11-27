#include <Python.h>
#include <stdio.h>
#include <math.h>

// Local includes
#include "capec_BaseFile.h"


// Read through file to count data lines
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
    // Remember current location
    pos = ftell(fp);
    
    // Read lines
    while (!feof(fp)) {
        // Read spaces
        capcec_FileAdvanceWhiteSpace(fp);
        // Get next character
        c = getc(fp);
        // Check it
        if (c == '\n') {
            // Empty line
            continue;
        } else if (feof(fp)) {
            // Last line
            break;
        }
        // Read to end of line (comment or not)
        capec_FileAdvanceEOL(fp);
        // Check if it was a comment
        if (c != '#') {
            // Increase line count
            nline += 1;
        }
    }
    
    
    // Return to original location
    fseek(fp, pos, SEEK_SET);
    // Decrease use count
    PyFile_DecUseCount((PyFileObject *) f);
    
    // Convert to integer
    n = PyInt_FromLong(nline);
    
    // Output
    return n;
}
    
