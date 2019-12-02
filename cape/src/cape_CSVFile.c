#include <Python.h>
#include <stdio.h>
#include <math.h>

// Local includes
#include "capec_Memory.h"
#include "capec_NumPy.h"
#include "capec_BaseFile.h"
#include "capec_CSVFile.h"


// Read through file to count data lines
PyObject *
cape_CSVFileCountLines(PyObject *self, PyObject *args)
{
   // --- Declarations ---
    // Line counts
    PyObject *n;
    size_t nline;
    // File handle
    PyObject *f;
    FILE *fp;
 
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
    n = PyInt_FromLong((long) nline);
    
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
    // Number of rows
    size_t irow;
    size_t jcol;
    size_t nrow;
    // Iterators
    Py_ssize_t i;
    // Data file interface
    PyObject *db;
    PyObject *cols;
    PyObject *col;
    PyObject *j;
    PyObject *dtypes;
    PyObject *V;
    // Column attributes
    Py_ssize_t ncol;
    int *DTYPES;
    long DTYPE;
    // Local pointer to all data
    void **coldata;
    // File handle
    PyObject *f;
    FILE *fp;
    long pos;
    // Strings
    char buff[80];
    char c;
    
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
    
    // Check type of file handle
    if (!PyFile_Check(f)) {
        // Not a file
        PyErr_SetString(PyExc_TypeError, "Input is not a file handle");
        return NULL;
    }
    
   // --- Setup ---
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
    
   // --- File Length ---
    // Convert to native C
    fp = PyFile_AsFile(f);
    // Remember current location
    pos = ftell(fp);
    // Increment use count
    PyFile_IncUseCount((PyFileObject *) f);
    
    // Get line count
    nrow = capec_CSVFileCountLines(fp);

   // --- Initialization ---
    // Allocate column data
    coldata = (void *) malloc(((size_t) ncol) * sizeof(char *));
    // Initialize each key
    for (i=0; i<ncol; ++i) {
        // Get column name
        col = PyList_GET_ITEM(cols, i);
        // Check if it's already present
        if (PyDict_Contains(db, col)) {
            // Delete it; with error check
            ierr = PyDict_DelItem(db, col);
            // Error check
            if (ierr) {
                PyErr_Format(PyExc_KeyError, 
                    "Failed to delete column '%s'", PyString_AsString(col));
                return NULL;
            }
        }
        // Initialize the column
        V = capeFILE_NewCol1D(DTYPES[i], nrow);
        // Check for errors
        if (V == NULL) {
            PyFile_DecUseCount((PyFileObject *) f);
            fseek(fp, pos, SEEK_SET);
            return NULL;
        }
        // Otherwise, set it
        ierr = PyDict_SetItem(db, col, V);
        // Check for errors
        if (ierr) {
            PyErr_Format(PyExc_KeyError, "Failed to set column '%s'",
                PyString_AsString(col));
            return NULL;
        }
        // Assign data to quick-access list
        if (PyList_Check(V)) {
            // Save pointer directly to Python list
            coldata[i] = V;
        } else {
            // Save pointer to array's data
            coldata[i] = PyArray_DATA((PyArrayObject *) V);
        }
    }

   // -- Read ---
    // Initialize number of rows actually read
    irow = 0;
    // Loop through file
    while (!feof(fp)) {
        // Read any current white space
        capcec_FileAdvanceWhiteSpace(fp);
        // Read next character
        c = getc(fp);
        // Filter character
        if (c == '\n') {
            // We just read an empty line
            continue;
        } else if (feof(fp)) {
            // Completed read already
            break;
        } else if (c == '#') {
            // Comment line
            capec_FileAdvanceEOL(fp);
            continue;
        } else {
            // Go back one character
            fseek(fp, -1, SEEK_CUR);
        }
        
        // Loop through columns
        for (jcol=0; jcol<ncol; jcol++)
        {
            // Read next entry
            ierr = capeCSV_ReadNext(fp, coldata[jcol], DTYPES[jcol], irow);
            if (ierr) {
                PyFile_DecUseCount((PyFileObject *) f);
                fseek(fp, pos, SEEK_SET);
                return NULL;
            }
            // Read next white space
            capcec_FileAdvanceWhiteSpace(fp);
            // Read next character
            c = getc(fp);
            // Check if we're in the last column
            if (jcol + 1 == ncol) {
                // Character should be newline
                if (c == '#') {
                    // Acceptable; comment after data
                    capec_FileAdvanceEOL(fp);
                } else if (c != '\n') {
                    // Line should be over
                    PyErr_Format(PyExc_ValueError,
                        "Data row %li extends past %i columns",
                        (long) irow, (int) jcol);
                    PyFile_DecUseCount((PyFileObject *) f);
                    fseek(fp, pos, SEEK_SET);
                    return NULL;
                }
            } else {
                // Filter character
                if (c != ',') {
                    // Early EOL
                    PyErr_Format(PyExc_ValueError,
                        "After col %i on data row %li: expected ',', not '%c'",
                        (int) (jcol + 1), (long) irow, c);
                    PyFile_DecUseCount((PyFileObject *) f);
                    fseek(fp, pos, SEEK_SET);
                    return NULL;
                }
                // Advance white space again
                capcec_FileAdvanceWhiteSpace(fp);
            }
        }
        
        // Increase row counter
        irow += 1;
    }
    
   // --- Cleanup ---
    // Decrease use count
    PyFile_DecUseCount((PyFileObject *) f);
    
    // Deallocate list of points
    free(coldata);
    
    // Output
    Py_RETURN_NONE;
}
