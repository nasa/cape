#include <Python.h>

// This is required to start the NumPy C-API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _ftypes_ARRAY_API

#include <numpy/arrayobject.h>

#include "cape_CSVFile.h"


static PyMethodDef Methods[] = {
    // CSV file utilities
    {
        "CSVFileCountLines",
        cape_CSVFileCountLines,
        METH_VARARGS,
        doc_CSVFileCountLines
    },
    // Sentinel
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_ftypes(void)
{
    // The module
    PyObject *m;
    
    // This must be called before using the NumPy API
    import_array();
    
    // Initialize module
    m = Py_InitModule("_ftypes", Methods);
    // Check for errors
    if (m == NULL)
        return;
}
