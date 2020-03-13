#include <Python.h>

// This is required to start the NumPy C-API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _cape_ARRAY_API
#include <numpy/arrayobject.h>

#include "capec_NumPy.h"
#include "capec_BaseFile.h"
#include "cape_CSVFile.h"

// Declare each module function
static PyMethodDef FTypesMethods[] = {
    // CSV file utilities
    {
        "CSVFileCountLines",
        cape_CSVFileCountLines,
        METH_VARARGS,
        doc_CSVFileCountLines
    },
    {
        "CSVFileReadData",
        cape_CSVFileReadData,
        METH_VARARGS,
        doc_CSVFileReadData
    },
    // Sentinel
    {NULL, NULL, 0, NULL}
};

// Create module's struct if compiling for Python 3
#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef ftypesmodule = {
        PyModuleDef_HEAD_INIT,
        "_ftypes",                   // Name of module
        "CAPE data file types module\n",   // Documentation
        -1,                          // -1 if module keeps state in globals
        FTypesMethods
    };
#endif

// Actually declare the module
PyMODINIT_FUNC
init_ftypes(void)
{
    // The module
    PyObject *m;
    
    // This must be called before using the NumPy API
    import_array();
    
    // Initialize module
    #if PY_MAJOR_VERSION >= 3
        // Python 3 syntax
        m = PyModule_Create(&ftypesmodule);
        // Check for errors
        if (m == NULL)
            return m;
    #else
        // Python 2 syntax
        m = Py_InitModule("_ftypes", FTypesMethods);
        // Check for errors
        if (m == NULL)
            return;
    #endif
    
    // Add data types to module
    capec_AddDTypes(m);
    
    // Return module
    #if PY_MAJOR_VERSION >= 3
        return m;
    #endif
}
