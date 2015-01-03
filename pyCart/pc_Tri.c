#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL _pyxflow_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>


// Function to write Components.i.tri file
PyObject *
pc_WriteTri(PyObject *self, PyObject *args)
{
    int ierr;
    
    // Return a number.
    return Py_BuildValue("n", 0);
}


// Function to write the component IDs
PyObject *
pc_WriteCompID(PyObject *self, PyObject *args)
{
    int ierr;
    
    // Return a number.
    return Py_BuildValue("n", 0);
}
