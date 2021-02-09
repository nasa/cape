#ifndef _CAPEC_PYTYPES_H
#define _CAPEC_PYTYPES_H

// Standard library
#include <Python.h>

// Local includes


// Define macros according to Python type
#if PY_MAJOR_VERSION >= 3
    // Unicode, long
    #define capePyString_Check(o)       PyUnicode_Check(o)
    #define capePyString_AsString(o)    PyUnicode_AsUTF8(o)
    #define capePyString_FromString(s)  PyUnicode_FromString(s)
    #define capePyInt_Check(i)          PyLong_Check(i)
    #define capePyInt_FromLong(i)       PyLong_FromLong(i)
    #define capePyInt_AS_LONG(i)        PyLong_AS_LONG(i)
    
    
    //const char* capePyString_AsString(PyObject* o);

#else

    // String, int
    #define capePyString_Check(o)       PyString_Check(o)
    #define capePyString_AsString(o)    PyString_AsString(o)
    #define capePyString_FromString(s)  PyString_FromString(s)
    #define capePyInt_Check(i)          PyInt_Check(i)
    #define capePyInt_FromLong(i)       PyInt_FromLong(i)
    #define capePyInt_AS_LONG(i)        PyInt_AS_LONG(i)

#endif

#endif  // _CAPEC_PYTYPES_H