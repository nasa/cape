// Standard library
#include <Python.h>
#include <stdio.h>

// Local includes


// Define macros according to Python type
#if PY_MAJOR_VERSION >= 3
    // Unicode, long
    #define capePyString_Check(o) PyUnicode_Check(o)

#else

    // String, int
    #define capePyString_Check(o) PyString_Check(o)

#endif