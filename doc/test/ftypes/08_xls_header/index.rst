
.. This documentation written by TestDriver()
   on 2020-02-16 at 21:48 PST

Test ``08_xls_header``
========================

This test is run in the folder:

    ``/home/dalle/usr/pycart/test/ftypes/08_xls_header/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_workbook.py
        $ python3 test01_workbook.py

**Included file:** ``test01_workbook.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Third-party modules
        import numpy as np
        
        # Import CSV module
        import cape.attdb.ftypes.xlsfile as xlsfile
        
        # Read CSV file
        db = xlsfile.XLSFile("header_categories.xlsx")
        
        # Get attributes
        for col in db.cols:
            # Get value
            v = db[col]
            # Check type
            if isinstance(v, np.ndarray):
                # Convert shape to string
                shape = "x".join([str(s) for s in v.shape])
                # Get data type
                dtype = v.dtype
                # Print name, type, and size
                print("    %-21s: array (shape=%s, dtype=%s)" % (col, shape, dtype))
            else:
                # Length
                n = len(v)
                # Type of first entry
                if n > 0:
                    dtype = v[0].__class__.__name__
                else:
                    dtype = "str"
                # Print type and size
                print("    %-21s: list (len=%i, type=%s)" % (col, n, dtype))

Command 1: Auto-Workbook with Arrays: Python 2
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_workbook.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.250967 seconds
    * Cumulative time: 0.250967 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            colnames.mach        : array (shape=75, dtype=float64)
            colnames.alpha       : array (shape=75, dtype=float64)
            colnames.beta        : array (shape=75, dtype=float64)
            colnames.config      : list (len=75, type=unicode)
            colnames.mach        : array (shape=75, dtype=float64)
            colnames.alpha       : array (shape=75, dtype=float64)
            colnames.beta        : array (shape=75, dtype=float64)
            colnames.config      : list (len=75, type=unicode)
            cols_with_array.mach : array (shape=3, dtype=float64)
            cols_with_array.alpha: array (shape=3, dtype=float64)
            cols_with_array.beta : array (shape=3, dtype=float64)
            cols_with_array.DCN  : array (shape=3x3, dtype=float64)
            cols_with_array.mach : array (shape=3, dtype=float64)
            cols_with_array.alpha: array (shape=3, dtype=float64)
            cols_with_array.beta : array (shape=3, dtype=float64)
            cols_with_array.DCN  : array (shape=3x3, dtype=float64)
        

:STDERR:
    * **PASS**

Command 2: Auto-Workbook with Arrays: Python 3
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_workbook.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.459722 seconds
    * Cumulative time: 0.710689 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            colnames.mach        : array (shape=75, dtype=float64)
            colnames.alpha       : array (shape=75, dtype=float64)
            colnames.beta        : array (shape=75, dtype=float64)
            colnames.config      : list (len=75, type=str)
            colnames.mach        : array (shape=75, dtype=float64)
            colnames.alpha       : array (shape=75, dtype=float64)
            colnames.beta        : array (shape=75, dtype=float64)
            colnames.config      : list (len=75, type=str)
            cols_with_array.mach : array (shape=3, dtype=float64)
            cols_with_array.alpha: array (shape=3, dtype=float64)
            cols_with_array.beta : array (shape=3, dtype=float64)
            cols_with_array.DCN  : array (shape=3x3, dtype=float64)
            cols_with_array.mach : array (shape=3, dtype=float64)
            cols_with_array.alpha: array (shape=3, dtype=float64)
            cols_with_array.beta : array (shape=3, dtype=float64)
            cols_with_array.DCN  : array (shape=3x3, dtype=float64)
        

:STDERR:
    * **PASS**
