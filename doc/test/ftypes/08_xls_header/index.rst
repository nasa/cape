
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``08_xls_header``: **FAIL** (command 1)
==============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/ftypes/08_xls_header/``

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
        

Command 1: Auto-Workbook with Arrays: Python 2 (**FAIL**)
----------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_workbook.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.24 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

            colnames.mach        : array (shape=75, dtype=float64)
            colnames.alpha       : array (shape=75, dtype=float64)
            colnames.beta        : array (shape=75, dtype=float64)
            colnames.config      : list (len=75, type=unicode)
            cols_with_array.mach : array (shape=3, dtype=float64)
            cols_with_array.alpha: array (shape=3, dtype=float64)
            cols_with_array.beta : array (shape=3, dtype=float64)
            cols_with_array.DCN  : array (shape=3x3, dtype=float64)
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_workbook.py", line 8, in <module>
            import cape.attdb.ftypes.xlsfile as xlsfile
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


