
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``07_xls_aerodb``: **FAIL** (command 1)
==============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/ftypes/07_xls_aerodb/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_xlsx.py
        $ python3 test01_xlsx.py
        $ python2 test02_xls.py
        $ python3 test02_xls.py
        $ python2 test03_dtypes.py
        $ python3 test03_dtypes.py

**Included file:** ``test01_xlsx.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.xlsfile as xlsfile
        
        # Read CSV file
        db = xlsfile.XLSFile("aero_arrow_no_base.xlsx")
        
        # Case number
        i = 6
        
        # Get attributes
        for col in db.cols:
            # Get value
            v = db[col][i]
            # Check type
            if isinstance(v, float):
                # Just use a few decimals
                print("%8s: %.2f" % (col, v))
            else:
                # Print default
                print("%8s: %s" % (col, v))

**Included file:** ``test02_xls.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.xlsfile as xlsfile
        
        # Read CSV file
        db = xlsfile.XLSFile("aero_arrow_no_base.xls")
        
        # Case number
        i = 6
        
        # Get attributes
        for col in db.cols:
            # Get value
            v = db[col][i]
            # Check type
            if isinstance(v, float):
                # Just use a few decimals
                print("%8s: %.2f" % (col, v))
            else:
                # Print default
                print("%8s: %s" % (col, v))

**Included file:** ``test03_dtypes.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.xlsfile as xlsfile
        
        # Read CSV file
        db = xlsfile.XLSFile("aero_arrow_no_base.xlsx",
            Types={
                "config": "str",
                "alpha": "float16",
                "mach": "float32",
                "nStats": "int"
            })
        
        # Print data types
        for col in db.cols:
            # Get array
            V = db[col]
            # Check type
            clsname = V.__class__.__name__
            # Data type
            dtype = V[0].__class__.__name__
            # Status message
            print("%8s: %s (%s)" % (col, dtype, clsname))
        

Command 1: XLSX File: Python 2 (**FAIL**)
------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_xlsx.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.07 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

            mach: 0.80
           alpha: 1.00
          config: poweroff
           Label: 
              CA: 0.34
              CY: -0.00
              CN: 0.15
             CLM: -0.11
         nOrders: 4.49
           nIter: 200.00
          nStats: 100.00
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_xlsx.py", line 5, in <module>
            import cape.attdb.ftypes.xlsfile as xlsfile
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


