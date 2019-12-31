
.. This documentation written by TestDriver()
   on 2019-12-31 at 10:21 PST

Test ``07_xls_aerodb``
========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/ftypes/07_xls_aerodb/``

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
        import cape.attdb.ftypes.xls as xls
        
        # Read CSV file
        db = xls.XLSFile("aero_arrow_no_base.xlsx")
        
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
        import cape.attdb.ftypes.xls as xls
        
        # Read CSV file
        db = xls.XLSFile("aero_arrow_no_base.xls")
        
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
        import cape.attdb.ftypes.xls as xls
        
        # Read CSV file
        db = xls.XLSFile("aero_arrow_no_base.xlsx",
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
        

Command 1: XLSX File: Python 2
-------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_xlsx.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.532903 seconds
    * Cumulative time: 0.532903 seconds
:STDOUT:
    * **PASS**
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
    * **PASS**

Command 2: XLSX File: Python 3
-------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_xlsx.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.783047 seconds
    * Cumulative time: 1.31595 seconds
:STDOUT:
    * **PASS**
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
    * **PASS**

Command 3: XLS File: Python 2
------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_xls.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.509683 seconds
    * Cumulative time: 1.82563 seconds
:STDOUT:
    * **PASS**
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
    * **PASS**

Command 4: XLS File: Python 3
------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_xls.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.620976 seconds
    * Cumulative time: 2.44661 seconds
:STDOUT:
    * **PASS**
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
    * **PASS**

Command 5: Specified dtypes: Python 2
--------------------------------------

:Command:
    .. code-block:: console

        $ python2 test03_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.492162 seconds
    * Cumulative time: 2.93877 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            mach: float32 (ndarray)
           alpha: float16 (ndarray)
          config: unicode (list)
           Label: unicode (list)
              CA: float64 (ndarray)
              CY: float64 (ndarray)
              CN: float64 (ndarray)
             CLM: float64 (ndarray)
         nOrders: float64 (ndarray)
           nIter: float64 (ndarray)
          nStats: int32 (ndarray)
        

:STDERR:
    * **PASS**

Command 6: Specified dtypes: Python 3
--------------------------------------

:Command:
    .. code-block:: console

        $ python3 test03_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.616545 seconds
    * Cumulative time: 3.55532 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            mach: float32 (ndarray)
           alpha: float16 (ndarray)
          config: str (list)
           Label: str (list)
              CA: float64 (ndarray)
              CY: float64 (ndarray)
              CN: float64 (ndarray)
             CLM: float64 (ndarray)
         nOrders: float64 (ndarray)
           nIter: float64 (ndarray)
          nStats: int32 (ndarray)
        

:STDERR:
    * **PASS**

