
.. This documentation written by TestDriver()
   on 2021-10-13 at 13:23 PDT

Test ``04_csv_mixed``: **FAIL** (command 1)
=============================================

This test **FAILED** (command 1) on 2021-10-13 at 13:23 PDT

This test is run in the folder:

    ``test/ftypes/04_csv_mixed/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_minimal.py
        $ python3 test01_minimal.py
        $ python2 test02_dtypes.py
        $ python3 test02_dtypes.py

**Included file:** ``test01_minimal.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Read CSV file
        db = csvfile.CSVFile("runmatrix.csv")
        
        # Case number
        i = 6
        
        # Get attributes
        for col in db.cols:
            print("%8s: %s" % (col, db[col][i]))
        

**Included file:** ``test02_dtypes.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Read CSV file
        db = csvfile.CSVFile("runmatrix.csv",
            DefaultType="float32",
            Types={
                "config": "str",
                "mach": "float16"
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
        

Command 1: Minimal Definitions: Python 2 (**FAIL**)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.43 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

            mach: 2.1
           alpha: 4.0
            beta: 1.5
          config: poweroff
           Label:   
            user: @user3
        

    * Target:

      .. code-block:: none

            mach: 2.1
           alpha: 4.0
            beta: 1.5
          config: poweroff
           Label: 
            user: @user3
        

:STDERR:
    * **PASS**

