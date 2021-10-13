
.. This documentation written by TestDriver()
   on 2021-10-13 at 14:18 PDT

Test ``05_csv_noheader``: **FAIL** (command 3)
================================================

This test **FAILED** (command 3) on 2021-10-13 at 14:18 PDT

This test is run in the folder:

    ``test/ftypes/05_csv_noheader/``

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
            cols=["mach", "alpha", "beta", "config", "Label", "user"],
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
        

Command 1: Minimal Definitions: Python 2 (PASS)
------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.44 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            col1: 2.1
            col2: 4.0
            col3: 1.5
            col4: poweroff
            col5: 
            col6: @user3
        

:STDERR:
    * **PASS**

Command 2: Minimal Definitions: Python 3 (PASS)
------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.58 seconds
    * Cumulative time: 1.02 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

            col1: 2.1
            col2: 4.0
            col3: 1.5
            col4: poweroff
            col5: 
            col6: @user3
        

:STDERR:
    * **PASS**

Command 3: Specified Column Titles: Python 2 (**FAIL**)
--------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_dtypes.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.39 seconds
    * Cumulative time: 1.40 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

            col1: float32 (ndarray)
            col2: float32 (ndarray)
            col3: float32 (ndarray)
            col4: str (list)
            col5: str (list)
            col6: str (list)
        

    * Target:

      .. code-block:: none

            mach: float16 (ndarray)
           alpha: float32 (ndarray)
            beta: float32 (ndarray)
          config: str (list)
           Label: str (list)
            user: str (list)
        

:STDERR:
    * **PASS**

