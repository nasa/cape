
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``02_csv_float``: **FAIL** (command 1)
=============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/ftypes/02_csv_float/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_clean.py
        $ python3 test01_clean.py
        $ python2 test02_dtypes.py
        $ python3 test02_dtypes.py
        $ python2 test03_simple.py
        $ python3 test03_simple.py
        $ python2 test05_py.py
        $ python3 test05_py.py
        $ python2 test04_c.py
        $ python3 test04_c.py

**Included file:** ``test01_clean.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Read CSV file
        db = csvfile.CSVFile("aeroenv.csv")
        
        # Case number
        i = 13
        
        # Get attributes
        mach = db["mach"][i]
        alph = db["alpha"][i]
        beta = db["beta"][i]
        
        # Create a string
        print("m%.2fa%.2fb%.2f" % (mach, alph, beta))
        

**Included file:** ``test02_dtypes.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Read CSV file
        db = csvfile.CSVFile("aeroenv.csv",
            DefaultType="float32",
            Definitions={"beta": {"Type": "float16"}})
        
        # Print data types
        for col in db.cols:
            print("%-5s: %s" % (col, db[col].dtype.name))
        

**Included file:** ``test03_simple.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Read CSV file
        db = csvfile.CSVSimple("aeroenv.csv")
        
        # Case number
        i = 13
        
        # Get attributes
        mach = db["mach"][i]
        alph = db["alpha"][i]
        beta = db["beta"][i]
        
        # Create a string
        print("m%.2fa%.2fb%.2f" % (mach, alph, beta))
        

**Included file:** ``test04_c.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Create empty CSV file
        db = csvfile.CSVFile()
        
        # Read in C
        db.c_read_csv("aeroenv.csv")
        
        # Case number
        i = 13
        
        # Get attributes
        mach = db["mach"][i]
        alph = db["alpha"][i]
        beta = db["beta"][i]
        
        # Create a string
        print("m%.2fa%.2fb%.2f" % (mach, alph, beta))
        

**Included file:** ``test05_py.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.csvfile as csvfile
        
        # Create empty CSV file
        db = csvfile.CSVFile()
        
        # Read in C
        db.py_read_csv("aeroenv.csv")
        
        # Case number
        i = 13
        
        # Get attributes
        mach = db["mach"][i]
        alph = db["alpha"][i]
        beta = db["beta"][i]
        
        # Create a string
        print("m%.2fa%.2fb%.2f" % (mach, alph, beta))
        

Command 1: Clean CSV read: Python 2 (**FAIL**)
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_clean.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.09 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        m1.20a0.00b4.00
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_clean.py", line 5, in <module>
            import cape.attdb.ftypes.csvfile as csvfile
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


