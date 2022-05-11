
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``10_mat_lineload``: **FAIL** (command 1)
================================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/ftypes/10_mat_lineload/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_read.py
        $ python3 test01_read.py
        $ python2 test02_write.py
        $ python3 test02_write.py

**Included file:** ``test01_read.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Standard library
        import sys
        
        # Import MAT module
        import cape.attdb.ftypes.matfile as matfile
        
        # Read MAT file
        db = matfile.MATFile("bullet.mat")
        
        # Get columns
        if sys.version_info.major == 2:
            # Can't control column order ...
            cols = sorted(db.cols)
        else:
            # Column order under our control
            cols = db.cols
        # Loop through colums
        for col in cols:
            # Display it and its shape
            print("%-10s: %s" % (col, db[col].shape))
        

**Included file:** ``test02_write.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Standard library
        import sys
        
        # Import MAT module
        import cape.attdb.ftypes.matfile as matfile
        
        # Read MAT file
        db = matfile.MATFile("bullet.mat")
        
        # Rename a column
        db["MACH"] = db.pop("mach")
        db.cols.remove("mach")
        db.cols.append("MACH")
        
        # Write it
        db.write_mat("bullet1.mat")
        
        # Reread
        db = matfile.MATFile("bullet1.mat")
        
        # Get columns
        if sys.version_info.major == 2:
            # Can't control column order ...
            cols = sorted(db.cols)
        else:
            # Column order under our control
            cols = db.cols
        
        # Loop through colums
        for col in cols:
            # Display it and its shape
            print("%-10s: %s" % (col, db[col].shape))
        

Command 1: Clean MAT read: Python 2 (**FAIL**)
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_read.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.13 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        T         : (6,)
        alpha     : (6,)
        aoap      : (6,)
        beta      : (6,)
        bullet.dCA: (101, 6)
        bullet.dCN: (101, 6)
        bullet.dCY: (101, 6)
        bullet.x  : (101,)
        mach      : (6,)
        phip      : (6,)
        q         : (6,)
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_read.py", line 8, in <module>
            import cape.attdb.ftypes.matfile as matfile
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


