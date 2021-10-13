
.. This documentation written by TestDriver()
   on 2021-10-13 at 14:17 PDT

Test ``10_mat_lineload``: **FAIL** (command 2)
================================================

This test **FAILED** (command 2) on 2021-10-13 at 14:17 PDT

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
        
        # Import MAT module
        import cape.attdb.ftypes.matfile as matfile
        
        # Read MAT file
        db = matfile.MATFile("bullet.mat")
        
        # Loop through colums
        for col in db.cols:
            # Display it and its shape
            print("%-10s: %s" % (col, db[col].shape))
        

**Included file:** ``test02_write.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
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
        
        # Loop through colums
        for col in db.cols:
            # Display it and its shape
            print("%-10s: %s" % (col, db[col].shape))
        

Command 1: Clean MAT read: Python 2 (PASS)
-------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_read.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.42 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        aoap      : (6,)
        bullet.x  : (101,)
        bullet.dCA: (101, 6)
        bullet.dCY: (101, 6)
        bullet.dCN: (101, 6)
        q         : (6,)
        beta      : (6,)
        T         : (6,)
        phip      : (6,)
        alpha     : (6,)
        mach      : (6,)
        

:STDERR:
    * **PASS**

Command 2: Clean MAT read: Python 3 (**FAIL**)
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_read.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.57 seconds
    * Cumulative time: 0.99 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

        mach      : (6,)
        alpha     : (6,)
        beta      : (6,)
        aoap      : (6,)
        phip      : (6,)
        q         : (6,)
        T         : (6,)
        bullet.x  : (101,)
        bullet.dCA: (101, 6)
        bullet.dCY: (101, 6)
        bullet.dCN: (101, 6)
        

    * Target:

      .. code-block:: none

        aoap      : (6,)
        bullet.x  : (101,)
        bullet.dCA: (101, 6)
        bullet.dCY: (101, 6)
        bullet.dCN: (101, 6)
        q         : (6,)
        beta      : (6,)
        T         : (6,)
        phip      : (6,)
        alpha     : (6,)
        mach      : (6,)
        

:STDERR:
    * **PASS**

