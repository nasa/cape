
.. This documentation written by TestDriver()
   on 2022-02-04 at 01:45 PST

Test ``10_mat_lineload``: PASS
================================

This test PASSED on 2022-02-04 at 01:45 PST

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
    * Command took 0.60 seconds
:STDOUT:
    * **PASS**
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
    * **PASS**

Command 2: Clean MAT read: Python 3 (PASS)
-------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_read.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.74 seconds
    * Cumulative time: 1.34 seconds
:STDOUT:
    * **PASS**
    * Target:

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
        

:STDERR:
    * **PASS**

Command 3: MATFile write: Python 2 (PASS)
------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_write.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.41 seconds
    * Cumulative time: 1.75 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        MACH      : (6,)
        T         : (6,)
        alpha     : (6,)
        aoap      : (6,)
        beta      : (6,)
        bullet.dCA: (101, 6)
        bullet.dCN: (101, 6)
        bullet.dCY: (101, 6)
        bullet.x  : (101,)
        phip      : (6,)
        q         : (6,)
        

:STDERR:
    * **PASS**

Command 4: MATFile write: Python 3 (PASS)
------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_write.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.53 seconds
    * Cumulative time: 2.28 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

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
        MACH      : (6,)
        

:STDERR:
    * **PASS**

