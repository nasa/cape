Command 1: First-column BoolMap: Python 2 (**FAIL**)
-----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.42 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

           _col1: p
            mach: 2.1
           alpha: 4.0
            beta: 1.5
          config: poweroff
           Label: 
            user: @user3
           ERROR: False
            PASS: True
        

    * Target:

      .. code-block:: none

           _col1: None
            mach: 2.1
           alpha: 4.0
            beta: 1.5
          config: poweroff
           Label: 
            user: @user3
           ERROR: False
            PASS: True
        

:STDERR:
    * **PASS**




.. This documentation written by TestDriver()
   on 2021-10-13 at 11:50 PDT

Test ``06_text_runmatrix``: **FAIL** (command 1)
==================================================

This test **FAILED** (command 1) on 2021-10-13 at 11:50 PDT

This test is run in the folder:

    ``test/ftypes/06_text_runmatrix/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_minimal.py
        $ python3 test01_minimal.py

**Included file:** ``test01_minimal.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import CSV module
        import cape.attdb.ftypes.textdata as td
        
        # Read as generic text with special first-column flag
        db = td.TextDataFile("runmatrix.csv",
            FirstColBoolMap={"PASS": "p", "ERROR": "e"})
        
        # Case number
        i = 6
        
        # Get attributes
        for col in db.cols:
            print("%8s: %s" % (col, db[col][i]))
        

Command 1: First-column BoolMap: Python 2 (**FAIL**)
-----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.42 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

           _col1: p
            mach: 2.1
           alpha: 4.0
            beta: 1.5
          config: poweroff
           Label: 
            user: @user3
           ERROR: False
            PASS: True
        

    * Target:

      .. code-block:: none

           _col1: None
            mach: 2.1
           alpha: 4.0
            beta: 1.5
          config: poweroff
           Label: 
            user: @user3
           ERROR: False
            PASS: True
        

:STDERR:
    * **PASS**

