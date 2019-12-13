
.. This documentation written by TestDriver()
   on 2019-12-13 at 13:25 PST

Test ``06_text_runmatrix``
============================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/ftypes/06_text_runmatrix/``

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
        

Command 1: First-column BoolMap: Python 2
------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.484268 seconds
    * Cumulative time: 0.484268 seconds
:STDOUT:
    * **PASS**
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

Command 2: First-column BoolMap: Python 3
------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_minimal.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.760662 seconds
    * Cumulative time: 1.24493 seconds
:STDOUT:
    * **PASS**
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

