
.. This documentation written by TestDriver()
   on 2019-08-29 at 09:50 PDT

Test ``11_runmatrix``
=======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/cape/11_runmatrix/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_nofile.py
        $ python3 test01_nofile.py
        $ python2 test02_conditions.py
        $ python3 test02_conditions.py

:Included Files:
    * :download:`test01_nofile.py`
    * :download:`test02_conditions.py`

Command 1: Run Matrix w/o File: Python 2
-----------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_nofile.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.374068 seconds
    * Cumulative time: 0.374068 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        01: Folder names
        m1.4a0.0b0.0
        m1.4a4.0b0.0
        m1.4a0.0b4.0
        m1.4a4.0b4.0
        02: Full folder names
        poweroff/m1.4a0.0b0.0
        poweroff/m1.4a4.0b0.0
        poweroff/m1.4a0.0b4.0
        poweroff/m1.4a4.0b4.0
        03: Total angle of attack
        aoap = 0.00, phip =  0.00
        aoap = 4.00, phip =  0.00
        aoap = 4.00, phip = 90.00
        aoap = 5.65, phip = 45.07
        04: Modified format
        m1.40_a0.0b4.0
        05: Conditions JSON file
        {
         "mach": 1.4,
         "alpha": 4.0,
         "beta": 0.0,
         "config": "poweroff"
        }
        

:STDERR:
    * **PASS**

Command 2: Run Matrix w/o File: Python 3
-----------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_nofile.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.698979 seconds
    * Cumulative time: 1.07305 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        01: Folder names
        m1.4a0.0b0.0
        m1.4a4.0b0.0
        m1.4a0.0b4.0
        m1.4a4.0b4.0
        02: Full folder names
        poweroff/m1.4a0.0b0.0
        poweroff/m1.4a4.0b0.0
        poweroff/m1.4a0.0b4.0
        poweroff/m1.4a4.0b4.0
        03: Total angle of attack
        aoap = 0.00, phip =  0.00
        aoap = 4.00, phip =  0.00
        aoap = 4.00, phip = 90.00
        aoap = 5.65, phip = 45.07
        04: Modified format
        m1.40_a0.0b4.0
        05: Conditions JSON file
        {
         "mach": 1.4,
         "alpha": 4.0,
         "beta": 0.0,
         "config": "poweroff"
        }
        

:STDERR:
    * **PASS**

Command 3: Conversions: Python 2
---------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_conditions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.382146 seconds
    * Cumulative time: 1.45519 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 4: Conversions: Python 3
---------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_conditions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.729918 seconds
    * Cumulative time: 2.18511 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

