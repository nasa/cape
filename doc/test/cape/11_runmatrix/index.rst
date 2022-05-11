
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``11_runmatrix``: **FAIL** (command 1)
=============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/cape/11_runmatrix/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_nofile.py
        $ python3 test01_nofile.py
        $ python2 test02_conditions.py
        $ python3 test02_conditions.py
        $ cape -I 5 --PASS
        $ cape -I 5 -c
        $ cape -I 6 --ERROR
        $ cape -I 6 -c
        $ cape -I 4:7 --ERROR
        $ cape -I 4:7 -c

:Included Files:
    * :download:`test01_nofile.py`
    * :download:`test02_conditions.py`

Command 1: Run Matrix w/o File: Python 2 (**FAIL**)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_nofile.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.26 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
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
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_nofile.py", line 9, in <module>
            import cape.runmatrix
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


