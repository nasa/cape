
.. This documentation written by TestDriver()
   on 2022-03-04 at 01:40 PST

Test ``01_bullet``: **FAIL** (command 1)
==========================================

This test **FAILED** (command 1) on 2022-03-04 at 01:40 PST

This test is run in the folder:

    ``test/pyfun/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyfun -I 8
        $ pyfun -I 8 -c
        $ pyfun -I 8 --fm
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Case 8 (**FAIL**)
---------------------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.83 seconds
:STDOUT:
    * **PASS**
    * Actual: (empty)
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "/u/wk/ddalle/usr/cape/bin/pyfun", line 5, in <module>
            from cape.pyfun.cli import main
          File "/u/wk/ddalle/usr/cape/cape/pyfun/__init__.py", line 70, in <module>
            from .cntl import Cntl
          File "/u/wk/ddalle/usr/cape/cape/pyfun/cntl.py", line 57, in <module>
            from . import dataBook
          File "/u/wk/ddalle/usr/cape/cape/pyfun/dataBook.py", line 72, in <module>
            from . import pointSensor
          File "/u/wk/ddalle/usr/cape/cape/pyfun/pointSensor.py", line 255, in <module>
            class DBTriqPoint(cdbook.DBTriqPoint):
        AttributeError: 'module' object has no attribute 'DBTriqPoint'
        


