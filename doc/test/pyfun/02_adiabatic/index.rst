
.. This documentation written by TestDriver()
   on 2022-03-04 at 01:40 PST

Test ``02_adiabatic``: **FAIL** (command 1)
=============================================

This test **FAILED** (command 1) on 2022-03-04 at 01:40 PST

This test is run in the folder:

    ``test/pyfun/02_adiabatic/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyfun -f pyFun01.json -I 0 --no-start
        $ cat bullet/m0.80a0.0b0.0/fun3d.00.nml
        $ pyfun -f pyFun02.json -I 1 --no-start
        $ cat bullet/m0.80a4.0b0.0/fun3d.00.nml
        $ pyfun -f pyFun03.json -I 2 --no-start
        $ cat bullet/m0.80a10.0b0.0/fun3d.00.nml

Command 1: Create Input Files (**FAIL**)
-----------------------------------------

:Command:
    .. code-block:: console

        $ pyfun -f pyFun01.json -I 0 --no-start

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.78 seconds
:STDOUT:
    * **PASS**
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
        


