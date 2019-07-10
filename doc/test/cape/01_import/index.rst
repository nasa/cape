
.. This documentation written by TestDriver()
   on 2019-07-09 at 21:55 PDT

Test ``01_import``
====================

This test is run in the folder:

    ``/home/dalle/usr/pycart/test/cape/01_import/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_import.py
        $ python3 test01_import.py

Command 1: ``python2 test01_import.py``
----------------------------------------
:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.221288 seconds
    * Cumulative time: 0.221288 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 2: ``python3 test01_import.py``
----------------------------------------
:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.101548 seconds
    * Cumulative time: 0.322836 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_import.py", line 5, in <module>
            import cape
          File "/home/dalle/usr/pycart/cape/__init__.py", line 102, in <module>
            from .tri    import Tri, Triq
          File "/home/dalle/usr/pycart/cape/tri.py", line 34, in <module>
            import numpy as np
        ModuleNotFoundError: No module named 'numpy'
        


