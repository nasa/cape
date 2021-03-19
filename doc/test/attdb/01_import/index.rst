
.. This documentation written by TestDriver()
   on 2021-03-19 at 09:48 PDT

Test ``01_import``
====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/attdb/01_import/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_import.py
        $ python3 test01_import.py

**Included file:** ``test01_import.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import ftypes module(s)
        import cape.attdb.rdbnull
        import cape.attdb.rdbscalar
        

Command 1: Import :mod:`cape.attdb.rdbnull`: Python2
-----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_import.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.476219 seconds
    * Cumulative time: 0.476219 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_import.py", line 5, in <module>
            import cape.attdb.rdbnull
        ImportError: No module named rdbnull
        


