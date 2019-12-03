
.. This documentation written by TestDriver()
   on 2019-12-03 at 14:52 PST

Test ``01_import``
====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/ftypes/01_import/``

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
        import cape.attdb.ftypes
        import cape.attdb.ftypes.csv
        

Command 1: Import :mod:`cape.attdb.ftypes`: Python2
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_import.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.482356 seconds
    * Cumulative time: 0.482356 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 2: Import :mod:`cape.attdb.ftypes`: Python3
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_import.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.609552 seconds
    * Cumulative time: 1.09191 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

