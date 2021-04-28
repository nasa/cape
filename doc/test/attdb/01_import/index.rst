
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:51 PDT

Test ``01_import``
====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/cape/test/attdb/01_import/``

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
        import cape.attdb
        import cape.attdb.rdb
        

Command 1: Import :mod:`cape.attdb.rdb`: Python2
-------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_import.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.587881 seconds
    * Cumulative time: 0.587881 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 2: Import :mod:`cape.attdb.rdb`: Python3
-------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_import.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.641435 seconds
    * Cumulative time: 1.22932 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

