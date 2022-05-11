
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``03_rdb_find``: **FAIL** (command 1)
============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/attdb/03_rdb_find/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_find.py
        $ python3 test01_find.py
        $ python2 test02_multi.py
        $ python3 test02_multi.py

Command 1: Basic DataKit search: Python 2 (**FAIL**)
-----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_find.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.07 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        Case 12: CT=0.0
        Case 15: CT=1.4
        Case 17: CT=1.4
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_find.py", line 5, in <module>
            import cape.attdb.rdb as rdb
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


