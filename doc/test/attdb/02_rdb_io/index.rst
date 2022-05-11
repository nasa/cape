
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``02_rdb_io``: **FAIL** (command 1)
==========================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/attdb/02_rdb_io/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_csv_to_mat.py
        $ python3 test01_csv_to_mat.py

Command 1: CSV read; MAT write/read: Python 2 (**FAIL**)
---------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_csv_to_mat.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.08 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        Case 13: m0.95a5.00 CA=0.526
        Case 13: m0.95a5.00 CA=0.526
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_csv_to_mat.py", line 5, in <module>
            import cape.attdb.rdb as rdb
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


