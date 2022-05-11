
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``04_rdb_regularize``: **FAIL** (command 1)
==================================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/attdb/04_rdb_regularize/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_griddata.py
        $ python3 test01_griddata.py
        $ python2 test02_rbf.py
        $ python3 test02_rbf.py

Command 1: Regularize using griddata: Python 2 (**FAIL**)
----------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_griddata.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.17 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        max error(regalpha) = 0.00
        max error(regbeta)  = 0.00
        monotonic(regCN): True
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_griddata.py", line 8, in <module>
            import cape.attdb.rdb as rdb
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


