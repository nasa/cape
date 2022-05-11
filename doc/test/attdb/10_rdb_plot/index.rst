
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``10_rdb_plot``: **FAIL** (command 1)
============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/attdb/10_rdb_plot/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_pre.py
        $ python3 test01_pre.py
        $ python2 test02_ll.py
        $ python3 test02_ll.py

Command 1: Prep plot args: Python 2 (**FAIL**)
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_pre.py

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

        Version 1: scalar by scalar args
          col  = 'bullet.CN'
          I    = []
          mach =  0.90
          aoa  =  1.90
          beta = -0.10
        Version 2: scalar by array indices
          col  = 'bullet.CN'
          I    = 200, 238
          mach =  0.80,  0.80
          aoa  =  0.75,  1.50
          beta =  1.25, -2.00
        Version 3:
          col = 'bullet.dCN'
          I    = 240, 251
          mach =  0.80,  0.80
          aoa  =  1.50,  1.50
          beta = -1.50,  1.25
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_pre.py", line 8, in <module>
            import cape.attdb.dbfm as dbfm
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


