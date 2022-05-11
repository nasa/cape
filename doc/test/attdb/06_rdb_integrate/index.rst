
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``06_rdb_integrate``: **FAIL** (command 1)
=================================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/attdb/06_rdb_integrate/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_trapz.py
        $ python3 test01_trapz.py

Command 1: Trapezoidal line load integration: Python 2 (**FAIL**)
------------------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_trapz.py

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

        cols:
            aoap        bullet.x    bullet.dCA  bullet.dCY 
            bullet.dCN  q           beta        T          
            phip        alpha       mach        bullet.dCLL
            bullet.dCLM bullet.dCLN bullet.CA   bullet.CY  
            bullet.CN   bullet.CLL  bullet.CLM  bullet.CLN 
        values:
               mach: 0.80
              alpha: 2.00
               beta: 0.00
          bullet.CN: 0.09
         bullet.CLM: 0.29
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_trapz.py", line 11, in <module>
            import cape.attdb.rdb as rdb
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


