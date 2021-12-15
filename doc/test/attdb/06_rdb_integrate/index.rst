
.. This documentation written by TestDriver()
   on 2021-12-11 at 01:45 PST

Test ``06_rdb_integrate``: PASS
=================================

This test PASSED on 2021-12-11 at 01:45 PST

This test is run in the folder:

    ``test/attdb/06_rdb_integrate/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_trapz.py
        $ python3 test01_trapz.py

Command 1: Trapezoidal line load integration: Python 2 (PASS)
--------------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_trapz.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.44 seconds
:STDOUT:
    * **PASS**
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
    * **PASS**

Command 2: Trapezoidal line load integration: Python 3 (PASS)
--------------------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_trapz.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.52 seconds
    * Cumulative time: 0.96 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        cols:
            mach        alpha       beta        aoap       
            phip        q           T           bullet.x   
            bullet.dCA  bullet.dCY  bullet.dCN  bullet.dCLL
            bullet.dCLM bullet.dCLN bullet.CA   bullet.CY  
            bullet.CN   bullet.CLL  bullet.CLM  bullet.CLN 
        values:
               mach: 0.80
              alpha: 2.00
               beta: 0.00
          bullet.CN: 0.09
         bullet.CLM: 0.29
        

:STDERR:
    * **PASS**

