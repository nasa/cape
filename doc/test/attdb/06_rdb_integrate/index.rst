
.. This documentation written by TestDriver()
   on 2020-03-25 at 10:09 PDT

Test ``06_rdb_integrate``
===========================

This test is run in the folder:

    ``/home/dalle/usr/pycart/test/attdb/06_rdb_integrate/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_trapz.py
        $ python3 test01_trapz.py

Command 1: Trapezoidal line load integration: Python 2
-------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_trapz.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.269042 seconds
    * Cumulative time: 0.269042 seconds
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

Command 2: Trapezoidal line load integration: Python 3
-------------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_trapz.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.496174 seconds
    * Cumulative time: 0.765216 seconds
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

