
.. This documentation written by TestDriver()
   on 2019-10-23 at 12:30 PDT

Test ``11_runmatrix``
=======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/cape/11_runmatrix/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_nofile.py
        $ python3 test01_nofile.py
        $ python2 test02_conditions.py
        $ python3 test02_conditions.py
        $ cape -I 5 --PASS
        $ cape -I 5 -c
        $ cape -I 6 --ERROR
        $ cape -I 6 -c
        $ cape -I 4:7 --ERROR
        $ cape -I 4:7 -c

:Included Files:
    * :download:`test01_nofile.py`
    * :download:`test02_conditions.py`

Command 1: Run Matrix w/o File: Python 2
-----------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_nofile.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.46535 seconds
    * Cumulative time: 0.46535 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        01: Folder names
        m1.4a0.0b0.0
        m1.4a4.0b0.0
        m1.4a0.0b4.0
        m1.4a4.0b4.0
        02: Full folder names
        poweroff/m1.4a0.0b0.0
        poweroff/m1.4a4.0b0.0
        poweroff/m1.4a0.0b4.0
        poweroff/m1.4a4.0b4.0
        03: Total angle of attack
        aoap = 0.00, phip =  0.00
        aoap = 4.00, phip =  0.00
        aoap = 4.00, phip = 90.00
        aoap = 5.65, phip = 45.07
        04: Modified format
        m1.40_a0.0b4.0
        05: Conditions JSON file
        {
         "mach": 1.4,
         "alpha": 4.0,
         "beta": 0.0,
         "config": "poweroff"
        }
        

:STDERR:
    * **PASS**

Command 2: Run Matrix w/o File: Python 3
-----------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_nofile.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.709334 seconds
    * Cumulative time: 1.17468 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        01: Folder names
        m1.4a0.0b0.0
        m1.4a4.0b0.0
        m1.4a0.0b4.0
        m1.4a4.0b4.0
        02: Full folder names
        poweroff/m1.4a0.0b0.0
        poweroff/m1.4a4.0b0.0
        poweroff/m1.4a0.0b4.0
        poweroff/m1.4a4.0b4.0
        03: Total angle of attack
        aoap = 0.00, phip =  0.00
        aoap = 4.00, phip =  0.00
        aoap = 4.00, phip = 90.00
        aoap = 5.65, phip = 45.07
        04: Modified format
        m1.40_a0.0b4.0
        05: Conditions JSON file
        {
         "mach": 1.4,
         "alpha": 4.0,
         "beta": 0.0,
         "config": "poweroff"
        }
        

:STDERR:
    * **PASS**

Command 3: Conversions: Python 2
---------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_conditions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.495985 seconds
    * Cumulative time: 1.67067 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        01: aoap, phip --> a, b
        0: aoap=0.0, phip=  0.0 -> a= 0.0000, b= 0.0000
        1: aoap=4.0, phip=  0.0 -> a= 4.0000, b= 0.0000
        2: aoap=4.0, phip= 45.0 -> a= 2.8307, b= 2.8273
        3: aoap=4.0, phip= 90.0 -> a= 0.0000, b= 4.0000
        4: aoap=4.0, phip=235.0 -> a=-2.2968, b=-3.2757
        02: a, b --> aoap, phip
        0: a= 0.0, b= 0.0 -> aoap=0.00, phip=  0.00
        1: a= 4.0, b= 0.0 -> aoap=4.00, phip=  0.00
        2: a= 4.0, b= 4.0 -> aoap=5.65, phip= 45.07
        3: a=-4.0, b=-2.0 -> aoap=4.47, phip=206.59
        03: a, b --> aoav, phiv
        0: a= 0.0, b= 0.0 -> aoav= 0.00, phiv=  0.00
        1: a= 4.0, b= 0.0 -> aoav= 4.00, phiv=  0.00
        2: a= 4.0, b= 4.0 -> aoav= 5.65, phiv= 45.07
        3: a=-4.0, b=-2.0 -> aoav=-4.47, phiv= 26.59
        04: mach, q --> p, p0
        0: mach=2.00, q=100.0 psf -> p= 35.71, p0=279.44
        1: mach=2.00, q=250.0 psf -> p= 89.29, p0=698.61
        2: mach=2.00, q=300.0 psf -> p=107.14, p0=838.33
        04: mach, q --> p, p0
        0: mach=2.00, q=100.0 psf  T=450.0 R -> Rey=23996.5/in
        1: mach=2.00, q=250.0 psf  T=450.0 R -> Rey=59991.2/in
        2: mach=2.00, q=300.0 psf  T=450.0 R -> Rey=71989.5/in
        

:STDERR:
    * **PASS**

Command 4: Conversions: Python 3
---------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_conditions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.732605 seconds
    * Cumulative time: 2.40327 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        01: aoap, phip --> a, b
        0: aoap=0.0, phip=  0.0 -> a= 0.0000, b= 0.0000
        1: aoap=4.0, phip=  0.0 -> a= 4.0000, b= 0.0000
        2: aoap=4.0, phip= 45.0 -> a= 2.8307, b= 2.8273
        3: aoap=4.0, phip= 90.0 -> a= 0.0000, b= 4.0000
        4: aoap=4.0, phip=235.0 -> a=-2.2968, b=-3.2757
        02: a, b --> aoap, phip
        0: a= 0.0, b= 0.0 -> aoap=0.00, phip=  0.00
        1: a= 4.0, b= 0.0 -> aoap=4.00, phip=  0.00
        2: a= 4.0, b= 4.0 -> aoap=5.65, phip= 45.07
        3: a=-4.0, b=-2.0 -> aoap=4.47, phip=206.59
        03: a, b --> aoav, phiv
        0: a= 0.0, b= 0.0 -> aoav= 0.00, phiv=  0.00
        1: a= 4.0, b= 0.0 -> aoav= 4.00, phiv=  0.00
        2: a= 4.0, b= 4.0 -> aoav= 5.65, phiv= 45.07
        3: a=-4.0, b=-2.0 -> aoav=-4.47, phiv= 26.59
        04: mach, q --> p, p0
        0: mach=2.00, q=100.0 psf -> p= 35.71, p0=279.44
        1: mach=2.00, q=250.0 psf -> p= 89.29, p0=698.61
        2: mach=2.00, q=300.0 psf -> p=107.14, p0=838.33
        04: mach, q --> p, p0
        0: mach=2.00, q=100.0 psf  T=450.0 R -> Rey=23996.5/in
        1: mach=2.00, q=250.0 psf  T=450.0 R -> Rey=59991.2/in
        2: mach=2.00, q=300.0 psf  T=450.0 R -> Rey=71989.5/in
        

:STDERR:
    * **PASS**

Command 5: Mark PASS
---------------------

:Command:
    .. code-block:: console

        $ cape -I 5 --PASS

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.510295 seconds
    * Cumulative time: 2.91357 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 6: Check PASS Status
-----------------------------

:Command:
    .. code-block:: console

        $ cape -I 5 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.525092 seconds
    * Cumulative time: 3.43866 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        5    poweroff/m1.1a2.0b0.0 PASS*   /           .            
        
        PASS*=1, 
        

:STDERR:
    * **PASS**

Command 7: Mark ERROR
----------------------

:Command:
    .. code-block:: console

        $ cape -I 6 --ERROR

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.544317 seconds
    * Cumulative time: 3.98298 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 8: Check ERROR Status
------------------------------

:Command:
    .. code-block:: console

        $ cape -I 6 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.539197 seconds
    * Cumulative time: 4.52218 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        6    poweroff/m1.5a0.0b0.0 ERROR   /           .            
        
        ERROR=1, 
        

:STDERR:
    * **PASS**

Command 9: Overwrite PASS/ERROR Marks
--------------------------------------

:Command:
    .. code-block:: console

        $ cape -I 4:7 --ERROR

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.520579 seconds
    * Cumulative time: 5.04275 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 10: Check Final Marks
------------------------------

:Command:
    .. code-block:: console

        $ cape -I 4:7 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.527084 seconds
    * Cumulative time: 5.56984 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        4    poweroff/m1.1a0.0b0.0 ERROR   /           .            
        5    poweroff/m1.1a2.0b0.0 ERROR   /           .            
        6    poweroff/m1.5a0.0b0.0 ERROR   /           .            
        
        ERROR=3, 
        

:STDERR:
    * **PASS**

