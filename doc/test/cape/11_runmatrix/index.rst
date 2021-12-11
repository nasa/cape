
.. This documentation written by TestDriver()
   on 2021-12-11 at 01:40 PST

Test ``11_runmatrix``: PASS
=============================

This test PASSED on 2021-12-11 at 01:40 PST

This test is run in the folder:

    ``test/cape/11_runmatrix/``

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

Command 1: Run Matrix w/o File: Python 2 (PASS)
------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_nofile.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.42 seconds
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

Command 2: Run Matrix w/o File: Python 3 (PASS)
------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_nofile.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.48 seconds
    * Cumulative time: 0.90 seconds
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

Command 3: Conversions: Python 2 (PASS)
----------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_conditions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.53 seconds
    * Cumulative time: 1.43 seconds
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

Command 4: Conversions: Python 3 (PASS)
----------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_conditions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.71 seconds
    * Cumulative time: 2.15 seconds
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

Command 5: Mark PASS (PASS)
----------------------------

:Command:
    .. code-block:: console

        $ cape -I 5 --PASS

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.57 seconds
    * Cumulative time: 2.71 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 6: Check PASS Status (PASS)
------------------------------------

:Command:
    .. code-block:: console

        $ cape -I 5 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.58 seconds
    * Cumulative time: 3.29 seconds
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

Command 7: Mark ERROR (PASS)
-----------------------------

:Command:
    .. code-block:: console

        $ cape -I 6 --ERROR

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.41 seconds
    * Cumulative time: 3.71 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 8: Check ERROR Status (PASS)
-------------------------------------

:Command:
    .. code-block:: console

        $ cape -I 6 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.44 seconds
    * Cumulative time: 4.15 seconds
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

Command 9: Overwrite PASS/ERROR Marks (PASS)
---------------------------------------------

:Command:
    .. code-block:: console

        $ cape -I 4:7 --ERROR

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.57 seconds
    * Cumulative time: 4.72 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 10: Check Final Marks (PASS)
-------------------------------------

:Command:
    .. code-block:: console

        $ cape -I 4:7 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.60 seconds
    * Cumulative time: 5.32 seconds
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

