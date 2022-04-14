
.. This documentation written by TestDriver()
   on 2022-04-14 at 01:41 PDT

Test ``01_bullet``: **FAIL** (command 3)
==========================================

This test **FAILED** (command 3) on 2022-04-14 at 01:41 PDT

This test is run in the folder:

    ``test/pyfun/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyfun -I 8
        $ pyfun -I 8 -c
        $ pyfun -I 8 --fm
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Case 8 (PASS)
-----------------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 45.79 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        8    bullet/m1.10a0.0b0.0  ---     /           .            
          Case name: 'bullet/m1.10a0.0b0.0' (index 8)
             Starting case 'bullet/m1.10a0.0b0.0'
         > nodet --animation_freq 100
             (PWD = 'bullet/m1.10a0.0b0.0/')
             (STDOUT = 'fun3d.out')
         > nodet --animation_freq 100
             (PWD = 'bullet/m1.10a0.0b0.0/')
             (STDOUT = 'fun3d.out')
        
        Submitted or ran 1 job(s).
        
        ---=1, 
        

:STDERR:
    * **PASS**

Command 2: Show DONE Status (PASS)
-----------------------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.81 seconds
    * Cumulative time: 46.60 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        8    bullet/m1.10a0.0b0.0  DONE    200/200     .        0.1 
        
        DONE=1, 
        

:STDERR:
    * **PASS**

Command 3: Collect Aero (**FAIL**)
-----------------------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8 --fm

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.91 seconds
    * Cumulative time: 47.51 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "/u/wk/ddalle/usr/cape/bin/pyfun", line 8, in <module>
            sys.exit(main())
          File "/u/wk/ddalle/usr/cape/cape/pyfun/cli.py", line 62, in main
            cntl.cli(*a, **kw)
          File "/u/wk/ddalle/usr/cape/cape/pyfun/cntl.py", line 232, in cli
            cmd = self.cli_cape(*a, **kw)
          File "/u/wk/ddalle/usr/cape/cape/cntl.py", line 802, in cli_cape
            self.UpdateFM(**kw)
          File "/u/wk/ddalle/usr/cape/cape/cntl.py", line 100, in wrapper_func
            v = func(self, *args, **kwargs)
          File "/u/wk/ddalle/usr/cape/cape/cntl.py", line 4045, in UpdateFM
            self.DataBook.UpdateDataBook(I, comp=comp)
          File "/u/wk/ddalle/usr/cape/cape/cfdx/dataBook.py", line 749, in UpdateDataBook
            self[comp].Write(merge=True, unlock=True)
          File "/u/wk/ddalle/usr/cape/cape/cfdx/dataBook.py", line 3416, in Write
            DBc = self.ReadCopy(check=True, lock=True)
          File "/u/wk/ddalle/usr/cape/cape/cfdx/dataBook.py", line 3134, in ReadCopy
            DBc = self.__class__(name, self.cntl, check=check, lock=lock)
        AttributeError: 'DBComp' object has no attribute 'cntl'
        


