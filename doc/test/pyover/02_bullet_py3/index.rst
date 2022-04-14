
.. This documentation written by TestDriver()
   on 2022-04-14 at 01:44 PDT

Test ``02_bullet_py3``: **FAIL** (command 3)
==============================================

This test **FAILED** (command 3) on 2022-04-14 at 01:44 PDT

This test is run in the folder:

    ``test/pyover/02_bullet_py3/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python3 -m cape.pyover -I 1
        $ python3 -m cape.pyover -I 1 -c
        $ python3 -m cape.pyover -I 1 --fm
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Case 1 (PASS)
-----------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pyover -I 1

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 95.50 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 ---     /           .            
          Case name: 'poweroff/m0.8a4.0b0.0' (index 1)
             Starting case 'poweroff/m0.8a4.0b0.0'
         > overrunmpi -np 24 run 01
             (PWD = 'poweroff/m0.8a4.0b0.0/')
             (STDOUT = 'overrun.out')
           Wall time used: 0.01 hrs (phase 0)
           Wall time used: 0.01 hrs
           Previous phase: 0.01 hrs
         > overrunmpi -np 24 run 02
             (PWD = 'poweroff/m0.8a4.0b0.0/')
             (STDOUT = 'overrun.out')
           Wall time used: 0.01 hrs (phase 1)
           Wall time used: 0.02 hrs
           Previous phase: 0.01 hrs
         > overrunmpi -np 24 run 03
             (PWD = 'poweroff/m0.8a4.0b0.0/')
             (STDOUT = 'overrun.out')
           Wall time used: 0.01 hrs (phase 2)
        
        Submitted or ran 1 job(s).
        
        ---=1, 
        

:STDERR:
    * **PASS**

Command 2: Show DONE Status (PASS)
-----------------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pyover -I 1 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.62 seconds
    * Cumulative time: 96.12 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 DONE    1500/1500   .        0.6 
        
        DONE=1, 
        

    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 DONE    1500/1500   .   ...
        
        DONE=1, 
        

:STDERR:
    * **PASS**

Command 3: Collect Aero (**FAIL**)
-----------------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pyover -I 1 --fm

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.58 seconds
    * Cumulative time: 96.70 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "/usr/lib64/python3.6/runpy.py", line 193, in _run_module_as_main
            "__main__", mod_spec)
          File "/usr/lib64/python3.6/runpy.py", line 85, in _run_code
            exec(code, run_globals)
          File "/u/wk/ddalle/usr/cape/cape/pyover/__main__.py", line 12, in <module>
            sys.exit(cli.main())
          File "/u/wk/ddalle/usr/cape/cape/pyover/cli.py", line 62, in main
            cntl.cli(*a, **kw)
          File "/u/wk/ddalle/usr/cape/cape/pyover/cntl.py", line 189, in cli
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
        


