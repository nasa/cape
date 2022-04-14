
.. This documentation written by TestDriver()
   on 2022-04-14 at 01:40 PDT

Test ``02_bullet_py3``: **FAIL** (command 3)
==============================================

This test **FAILED** (command 3) on 2022-04-14 at 01:40 PDT

This test is run in the folder:

    ``test/pycart/02_bullet_py3/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python3 -m cape.pycart -c
        $ python3 -m cape.pycart -I 0
        $ python3 -m cape.pycart -I 0 --aero
        $ python3 test_databook.py

Command 1: Run Matrix Status (PASS)
------------------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pycart -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.70 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 ---     /           .            
        1    poweroff/m2.0a0.0b0.0 ---     /           .            
        2    poweroff/m2.0a2.0b0.0 ---     /           .            
        3    poweroff/m2.0a2.0b2.0 ---     /           .            
        
        ---=4, 
        

:STDERR:
    * **PASS**

Command 2: Run Case 0 (PASS)
-----------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pycart -I 0

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 6.16 seconds
    * Cumulative time: 6.86 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 3: Collect Aero Data (**FAIL**)
----------------------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pycart -I 0 --aero

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.66 seconds
    * Cumulative time: 7.53 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        FM component 'bullet_no_base'...
        poweroff/m1.5a0.0b0.0
          Adding new databook entry at iteration 200.
        Writing 1 new or updated entries
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "/usr/lib64/python3.6/runpy.py", line 193, in _run_module_as_main
            "__main__", mod_spec)
          File "/usr/lib64/python3.6/runpy.py", line 85, in _run_code
            exec(code, run_globals)
          File "/u/wk/ddalle/usr/cape/cape/pycart/__main__.py", line 12, in <module>
            sys.exit(cli.main())
          File "/u/wk/ddalle/usr/cape/cape/pycart/cli.py", line 62, in main
            cntl.cli(*a, **kw)
          File "/u/wk/ddalle/usr/cape/cape/pycart/cntl.py", line 178, in cli
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
        


