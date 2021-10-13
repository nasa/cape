
.. This documentation written by TestDriver()
   on 2021-10-13 at 11:14 PDT

Test ``02_bullet_py3``: **FAIL** (command 1)
==============================================

This test **FAILED** (command 1) on 2021-10-13 at 11:14 PDT

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

Command 1: Run Matrix Status (**FAIL**)
----------------------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pycart -c

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.80 seconds
:STDOUT:
    * **FAIL**
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
          File "/u/wk/ddalle/usr/cape/cape/pycart/cli.py", line 43, in main
            if cmd.lower() in {"run_flowcart", "run_cart3d", "run"}:
        AttributeError: 'NoneType' object has no attribute 'lower'
        


