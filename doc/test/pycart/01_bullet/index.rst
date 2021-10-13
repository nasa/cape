
.. This documentation written by TestDriver()
   on 2021-10-13 at 11:14 PDT

Test ``01_bullet``: **FAIL** (command 1)
==========================================

This test **FAILED** (command 1) on 2021-10-13 at 11:14 PDT

This test is run in the folder:

    ``test/pycart/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pycart -c
        $ pycart -I 0
        $ pycart -I 0 --aero
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Matrix Status (**FAIL**)
----------------------------------------

:Command:
    .. code-block:: console

        $ pycart -c

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.86 seconds
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
          File "/u/wk/ddalle/usr/cape/bin/pycart", line 8, in <module>
            sys.exit(main())
          File "/u/wk/ddalle/usr/cape/cape/pycart/cli.py", line 43, in main
            if cmd.lower() in {"run_flowcart", "run_cart3d", "run"}:
        AttributeError: 'NoneType' object has no attribute 'lower'
        


