
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``02_cli``: **FAIL** (command 1)
=======================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/cape/02_cli/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ cape -c
        $ cape -c --filter b2
        $ cape -c --cons 'beta==2,Mach%1==0.5'
        $ cape -c --glob 'poweroff/m0*'
        $ cape -c --re 'm.\.5.*b2'
        $ cape -c -I 2:5,7,18:
        $ cape -c -I 15: --cons Mach%1=0.5 --re b2

Command 1: Status (**FAIL**)
-----------------------------

:Command:
    .. code-block:: console

        $ cape -c

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.26 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m0.5a0.0b0.0 ---     /           .            
        1    poweroff/m0.5a2.0b0.0 ---     /           .            
        2    poweroff/m0.5a0.0b2.0 ---     /           .            
        3    poweroff/m0.5a2.0b2.0 ---     /           .            
        4    poweroff/m0.8a0.0b0.0 ---     /           .            
        5    poweroff/m0.8a2.0b0.0 ---     /           .            
        6    poweroff/m0.8a0.0b2.0 ---     /           .            
        7    poweroff/m0.8a2.0b2.0 ---     /           .            
        8    poweroff/m1.1a0.0b0.0 ---     /           .            
        9    poweroff/m1.1a2.0b0.0 ---     /           .            
        10   poweroff/m1.1a0.0b2.0 ---     /           .            
        11   poweroff/m1.1a2.0b2.0 ---     /           .            
        12   poweroff/m1.5a0.0b0.0 ---     /           .            
        13   poweroff/m1.5a2.0b0.0 ---     /           .            
        14   poweroff/m1.5a0.0b2.0 ---     /           .            
        15   poweroff/m1.5a2.0b2.0 ---     /           .            
        16   poweroff/m2.5a0.0b0.0 ---     /           .            
        17   poweroff/m2.5a2.0b0.0 ---     /           .            
        18   poweroff/m2.5a0.0b2.0 ---     /           .            
        19   poweroff/m2.5a2.0b2.0 ---     /           .            
        
        ---=20, 
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "/u/wk/ddalle/usr/cape/bin/cape", line 5, in <module>
            from cape.cfdx.cli import main
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


