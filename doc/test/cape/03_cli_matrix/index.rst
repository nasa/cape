
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``03_cli_matrix``: **FAIL** (command 2)
==============================================

This test **FAILED** (command 2) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/cape/03_cli_matrix/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ cape -c
        $ cape -c -f cape-json.json
        $ cape -c -f cape-json.json --re a0
        $ cape -c -f cape-mixed.json
        $ cape -c -f cape-mixed.json --re a2

Command 1: Missing JSON File (PASS)
------------------------------------

:Command:
    .. code-block:: console

        $ cape -c

:Return Code:
    * **PASS**
    * Output: ``1``
    * Target: ``1``
:Time Taken:
    * **PASS**
    * Command took 0.23 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**
    * Target:

      .. code-block:: pytb

        ValueError: No cape control file 'cape.json' found
        


Command 2: JSON-only Matrix (**FAIL**)
---------------------------------------

:Command:
    .. code-block:: console

        $ cape -c -f cape-json.json

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.21 seconds
    * Cumulative time: 0.44 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m0.80a0.0b0.0 ---     /           .            
        1    poweroff/m0.80a4.0b0.0 ---     /           .            
        2    poweroff/m0.90a0.0b0.0 ---     /           .            
        3    poweroff/m0.90a4.0b0.0 ---     /           .            
        4    poweroff/m1.05a0.0b0.0 ---     /           .            
        5    poweroff/m1.05a4.0b0.0 ---     /           .            
        
        ---=6, 
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "/u/wk/ddalle/usr/cape/bin/cape", line 5, in <module>
            from cape.cfdx.cli import main
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


