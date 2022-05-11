
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``02_adiabatic``: **FAIL** (command 1)
=============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/pyfun/02_adiabatic/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyfun -f pyFun01.json -I 0 --no-start
        $ cat bullet/m0.80a0.0b0.0/fun3d.00.nml
        $ pyfun -f pyFun02.json -I 1 --no-start
        $ cat bullet/m0.80a4.0b0.0/fun3d.00.nml
        $ pyfun -f pyFun03.json -I 2 --no-start
        $ cat bullet/m0.80a10.0b0.0/fun3d.00.nml

Command 1: Create Input Files (**FAIL**)
-----------------------------------------

:Command:
    .. code-block:: console

        $ pyfun -f pyFun01.json -I 0 --no-start

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.25 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "/u/wk/ddalle/usr/cape/bin/pyfun", line 5, in <module>
            from cape.pyfun.cli import main
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


