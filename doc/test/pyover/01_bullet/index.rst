
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``01_bullet``: **FAIL** (command 1)
==========================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/pyover/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyover -I 1
        $ pyover -I 1 -c
        $ pyover -I 1 --fm
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Case 1 (**FAIL**)
---------------------------------

:Command:
    .. code-block:: console

        $ pyover -I 1

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.21 seconds
:STDOUT:
    * **PASS**
    * Actual: (empty)
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "/u/wk/ddalle/usr/cape/bin/pyover", line 5, in <module>
            from cape.pyover.cli import main
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


