
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``08_cntl``: **FAIL** (command 1)
========================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/cape/08_cntl/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_cntl.py
        $ python3 test01_cntl.py

**Included file:** ``test01_cntl.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape
        
        # Initiate object
        cntl = cape.Cntl()
        
        # print results
        print(cntl)

Command 1: Python 2 (**FAIL**)
-------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_cntl.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.13 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        Importing module 'dac'
          InitFunction: dac.InitCntl()
        <cape.Cntl(nCase=20)>
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_cntl.py", line 5, in <module>
            import cape
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


