
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``04_subset_api``: **FAIL** (command 1)
==============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/cape/04_subset_api/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_subset.py
        $ python3 test01_subset.py

**Included file:** ``test01_subset.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape
        
        # Load interface
        cntl = cape.Cntl()
        
        # Display subsets
        print(list(cntl.x.FilterString("b2")))
        print(list(cntl.x.FilterWildcard("poweroff/m0*")))
        print(list(cntl.x.FilterRegex("m.\.5.*b2")))
        print(list(cntl.x.GetIndices(I=range(15,20), cons=["Mach%1==0.5"])))
        

Command 1: Python 2 (**FAIL**)
-------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_subset.py

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

        [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]
        [0, 1, 2, 3, 4, 5, 6, 7]
        [2, 3, 14, 15, 18, 19]
        [15, 16, 17, 18, 19]
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_subset.py", line 5, in <module>
            import cape
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


