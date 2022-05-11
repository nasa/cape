
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``09_cmd``: **FAIL** (command 1)
=======================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/cape/09_cmd/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_cmd.py
        $ python2 test02_cmd.py
        $ python3 test01_cmd.py
        $ python3 test02_cmd.py

**Included file:** ``test01_cmd.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape
        import cape.cfdx.cmd
        import cape.cfdx.case
        
        # Read settings
        rc = cape.cfdx.case.ReadCaseJSON()
        
        # Form command
        cmd1 = cape.cfdx.cmd.aflr3(rc)
        # Alternate form
        cmd2 = cape.cfdx.cmd.aflr3(
            i="pyfun.surf",
            o="pyfun.lb8.ugrid",
            blr=10,
            flags={"someflag": 2},
            keys={"somekey": 'c'})
        
        # Output
        print(cmd1[-1])
        print(' '.join(cmd2))

**Included file:** ``test02_cmd.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape
        import cape.cfdx.case
        import cape.cfdx.cmd
        
        # Read settings
        rc = cape.cfdx.case.ReadCaseJSON()
        
        # Form command
        cmd1 = cape.cfdx.cmd.intersect(rc)
        # Alternate form
        cmd2 = cape.cfdx.cmd.intersect(T=True)
        
        # Output
        print(' '.join(cmd1))
        print(' '.join(cmd2))

Command 1: AFLR3 Commands Python 2 (**FAIL**)
----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_cmd.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.14 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        somekey=c
        aflr3 -i pyfun.surf -o pyfun.lb8.ugrid -blr 10 -someflag 2 somekey=c
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_cmd.py", line 5, in <module>
            import cape
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


