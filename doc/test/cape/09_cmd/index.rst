
.. This documentation written by TestDriver()
   on 2019-08-29 at 09:48 PDT

Test ``09_cmd``
=================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/cape/09_cmd/``

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
        
        # Read settings
        rc = cape.case.ReadCaseJSON()
        
        # Form command
        cmd1 = cape.cmd.aflr3(rc)
        # Alternate form
        cmd2 = cape.cmd.aflr3(
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
        
        # Read settings
        rc = cape.case.ReadCaseJSON()
        
        # Form command
        cmd1 = cape.cmd.intersect(rc)
        # Alternate form
        cmd2 = cape.cmd.intersect(T=True)
        
        # Output
        print(' '.join(cmd1))
        print(' '.join(cmd2))

Command 1: AFLR3 Commands Python 2
-----------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_cmd.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.369757 seconds
    * Cumulative time: 0.369757 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        somekey=c
        aflr3 -i pyfun.surf -o pyfun.lb8.ugrid -blr 10 -someflag 2 somekey=c
        

:STDERR:
    * **PASS**

Command 2: CART3D ``intersect`` Commands Python 2
--------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_cmd.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.371401 seconds
    * Cumulative time: 0.741158 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        intersect -i Components.tri -o Components.i.tri -ascii -T
        intersect -i Components.tri -o Components.i.tri -ascii -T
        

:STDERR:
    * **PASS**

Command 3: AFLR3 Commands Python 3
-----------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_cmd.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.731934 seconds
    * Cumulative time: 1.47309 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        somekey=c
        aflr3 -i pyfun.surf -o pyfun.lb8.ugrid -blr 10 -someflag 2 somekey=c
        

:STDERR:
    * **PASS**

Command 4: CART3D ``intersect`` Commands Python 3
--------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_cmd.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.71848 seconds
    * Cumulative time: 2.19157 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        intersect -i Components.tri -o Components.i.tri -ascii -T
        intersect -i Components.tri -o Components.i.tri -ascii -T
        

:STDERR:
    * **PASS**

