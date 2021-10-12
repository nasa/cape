
.. This documentation written by TestDriver()
   on 2021-10-12 at 14:16 PDT

Test ``09_cmd``
=================

This test is run in the folder:

    ``/home/dalle/usr/cape/test/cape/09_cmd/``

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
    * Command took 0.167178 seconds
    * Cumulative time: 0.167178 seconds
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
    * Command took 0.168101 seconds
    * Cumulative time: 0.335279 seconds
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
    * Command took 0.255696 seconds
    * Cumulative time: 0.590974 seconds
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
    * Command took 0.255267 seconds
    * Cumulative time: 0.846241 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        intersect -i Components.tri -o Components.i.tri -ascii -T
        intersect -i Components.tri -o Components.i.tri -ascii -T
        

:STDERR:
    * **PASS**

