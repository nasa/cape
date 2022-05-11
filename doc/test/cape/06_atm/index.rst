
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``06_atm``: **FAIL** (command 1)
=======================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

This test is run in the folder:

    ``test/cape/06_atm/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_atm76.py
        $ python3 test01_atm76.py
        $ python2 test02_sutherlandMKS.py
        $ python3 test02_sutherlandMKS.py
        $ python2 test03_h.py
        $ python3 test03_h.py

**Included file:** ``test01_atm76.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.atm
        
        # Set altitudes (in km)
        h0 = 0.0
        h1 = 2.0
        h2 = 26.0
        
        # Call the standard atmosphere
        s0 = cape.atm.atm76(h0)
        s1 = cape.atm.atm76(h1)
        s2 = cape.atm.atm76(h2)
        
        # Print results
        print("%0.2f" % s0.p)
        print("%0.2f" % s1.p)
        print("%0.2f" % s2.p)
        print("%0.2f" % s0.rho)
        print("%0.2f" % s1.a)
        print("%0.2f" % s2.T)

**Included file:** ``test02_sutherlandMKS.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.atm
        
        # Get sea level conditions
        s0 = cape.atm.atm76(0.0)
        # Calculate viscosity
        mu0 = cape.atm.SutherlandMKS(s0.T)
        
        # Print results
        print("%0.2e" % mu0)

**Included file:** ``test03_h.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.atm
        
        # Pick a reference specific enthalpy [J/kg]
        href = 1.2000e+06
        
        # Calculate temperature [K]
        Tref = cape.atm.get_T(href)
        
        # Print results
        print("%0.2e" % Tref)

Command 1: Standard Atmosphere: Python 2 (**FAIL**)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_atm76.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.10 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

        101325.00
        79498.14
        2187.96
        1.23
        332.50
        222.54
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_atm76.py", line 5, in <module>
            import cape.atm
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


