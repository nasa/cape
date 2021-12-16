
.. This documentation written by TestDriver()
   on 2021-12-16 at 01:40 PST

Test ``06_atm``: PASS
=======================

This test PASSED on 2021-12-16 at 01:40 PST

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

Command 1: Standard Atmosphere: Python 2 (PASS)
------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_atm76.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.39 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        101325.00
        79498.14
        2187.96
        1.23
        332.50
        222.54
        

:STDERR:
    * **PASS**

Command 2: Standard Atmosphere: Python 3 (PASS)
------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_atm76.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.48 seconds
    * Cumulative time: 0.87 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        101325.00
        79498.14
        2187.96
        1.23
        332.50
        222.54
        

:STDERR:
    * **PASS**

Command 3: Sutherland's Law: Python 2 (PASS)
---------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_sutherlandMKS.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.45 seconds
    * Cumulative time: 1.32 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        1.79e-05
        

:STDERR:
    * **PASS**

Command 4: Sutherland's Law: Python 3 (PASS)
---------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_sutherlandMKS.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.51 seconds
    * Cumulative time: 1.83 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        1.79e-05
        

:STDERR:
    * **PASS**

Command 5: Temperature from Enthalpy: Python 2 (PASS)
------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test03_h.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.45 seconds
    * Cumulative time: 2.27 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        1.18e+03
        

:STDERR:
    * **PASS**

Command 6: Temperature from Enthalpy: Python 3 (PASS)
------------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test03_h.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.49 seconds
    * Cumulative time: 2.77 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        1.18e+03
        

:STDERR:
    * **PASS**

