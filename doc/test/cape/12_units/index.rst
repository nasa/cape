
.. This documentation written by TestDriver()
   on 2022-03-08 at 01:40 PST

Test ``12_units``: PASS
=========================

This test PASSED on 2022-03-08 at 01:40 PST

This test is run in the folder:

    ``test/cape/12_units/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_units.py
        $ python3 test01_units.py

**Included file:** ``test01_units.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        from cape.units import mks
        
        # Basic Tests
        
        # Inch to meter conversions
        n = 1
        u1 = "in"
        u2 = "m"
        v1 = 1.0
        s1 = "inch"
        # Status update
        print("%02i (length): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.5f" % mks(s1))
        
        
        # Horsepower and prefix
        n = 2
        u1 = "khp"
        u2 = "W"
        v1 = 1.0
        s1 = "khp"
        # Status update
        print("%02i (imperial w/ mks prefix): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.5f" % mks(s1))
        
        
        # Hectares
        n = 3
        u1 = "hectare"
        u2 = "m^2"
        v1 = 1.0
        s1 = "hectare"
        # Status update
        print("%02i (area): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.3f" % mks(s1))
        
        
        # Speeds
        n = 4
        u1 = "ft/mhr"
        u2 = "m/s"
        v1 = 1.0
        s1 = "ft/mhr"
        # Status update
        print("%02i (slash): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.5f" % mks(s1))
        
        
        # Unicode
        n = 5
        u1 = "um"
        u2 = "m"
        v1 = 1.0
        s1 = u"μm"
        # Status update
        print("%02i (micro): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.2e" % mks(s1))
        
        
        # Dual prefix
        n = 6
        u1 = "nGm"
        u2 = "m"
        v1 = 1.0
        s1 = "nGm"
        # Status update
        print("%02i (double-prefix): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.2e" % mks(s1))
        
        
        # Asterisk
        n = 7
        u1 = "ft*lbf"
        u2 = "N*m"
        v1 = 5.0
        s1 = u1
        # Status update
        print("%02i (asterisk): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.4f" % (v1*mks(s1)))
        
        
        # Space
        n = 8
        u1 = "ft*lbf"
        u2 = "N*m"
        v1 = 5.0
        s1 = "ft lbf"
        # Status update
        print("%02i (space): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.4f" % (v1*mks(s1)))
        
        
        # Carat
        n = 9
        u1 = "in^3"
        u2 = "L"
        v1 = 100.0
        s1 = "in^3/L"
        # Status update
        print("%02i (exponent): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.4f" % (v1*mks(s1)))
        
        
        # Included number
        n = 10
        u1 = "slug"
        u2 = "kg"
        v1 = 12.0
        s1 = "12 slug"
        # Status update
        print("%02i (number): %.2f %s --> %s" % (n, v1, u1, u2))
        # Display conversion
        print("  %.4f" % (mks(s1)))
        

Command 1: Unit Conversions: Python 2 (PASS)
---------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_units.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.51 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        01 (length): 1.00 in --> m
          0.02540
        02 (imperial w/ mks prefix): 1.00 khp --> W
          745699.87238
        03 (area): 1.00 hectare --> m^2
          10000.000
        04 (slash): 1.00 ft/mhr --> m/s
          0.08467
        05 (micro): 1.00 um --> m
          1.00e-06
        06 (double-prefix): 1.00 nGm --> m
          1.00e+00
        07 (asterisk): 5.00 ft*lbf --> N*m
          6.7791
        08 (space): 5.00 ft*lbf --> N*m
          6.7791
        09 (exponent): 100.00 in^3 --> L
          1.6387
        10 (number): 12.00 slug --> kg
          175.1268
        

:STDERR:
    * **PASS**

Command 2: Unit Conversions: Python 3 (PASS)
---------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_units.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.58 seconds
    * Cumulative time: 1.09 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        01 (length): 1.00 in --> m
          0.02540
        02 (imperial w/ mks prefix): 1.00 khp --> W
          745699.87238
        03 (area): 1.00 hectare --> m^2
          10000.000
        04 (slash): 1.00 ft/mhr --> m/s
          0.08467
        05 (micro): 1.00 um --> m
          1.00e-06
        06 (double-prefix): 1.00 nGm --> m
          1.00e+00
        07 (asterisk): 5.00 ft*lbf --> N*m
          6.7791
        08 (space): 5.00 ft*lbf --> N*m
          6.7791
        09 (exponent): 100.00 in^3 --> L
          1.6387
        10 (number): 12.00 slug --> kg
          175.1268
        

:STDERR:
    * **PASS**

