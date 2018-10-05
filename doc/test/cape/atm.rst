
.. _test-cape-atm:

Testing :mod:`cape.atm`: Standard Atmospheric Models
================================================================

This section tests the CAPE module that contains simple atmosphere models.
Many aerodynamic simulations assume the 1976 Standard Atmosphere as a
relationship between temperature, pressure, and density with altitude.  Thus
it's useful to have tools to access this common relationship.

    .. code-block:: python
    
        import cape.atm
        
        # Call atmosphere module at 2 km altitude
        s = cape.atm.atm76(2.0)
        # Show the static pressure
        print(s.p)

.. testsetup:: *
    
    # Modules to import
    import cape.atm
    
.. _test-atm-atm76:

Standard Atmosphere
--------------------
This tests the standard atmosphere at several altitudes.

.. testcode::
    
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
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    101325.00
    79498.14
    2187.96
    1.23
    332.50
    222.54
    

.. _test-atm-viscosity:

Viscosity
----------
Show the dynamic viscosity at sea level according to standard models.

.. testcode::

    # Get sea level conditions
    s0 = cape.atm.atm76(0.0)
    # Calculate viscosity
    mu0 = cape.atm.SutherlandMKS(s0.T)
    
    # Print results
    print("%0.2e" % mu0)
    
.. testoutput::

    1.79e-05
