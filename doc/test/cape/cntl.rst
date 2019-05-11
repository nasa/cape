
.. _test-cape-cntl:

Testing :mod:`cape.cntl`: Primary Control Interface
=====================================================

This section tests the :mod:`cape.cntl` module that forms the basis for the
main control classes for each specific solver.  It provides a class
:class:`cape.cntl.Cntl`, which has many methods.  Many of these methods are
either used unmodified or the basis for the methods that operate each
individual solver.

.. testsetup:: *

    # System modules
    import os
    
    # Modules to import
    import cape.test
    import cape.cntl
    import cape
    
.. _test-cape-cntl-mod:

Cntl Module Commands
------------------------
This test imports a JSON file that includes a reference to another JSON file,
loads a module in the ``tools/`` folder, and calls an initialization function
that modifies the original settings from JSON.

.. testcode::

    # Case test folder
    fcape = os.path.join(cape.test.ftcape, "cntl")
    # Go to this folder.
    os.chdir(fcape)
    
    # Initiate object
    cntl = cape.Cntl()
    
    # Show the thing
    print(cntl)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Importing module 'dac'
      InitFunction: dac.InitCntl()
    <cape.Cntl(nCase=20)>

