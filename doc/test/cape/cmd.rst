
.. _test-cape-cmd:

Testing :mod:`cape.cmd`: Command-line Call Generation
=====================================================

This section tests the :mod:`cape.cmd` module that forms command to generic
binaries.  Each flow solver module has a pair of modules :mod:`cmd` (which
forms command) and :mod:`bin` (which calls them).

.. testsetup:: *

    # System modules
    import os
    
    # Modules to import
    import cape.test
    import cape.case
    import cape.cmd
    
.. _test-cape-cmd-aflr3:

AFLR3 Commands
----------------
This function forms a call to ``aflr3``, a mesh generation tool, using two
methods.  The first version reads the ``case.json`` settings file while the
other uses keyword arguments.

.. testcode::

    # Case test folder
    fcape = os.path.join(cape.test.ftcape, "cmd")
    # Go to this folder.
    os.chdir(fcape)
    
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
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    somekey=c
    aflr3 -i pyfun.surf -o pyfun.lb8.ugrid -blr 10 -someflag 2 somekey=c

    
.. _test-cape-cmd-intersect:

Cart3D ``intersect`` Commands
-------------------------------
The Cart3D utility ``intersect`` is used to perform Boolean operations on
triangulated surfaces, but it can also be used with other surface grids (and is
in particular used by pyFun to create FUN3D meshes in some cases).  Therefore
the command generation is placed in the generic :mod:`cape` module rather than
:mod:`pyCart`.


.. testcode::

    # Case test folder
    fcape = os.path.join(cape.test.ftcape, "cmd")
    # Go to this folder.
    os.chdir(fcape)
    
    # Read settings
    rc = cape.case.ReadCaseJSON()
    
    # Form command
    cmd1 = cape.cmd.intersect(rc)
    # Alternate form
    cmd2 = cape.cmd.intersect(T=True)
    
    # Output
    print(' '.join(cmd1))
    print(' '.join(cmd2))
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    intersect -i Components.tri -o Components.i.tri -ascii -T
    intersect -i Components.tri -o Components.i.tri -ascii -T
