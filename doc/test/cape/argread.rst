
.. _test-cape-argread:

Testing :mod:`cape.argread`: Command-Line Argument Parsing
================================================================

This section tests the CAPE module for parsing command-line inputs.  For
instance the command

    .. code-block:: console
    
        $ cape -c --filter b2
        
would be parsed into a dictionary

    .. code-block:: python
    
        {
            "c": True,
            "filter": "b2"
        }

A typical command-line parsing call is shown below.

    .. code-block:: python
    
        import os, sys
        import cape.argread
        
        # Parse inputs
        a, kw = cape.argread.readkeys(sys.argv)

.. testsetup:: *
    
    # Modules to import
    import cape.argread
    
.. _test-readkeys:

Basic Keys
------------
A basic call that contains an argument and both kinds of options

.. testcode::

    # List of arguments
    argv = ['cape', 'run', '-c', '--filter', 'b2']
    
    # Process it
    a, kw = cape.argread.readkeys(argv)
    # Print results
    print(a)
    print(kw)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    ['run']
    {'filter': 'b2', 'c': True, '_old': []}

Now let's try that again without the double dash.  The results should be
unchanged.

.. testcode::

    # List of arguments
    argv = ['cape', 'run', '-c', '-filter', 'b2']
    
    # Process it
    a, kw = cape.argread.readkeys(argv)
    # Print results
    print(a)
    print(kw)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    ['run']
    {'filter': 'b2', 'c': True, '_old': []}
    
Finally, let's repeat an input.  In practice the only good use for this in CAPE
is to use ``-x`` multiple times to apply more than one global scripts.

.. testcode::

    # List of arguments
    argv = ['cape', '--aero', 'body', '--aero', 'base', '--aero', 'fin', '-c']
    
    # Process it
    a, kw = cape.argread.readkeys(argv)
    # Print results
    print(a)
    print(kw)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    []
    {'c': True, 'aero': 'fin', '_old': [{'aero': 'body'}, {'aero': 'base'}]}
    
.. _test-readflags:

Read As Flags
---------------
A basic call that contains an argument and both kinds of options

.. testcode::

    # List of arguments
    argv = ['cape', 'run', '-c', '--filter', 'b2']
    
    # Process it
    a, kw = cape.argread.readflags(argv)
    # Print results
    print(a)
    print(kw)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    ['run']
    {'filter': 'b2', 'c': True, '_old': []}

Now let's try that again without the double dash.  The ``-fr`` flag should be
interpreted as ``-f -r``.

.. testcode::

    # List of arguments
    argv = ['cape', 'run', '-c', '-fr', 'b2']
    
    # Process it
    a, kw = cape.argread.readflags(argv)
    # Print results
    print(a)
    print(kw)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    ['run', 'b2']
    {'c': True, 'r': True, '_old': [], 'f': True}
    
.. _test-readflagstar:

Read Flags like ``tar`` Command
---------------------------------
This command is like :func:`readflags`, but the last flag can take a following
argument when only one hyphen is used.

.. testcode::

    # List of arguments
    argv = ['cape', 'run', '-c', '-fr', 'b2']
    
    # Process it
    a, kw = cape.argread.readflagstar(argv)
    # Print results
    print(a)
    print(kw)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    ['run']
    {'c': True, 'r': 'b2', '_old': [], 'f': True}

