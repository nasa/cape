
.. This documentation written by TestDriver()
   on 2019-11-25 at 13:11 PST

Test ``05_argread``
=====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/cape/05_argread/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_argread.py
        $ python3 test01_argread.py

**Included file:** ``test01_argread.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape
        
        # Basic Keys
        # Test 1
        # List of arguments
        argv = ['cape', 'run', '-c', '--filter', 'b2']
        
        # Process it
        a, kw = cape.argread.readkeys(argv)
        # Print results
        print(a)
        print(kw)
        
        # Test 2
        # List of arguments
        argv = ['cape', 'run', '-c', '-filter', 'b2']
        
        # Process it
        a, kw = cape.argread.readkeys(argv)
        # Print results
        print(a)
        print(kw)
        
        # Test 3
        # List of arguments
        argv = ['cape', '--aero', 'body', '--aero', 'base', '--aero', 'fin', '-c']
        
        # Process it
        a, kw = cape.argread.readkeys(argv)
        # Print results
        print(a)
        print(kw)
        
        # Read as Flags
        # Test 4
        # List of arguments
        argv = ['cape', 'run', '-c', '--filter', 'b2']
        
        # Process it
        a, kw = cape.argread.readflags(argv)
        # Print results
        print(a)
        print(kw)
        
        # Test 5
        # List of arguments
        argv = ['cape', 'run', '-c', '-fr', 'b2']
        
        # Process it
        a, kw = cape.argread.readflags(argv)
        # Print results
        print(a)
        print(kw)
        
        # Read flags like tar command
        # Test 6
        # List of arguments
        argv = ['cape', 'run', '-c', '-fr', 'b2']
        
        # Process it
        a, kw = cape.argread.readflagstar(argv)
        # Print results
        print(a)
        print(kw)

Command 1: Python 2
--------------------

:Command:
    .. code-block:: console

        $ python2 test01_argread.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.480624 seconds
    * Cumulative time: 0.480624 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        ['run']
        {'filter': 'b2', 'c': True, '_old': []}
        ['run']
        {'filter': 'b2', 'c': True, '_old': []}
        []
        {'c': True, 'aero': 'fin', '_old': [{'aero': 'body'}, {'aero': 'base'}]}
        ['run']
        {'filter': 'b2', 'c': True, '_old': []}
        ['run', 'b2']
        {'c': True, 'r': True, '_old': [], 'f': True}
        ['run']
        {'c': True, 'r': 'b2', '_old': [], 'f': True}
        

:STDERR:
    * **PASS**

Command 2: Python 3
--------------------

:Command:
    .. code-block:: console

        $ python3 test01_argread.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.59929 seconds
    * Cumulative time: 1.07991 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        ['run']
        {'c': True, 'filter': 'b2', '_old': []}
        ['run']
        {'c': True, 'filter': 'b2', '_old': []}
        []
        {'aero': 'fin', 'c': True, '_old': [{'aero': 'body'}, {'aero': 'base'}]}
        ['run']
        {'c': True, 'filter': 'b2', '_old': []}
        ['run', 'b2']
        {'c': True, 'f': True, 'r': True, '_old': []}
        ['run']
        {'c': True, 'r': 'b2', 'f': True, '_old': []}
        

:STDERR:
    * **PASS**

