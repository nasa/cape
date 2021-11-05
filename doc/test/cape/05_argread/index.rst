
.. This documentation written by TestDriver()
   on 2021-11-02 at 01:40 PDT

Test ``05_argread``: PASS
===========================

This test PASSED on 2021-11-02 at 01:40 PDT

This test is run in the folder:

    ``test/cape/05_argread/``

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
        
        # Standard library
        import sys
        
        # Import cape module
        import cape
        
        
        # Print dict in sorted order
        def printdict(kw):
            sys.stdout.write("{")
            for j, k in enumerate(sorted(list(kw.keys()))):
                v = kw[k]
                if j > 0:
                    sys.stdout.write(", ")
                sys.stdout.write("'%s': %s" % (k, repr(v)))
            sys.stdout.write("}\n")
            sys.stdout.flush()
        
        # Basic Keys
        # Test 1
        # List of arguments
        argv = ['cape', 'run', '-c', '--filter', 'b2']
        
        # Process it
        a, kw = cape.argread.readkeys(argv)
        # Print results
        print(a)
        printdict(kw)
        
        # Test 2
        # List of arguments
        argv = ['cape', 'run', '-c', '-filter', 'b2']
        
        # Process it
        a, kw = cape.argread.readkeys(argv)
        # Print results
        print(a)
        printdict(kw)
        
        # Test 3
        # List of arguments
        argv = ['cape', '--aero', 'body', '--aero', 'base', '--aero', 'fin', '-c']
        
        # Process it
        a, kw = cape.argread.readkeys(argv)
        # Print results
        print(a)
        printdict(kw)
        
        # Read as Flags
        # Test 4
        # List of arguments
        argv = ['cape', 'run', '-c', '--filter', 'b2']
        
        # Process it
        a, kw = cape.argread.readflags(argv)
        # Print results
        print(a)
        printdict(kw)
        
        # Test 5
        # List of arguments
        argv = ['cape', 'run', '-c', '-fr', 'b2']
        
        # Process it
        a, kw = cape.argread.readflags(argv)
        # Print results
        print(a)
        printdict(kw)
        
        # Read flags like tar command
        # Test 6
        # List of arguments
        argv = ['cape', 'run', '-c', '-fr', 'b2']
        
        # Process it
        a, kw = cape.argread.readflagstar(argv)
        # Print results
        print(a)
        printdict(kw)

Command 1: Python 2 (PASS)
---------------------------

:Command:
    .. code-block:: console

        $ python2 test01_argread.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.50 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        ['run']
        {'_old': [], 'c': True, 'filter': 'b2'}
        ['run']
        {'_old': [], 'c': True, 'filter': 'b2'}
        []
        {'_old': [{'aero': 'body'}, {'aero': 'base'}], 'aero': 'fin', 'c': True}
        ['run']
        {'_old': [], 'c': True, 'filter': 'b2'}
        ['run', 'b2']
        {'_old': [], 'c': True, 'f': True, 'r': True}
        ['run']
        {'_old': [], 'c': True, 'f': True, 'r': 'b2'}
        

:STDERR:
    * **PASS**

Command 2: Python 3 (PASS)
---------------------------

:Command:
    .. code-block:: console

        $ python3 test01_argread.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.55 seconds
    * Cumulative time: 1.04 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        ['run']
        {'_old': [], 'c': True, 'filter': 'b2'}
        ['run']
        {'_old': [], 'c': True, 'filter': 'b2'}
        []
        {'_old': [{'aero': 'body'}, {'aero': 'base'}], 'aero': 'fin', 'c': True}
        ['run']
        {'_old': [], 'c': True, 'filter': 'b2'}
        ['run', 'b2']
        {'_old': [], 'c': True, 'f': True, 'r': True}
        ['run']
        {'_old': [], 'c': True, 'f': True, 'r': 'b2'}
        

:STDERR:
    * **PASS**

