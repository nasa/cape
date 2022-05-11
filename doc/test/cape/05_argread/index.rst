
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``05_argread``: **FAIL** (command 1)
===========================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

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

Command 1: Python 2 (**FAIL**)
-------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_argread.py

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
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_argread.py", line 8, in <module>
            import cape
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


