
.. _test-cape-cli:

Testing :mod:`cape` Command-Line Interface
===========================================

This section tests basic command-line interface usage for the CAPE package.
These examples use the ``cape`` executable.  While this is not necessary the
most likely use case since ``cape`` is not tied to any specific CFD solver, the
more specific executables share the same behavior.

This section is set up to test command-line calls, but in the test code below
such calls are hidden inside :func:`cape.test.check_output` Python calls.  For
example the system call

    .. code-block:: console
    
        $ cape -c --filter b2
        
becomes the Python call

    .. code-block:: python
    
        txt = cape.test.check_output("cape -c --filter b2")
        print(txt)
        
The primary reason for this procedure is to take advantage of the integration
into Sphinx testing.  This connection is only for Python code, and so a wrapper
of system calls into Python is needed.  In addition, performing the testing in
Python enables a more extensible setup environment, so the requirement to use
Python is not a particularly high burden.

.. testsetup:: *

    # System modules
    import os
    import shutil
    # Numerics
    import numpy as np
    
    # Modules to import
    import cape.test
    import cape
    
.. _test-cape-c:

Status List
------------
First list the cases and see if we are getting the expected results.

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Remove log file if necessary
    if os.path.isfile('test.out'): os.remove('test.out')
    
    # Run the ``-c`` command
    txt = cape.test.check_output("cape -c")
    # Print it
    print(txt)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Case Config/Run Directory  Status  Iterations  Que CPU Time 
    ---- --------------------- ------- ----------- --- --------
    0    poweroff/m0.5a0.0b0.0 ---     /           .            
    1    poweroff/m0.5a2.0b0.0 ---     /           .            
    2    poweroff/m0.5a0.0b2.0 ---     /           .            
    3    poweroff/m0.5a2.0b2.0 ---     /           .            
    4    poweroff/m0.8a0.0b0.0 ---     /           .            
    5    poweroff/m0.8a2.0b0.0 ---     /           .            
    6    poweroff/m0.8a0.0b2.0 ---     /           .            
    7    poweroff/m0.8a2.0b2.0 ---     /           .            
    8    poweroff/m1.1a0.0b0.0 ---     /           .            
    9    poweroff/m1.1a2.0b0.0 ---     /           .            
    10   poweroff/m1.1a0.0b2.0 ---     /           .            
    11   poweroff/m1.1a2.0b2.0 ---     /           .            
    12   poweroff/m1.5a0.0b0.0 ---     /           .            
    13   poweroff/m1.5a2.0b0.0 ---     /           .            
    14   poweroff/m1.5a0.0b2.0 ---     /           .            
    15   poweroff/m1.5a2.0b2.0 ---     /           .            
    16   poweroff/m2.5a0.0b0.0 ---     /           .            
    17   poweroff/m2.5a2.0b0.0 ---     /           .            
    18   poweroff/m2.5a0.0b2.0 ---     /           .            
    19   poweroff/m2.5a2.0b2.0 ---     /           .            
    
    ---=20, 
    
Now let's try each of the subsetting commands.

    * ``--filter``: Show any case containing specified text
    * ``--glob``: Show any case matching a file glob
    * ``--re``: Show any case containing a specified regular expression
    * ``--cons``: Show any case meeting a list of constraints
    
.. _test-cape-filter:

Filter
-------
Show the cases from above that contain the text ``b2``.  This will meetin any
cases with a sideslip angle between 2 degrees (inclusive) and 3 degrees
(exclusive).

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Remove log file if necessary
    if os.path.isfile('test.out'): os.remove('test.out')
    
    # Run the ``-c`` command
    txt = cape.test.check_output("cape -c --filter b2")
    # Print it
    print(txt)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Case Config/Run Directory  Status  Iterations  Que CPU Time 
    ---- --------------------- ------- ----------- --- --------
    2    poweroff/m0.5a0.0b2.0 ---     /           .            
    3    poweroff/m0.5a2.0b2.0 ---     /           .            
    6    poweroff/m0.8a0.0b2.0 ---     /           .            
    7    poweroff/m0.8a2.0b2.0 ---     /           .            
    10   poweroff/m1.1a0.0b2.0 ---     /           .            
    11   poweroff/m1.1a2.0b2.0 ---     /           .            
    14   poweroff/m1.5a0.0b2.0 ---     /           .            
    15   poweroff/m1.5a2.0b2.0 ---     /           .            
    18   poweroff/m2.5a0.0b2.0 ---     /           .            
    19   poweroff/m2.5a2.0b2.0 ---     /           .            
    
    ---=10, 
    
.. _test-cape-glob:

Glob
-----
The glob ``poweroff/m0*`` will match any subsonic ``poweroff`` case.

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Remove log file if necessary
    if os.path.isfile('test.out'): os.remove('test.out')
    
    # Run the ``-c`` command
    txt = cape.test.check_output("cape -c --glob 'poweroff/m0*'")
    # Print it
    print(txt)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Case Config/Run Directory  Status  Iterations  Que CPU Time 
    ---- --------------------- ------- ----------- --- --------
    0    poweroff/m0.5a0.0b0.0 ---     /           .            
    1    poweroff/m0.5a2.0b0.0 ---     /           .            
    2    poweroff/m0.5a0.0b2.0 ---     /           .            
    3    poweroff/m0.5a2.0b2.0 ---     /           .            
    4    poweroff/m0.8a0.0b0.0 ---     /           .            
    5    poweroff/m0.8a2.0b0.0 ---     /           .            
    6    poweroff/m0.8a0.0b2.0 ---     /           .            
    7    poweroff/m0.8a2.0b2.0 ---     /           .           
    
    ---=8, 

    
.. _test-cape-re:

Regular Expression
-------------------
A relatively complex regular expression test is used.  The regular expression
``m.\.5.*b2`` matches any case with a Mach number in [0.5,0.6), [1.5,1.6),
... [0.5,0.6) and the sideslip angle is in the range [2.0,3.0).

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Remove log file if necessary
    if os.path.isfile('test.out'): os.remove('test.out')
    
    # Run the ``-c`` command
    txt = cape.test.check_output("cape -c --re 'm.\.5.*b2'")
    # Print it
    print(txt)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Case Config/Run Directory  Status  Iterations  Que CPU Time 
    ---- --------------------- ------- ----------- --- --------
    2    poweroff/m0.5a0.0b2.0 ---     /           .            
    3    poweroff/m0.5a2.0b2.0 ---     /           .            
    14   poweroff/m1.5a0.0b2.0 ---     /           .            
    15   poweroff/m1.5a2.0b2.0 ---     /           .            
    18   poweroff/m2.5a0.0b2.0 ---     /           .            
    19   poweroff/m2.5a2.0b2.0 ---     /           .            
    
    ---=6, 


.. _test-cape-cons:

Constraints
------------
The ``--cons`` command is tested by issuing a command that is nearly equivalent
to the ``--re`` test preceding this.  It is different in the general case since
it will match any case with a Mach number exactly equal to an integer plus 0.5
and a sideslip that is exactly 2 degrees.  For the cases in this test setup,
this is the same as the less tightly constrained ``--re`` example.

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Remove log file if necessary
    if os.path.isfile('test.out'): os.remove('test.out')
    
    # Run the ``-c`` command
    txt = cape.test.check_output("cape -c --cons 'beta==2,Mach%1==0.5'")
    # Print it
    print(txt)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Case Config/Run Directory  Status  Iterations  Que CPU Time 
    ---- --------------------- ------- ----------- --- --------
    2    poweroff/m0.5a0.0b2.0 ---     /           .            
    3    poweroff/m0.5a2.0b2.0 ---     /           .            
    14   poweroff/m1.5a0.0b2.0 ---     /           .            
    15   poweroff/m1.5a2.0b2.0 ---     /           .            
    18   poweroff/m2.5a0.0b2.0 ---     /           .            
    19   poweroff/m2.5a2.0b2.0 ---     /           .            
    
    ---=6, 


.. _test-cape-index:

Index List
-----------
The ``-I`` command is the simplest subsetting command.  This command lists
cases 2, 3, and 4.

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Remove log file if necessary
    if os.path.isfile('test.out'): os.remove('test.out')
    
    # Run the ``-c`` command
    txt = cape.test.check_output("cape -c -I 2:5")
    # Print it
    print(txt)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Case Config/Run Directory  Status  Iterations  Que CPU Time 
    ---- --------------------- ------- ----------- --- --------
    2    poweroff/m0.5a0.0b2.0 ---     /           .            
    3    poweroff/m0.5a2.0b2.0 ---     /           .            
    4    poweroff/m0.8a0.0b0.0 ---     /           .            
    
    ---=3,


.. _test-cape-c-compound:

Compound Subsets
-------------------
This test combines the ``-I``, ``--re``, and ``--cons`` commands to ensure that
compounding subsetting commands works as expected.

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Remove log file if necessary
    if os.path.isfile('test.out'): os.remove('test.out')
    
    # Run the ``-c`` command
    txt = cape.test.check_output("cape -c -I 15: --cons Mach%1=0.5 --re b2")
    # Print it
    print(txt)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Case Config/Run Directory  Status  Iterations  Que CPU Time 
    ---- --------------------- ------- ----------- --- --------
    15   poweroff/m1.5a2.0b2.0 ---     /           .            
    18   poweroff/m2.5a0.0b2.0 ---     /           .            
    19   poweroff/m2.5a2.0b2.0 ---     /           .            
    
    ---=3, 

