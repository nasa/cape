
.. _test-cape-subset:

Testing :mod:`cape` API case subset commands
=============================================

This section tests basic API interface usage for the CAPE package to test the
various ways in which a user may specific a subset of a run matrix. These
examples use the :mod:`cape.cntl` module and calls API functions from the run
matrix.

This test is related to the :ref:`CAPE command-line interface <test-cape-cli>`
test suite but uses the API instead of the command line.

.. testsetup:: *

    # System modules
    import os
    import shutil
    # Numerics
    import numpy as np
    
    # Modules to import
    import cape.test
    import cape
    
.. _test-cape-DisplayStatus:

Status List
------------
First list the cases and see if we are getting the expected results.

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Read the interface
    cntl = cape.Cntl()
    # Run the ``-c`` command
    cntl.DisplayStatus()
    
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
    
.. _test-cntl-filter:

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
    
    # Read the interface
    cntl = cape.Cntl()
    # Run the API --filter command
    I = cntl.x.FilterString("b2")
    # Print it
    print(I)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    [ 2  3  6  7 10 11 14 15 18 19]
    
.. _test-cntl-glob:

Glob
-----
The glob ``poweroff/m0*`` will match any subsonic ``poweroff`` case.

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Read the interface
    cntl = cape.Cntl()
    # Run the API --glob command
    I = cntl.x.FilterWildcard("poweroff/m0*")
    # Print it
    print(I)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    [0 1 2 3 4 5 6 7]

    
.. _test-cntl-re:

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
    
    # Read the interface
    cntl = cape.Cntl()
    # Run the API --re command
    I = cntl.x.FilterRegex("m.\.5.*b2")
    # Print it
    print(I)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    [ 2  3  14 15 18 19]


.. _test-cntl-cons:

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
    
    # Read the interface
    cntl = cape.Cntl()
    # Run the API --cons command
    I = cntl.x.Filter(["beta==2", "Mach%1==0.5"])
    # Print it
    print(I)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    [ 2  3 14 15 18 19]


.. _test-cntl-index:

Index List
-----------
The ``-I`` command is the simplest subsetting command.  This command lists
cases 2, 3, and 4.

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Read the interface
    cntl = cape.Cntl()
    # Run the ``-I`` command
    cntl.DisplayStatus(I=[2,3,4])
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Case Config/Run Directory  Status  Iterations  Que CPU Time 
    ---- --------------------- ------- ----------- --- --------
    2    poweroff/m0.5a0.0b2.0 ---     /           .            
    3    poweroff/m0.5a2.0b2.0 ---     /           .            
    4    poweroff/m0.8a0.0b0.0 ---     /           .            
    
    ---=3,


.. _test-cntl-compound:

Compound Subsets
-------------------
This test combines the ``-I``, ``--re``, and ``--cons`` commands to ensure that
compounding subsetting commands works as expected.

.. testcode::

    # Bullet test folder
    fcape = os.path.join(cape.test.ftcape, "cape")
    # Go to this folder.
    os.chdir(fcape)
    
    # Read the interface
    cntl = cape.Cntl()
    # Run a compound API command
    I = cntl.x.GetIndices(I=np.arange(15,20), cons=["Mach%1==0.5"], re="b2")
    # Print it
    print(I)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    [15 18 19]

