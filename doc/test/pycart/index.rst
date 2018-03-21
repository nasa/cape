
.. _test-pycart:

-----------------------
Testing :mod:`pyCart`
-----------------------

.. testsetup:: *

    # System modules
    import os
    import shutil
    # Numerics
    import numpy as np
    
    # Modules to import
    import cape.test
    import pyCart
    import pyCart.tri
    
Sample test: read a triangulation file from

.. testcode::

    # Path to bullet triangulate
    fbullet = os.path.join(cape.test.ftpycart, "bullet")
    # Paths to triangulation and Config files
    ftri = os.path.join(fbullet, "bullet.tri")
    fxml = os.path.join(fbullet, "Config.xml")
    
    # Read the triangulation
    tri = pyCart.tri.Tri(ftri, c=fxml)
    # See it
    print(tri)
    
    # Get the bounding box for component 'cap'
    BBox = tri.GetCompBBox('cap')
    
    # Calculate deltas
    dx = np.abs(BBox - np.array([0.0, 1, -1,1, -1,1]))
    # Tolerances
    print(np.any(dx > 0.02))
    
    
    
.. testoutput::

    <cape.tri.Tri(nNode=2447, nTri=4890)>
    False
    
Test a full run and databook example.

.. testcode::

    # Bullet test folder
    fbullet = os.path.join(cape.test.ftpycart, "bullet")
    # Go to this folder.
    os.chdir(fbullet)
    
    # Clean up if necessary.
    if os.path.isdir('poweroff'): shutil.rmtree('poweroff')
    if os.path.isdir('data'):     shutil.rmtree('data')
    # Remove log file if necessary
    if os.path.isfile('test.out'): os.remove('test.out')
    
    # Show status
    cape.test.shell('pycart -c')
    
    # Run one case
    cape.test.shell('pycart -I 0')
    
    # Assemble the data book
    cape.test.shell('pycart -I 0 --aero')
   
    # Read the interface
    cart3d = pyCart.Cart3d()
    # Show it
    print(cart3d)
    # Read the databook
    cart3d.ReadDataBook()
    # Get the value.
    CA = cart3d.DataBook['bullet_no_base']['CA'][0]
    # Test it
    print(abs(CA - 0.745) <= 0.02)
    
.. testoutput::

    <pyCart.Cart3d(nCase=4, tri='bullet.tri')>
    True

    