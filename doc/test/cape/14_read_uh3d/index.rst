
.. This documentation written by TestDriver()
   on 2022-01-15 at 01:40 PST

Test ``14_read_uh3d``: PASS
=============================

This test PASSED on 2022-01-15 at 01:40 PST

This test is run in the folder:

    ``test/cape/14_read_uh3d/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_read_uh3d.py
        $ python3 test01_read_uh3d.py
        $ python2 test02_tri_ids.py
        $ python3 test02_tri_ids.py
        $ python2 test03_read_plt.py
        $ python3 test03_read_plt.py

**Included file:** ``test01_read_uh3d.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import sys library
        import sys
        
        # Import cape module
        import cape.tri
        import cape.plt
        
        # Config file
        CONFIGJSONFILE = "arrow.json"
        # Prefix
        OUTPUT_PREFIX = "arrow"
        # XML file
        XMLFILE = OUTPUT_PREFIX + ".xml"
        # Source uh3d
        SOURCE = "arrow.uh3d"
        # Output tri file
        TRIFILEOUT = "arrow.tri"
        
        
        # Read uh3d triangulation
        print("Reading source triangulation")
        tri = cape.tri.Tri(uh3d=SOURCE, c=CONFIGJSONFILE)
        
        # Write XML file
        print("Writing ConfigXML file")
        print("  %s" % XMLFILE)
        tri.WriteConfigXML(XMLFILE)
        
        # Map the AFLR3 boudnary conditions
        print("Mapping AFLR3 boundary conditions")
        tri.MapBCs_ConfigAFLR3()
        
        # Write the AFLR3 boundary condition summary
        print("Writing AFLR3 boundary conditions summary")
        tri.config.WriteAFLR3BC(OUTPUT_PREFIX + ".bc")
        
        # Map the FUN3D boundary conditions
        print("Mapping FUN3D boundary conditions")
        tri.config.WriteFun3DMapBC(OUTPUT_PREFIX + ".mapbc")
        
        # Write surface
        print("Writing surface TRI file")
        # Number of triangles
        ntrik = (tri.nTri - 10) // 1000 + 1
        # File name
        fname = OUTPUT_PREFIX + ("-tri%ik.lr4.tri" % ntrik)
        # Status update
        print(" Writing %s" % fname)
        # Write it
        tri.WriteTri_lr4(fname)
        
        print("Writing combined surface PLT file")
        print("  Creating PLT interface")
        # Reread tri file
        tri0 = cape.tri.Tri(fname, c=XMLFILE)
        # Create PLTFile interface
        plt = cape.plt.Plt(triq=tri0, c=XMLFILE)
        # Number of triangles
        ntrik = (tri.nTri - 10) // 1000 + 1
        # File name
        fname = OUTPUT_PREFIX + ("-tri%ik.plt" % ntrik)
        # Status update
        print("  Writing %s" % fname)
        # Write it
        plt.Write(fname)
        

**Included file:** ``test02_tri_ids.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import third-party libraries
        import numpy as np
        
        # Import cape module
        import cape.tri
        import cape.config
        
        # Output tri file
        TRIFILE = "arrow-tri10k.lr4.tri"
        
        # JSON file
        JSONCONFIGFILE = "arrow.json"
        
        # Read triangulation output from test01
        tri = cape.tri.Tri(fname=TRIFILE)
        
        # Check unique CompIDs
        tricids = np.unique(tri.CompID)
        print("CompIDs from TRI file")
        # Print unique CompIDs from tri
        print(tricids)
        
        # Read JSON configuation
        cfgj = cape.config.ConfigJSON(fname="arrow.json")
        # Check unique CompIDs
        jcids = np.unique(cfgj.IDs)
        print("CompIDs from JSON file")
        # Print unique CompIDs from JSON
        print(jcids)

**Included file:** ``test03_read_plt.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.plt
        
        # Output tri file
        PLTFILE = "arrow-tri10k.plt"
        
        # Read triangulation output from test01
        plt = cape.plt.Plt(fname=PLTFILE)

Command 1: Read UH3D: Python 2 (PASS)
--------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_read_uh3d.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.43 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Reading source triangulation
        Writing ConfigXML file
          arrow.xml
          No parent for component 'bullet_total'
        Mapping AFLR3 boundary conditions
        Writing AFLR3 boundary conditions summary
        Mapping FUN3D boundary conditions
        Writing surface TRI file
         Writing arrow-tri10k.lr4.tri
        Writing combined surface PLT file
          Creating PLT interface
          Writing arrow-tri10k.plt
        

:STDERR:
    * **PASS**

Command 2: Read UH3D: Python 3 (PASS)
--------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_read_uh3d.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.85 seconds
    * Cumulative time: 1.28 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Reading source triangulation
        Writing ConfigXML file
          arrow.xml
          No parent for component 'bullet_total'
        Mapping AFLR3 boundary conditions
        Writing AFLR3 boundary conditions summary
        Mapping FUN3D boundary conditions
        Writing surface TRI file
         Writing arrow-tri10k.lr4.tri
        Writing combined surface PLT file
          Creating PLT interface
          Writing arrow-tri10k.plt
        

:STDERR:
    * **PASS**

Command 3: Check TRI CompIDs: Python 2 (PASS)
----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_tri_ids.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.53 seconds
    * Cumulative time: 1.81 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        CompIDs from TRI file
        [ 1  2  3 11 12 13 14]
        CompIDs from JSON file
        [ 1  2  3 11 12 13 14]
        

:STDERR:
    * **PASS**

Command 4: Check TRI CompIDs: Python 3 (PASS)
----------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_tri_ids.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.73 seconds
    * Cumulative time: 2.55 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        CompIDs from TRI file
        [ 1  2  3 11 12 13 14]
        CompIDs from JSON file
        [ 1  2  3 11 12 13 14]
        

:STDERR:
    * **PASS**

Command 5: Read PLT: Python 2 (PASS)
-------------------------------------

:Command:
    .. code-block:: console

        $ python2 test03_read_plt.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.46 seconds
    * Cumulative time: 3.00 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 6: Read PLT: Python 3 (PASS)
-------------------------------------

:Command:
    .. code-block:: console

        $ python3 test03_read_plt.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.62 seconds
    * Cumulative time: 3.62 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

