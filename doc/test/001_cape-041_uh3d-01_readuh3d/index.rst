
------------------------------------------------------------
Test results: :mod:`test.001_cape.041_uh3d.test_01_readuh3d`
------------------------------------------------------------

This page documents from the folder

    ``test/001_cape/041_uh3d``

using the file

    ``test_01_readuh3d.py``

.. literalinclude:: _test_01_readuh3d.py
    :caption: test_01_readuh3d.py
    :language: python

Test case: :func:`test_01_convertuh3d`
--------------------------------------
This test case runs the function:

.. literalinclude:: _test_01_readuh3d.py
    :caption: test_01_convertuh3d
    :language: python
    :pyobject: test_01_convertuh3d

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_01_convertuh3d():
            # Read source
            tri = trifile.Tri(uh3d=SOURCE, c=CONFIGJSONFILE)
            # Write XML file
            tri.WriteConfigXML(XMLFILE)
            # Map the AFLR3 boudnary conditions
            tri.MapBCs_ConfigAFLR3()
            # Write the AFLR3 boundary condition summary
            tri.config.WriteAFLR3BC(OUTPUT_PREFIX + ".bc")
            # Map the FUN3D boundary conditions
            tri.config.WriteFun3DMapBC(OUTPUT_PREFIX + ".mapbc")
            # Write surface
            print("Writing surface TRI file")
            # Number of triangles
            ntrik = (tri.nTri - 10) // 1000 + 1
            assert ntrik == 10
            # File name
            fname = OUTPUT_PREFIX + ("-tri%ik.lr4.tri" % ntrik)
            # Write it
            tri.WriteTri_lr4(fname)
            # Reread tri file
            tri0 = trifile.Tri(fname, c=XMLFILE)
            # Create PLTFile interface
            plt = pltfile.Plt(triq=tri0, c=XMLFILE)
            # File name
            fname = OUTPUT_PREFIX + ("-tri%ik.plt" % ntrik)
            # Write it
    >       plt.Write(fname)
    
    test/001_cape/041_uh3d/test_01_readuh3d.py:61: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <cape.plt.Plt object at 0x6ed5700>, fname = 'arrow-tri10k.plt'
    Vars = ['x', 'y', 'z'], kw = {}, nVar = 3, IZone = range(0, 7), nZone = 7
    IVar = array([0, 1, 2]), f = <_io.BufferedWriter name='arrow-tri10k.plt'>
    s = array(b'#!TDV112', dtype='|S8')
    
        def Write(self, fname, Vars=None, **kw):
            """Write a Fun3D boundary Tecplot binary file
        
            :Call:
                >>> plt.Write(fname, Vars=None, **kw)
            :Inputs:
                *plt*: :class:`pyFun.plt.Plt`
                    Tecplot PLT interface
                *fname*: :class:`str`
                    Name of file to read
                *Vars*: {``None``} | :class:`list` (:class:`str`)
                    List of variables (by default, use all variables)
                *CompID*: {``range(len(plt.nZone))``} | :class:`list`
                    Optional list of zone numbers to use
            :Versions:
                * 2017-03-29 ``@ddalle``: Version 1.0
                * 2017-05-16 ``@ddalle``: Version 1.1; variable list
                * 2017-12-18 ``@ddalle``: Version 1.2; *CompID* input
            """
            # Default variable list
            if Vars is None:
                Vars = self.Vars
            # Number of variables
            nVar = len(Vars)
            # Check for CompID list
            IZone = kw.get("CompID", range(self.nZone))
            # Number of output zones
            nZone = len(IZone)
            # Indices of variabels
            IVar = np.array([self.Vars.index(v) for v in Vars])
            # Open the file
            f = open(fname, 'wb')
            # Write the opening string
            s = np.array('#!TDV112', dtype='|S8')
            # Write it
            s.tofile(f)
            # Write specifier
            capeio.tofile_ne4_i(f, [1, 0])
            # Write title
    >       capeio.tofile_ne4_s(f, self.title)
    E       AttributeError: module 'cape.capeio' has no attribute 'tofile_ne4_s'
    
    cape/plt.py:460: AttributeError

Test case: :func:`test_02_compids`
----------------------------------
This test case runs the function:

.. literalinclude:: _test_01_readuh3d.py
    :caption: test_02_compids
    :language: python
    :pyobject: test_02_compids

PASS

Test case: :func:`test_03_readplt`
----------------------------------
This test case runs the function:

.. literalinclude:: _test_01_readuh3d.py
    :caption: test_03_readplt
    :language: python
    :pyobject: test_03_readplt

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_03_readplt():
            # Read PLT file
    >       plt = pltfile.Plt(PLTFILE)
    
    test/001_cape/041_uh3d/test_01_readuh3d.py:77: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/plt.py:169: in __init__
        self.Read(fname)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <cape.plt.Plt object at 0x79f3d70>, fname = 'arrow-tri10k.plt'
    
        def Read(self, fname):
            """Read a Fun3D boundary Tecplot binary file
        
            :Call:
                >>> plt.Read(fname)
            :Inputs:
                *plt*: :class:`pyFun.plt.Plt`
                    Tecplot PLT interface
                *fname*: :class:`str`
                    Name of file to read
            :Versions:
                * 2016-11-22 ``@ddalle``: Version 1.0
                * 2022-09-16 ``@ddalle``: Version 2.0; unstruc volume
            """
            # Open the file
            f = open(fname, 'rb')
            # Read the opening string
            s = np.fromfile(f, count=1, dtype='|S8')
            # Get string
            if len(s) == 0:
                # No string
                header = None
            else:
                # Get and convert
                header = s[0].decode("ascii")
            # Check it
            if header != '#!TDV112':
                f.close()
                raise ValueError("File '%s' must start with '#!TDV112'" % fname)
            # Throw away the next two integers
            self.line2 = np.fromfile(f, count=2, dtype='i4')
            # Read the title
            self.title = capeio.read_lb4_s(f)
            # Get number of variables (, unpacks the list)
    >       self.nVar, = np.fromfile(f, count=1, dtype='i4')
    E       ValueError: not enough values to unpack (expected 1, got 0)
    
    cape/plt.py:219: ValueError

