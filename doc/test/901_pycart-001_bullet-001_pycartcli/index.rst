
------------------------------------------------------------------
Test results: :mod:`test.901_pycart.001_bullet.test_001_pycartcli`
------------------------------------------------------------------

This page documents from the folder

    ``test/901_pycart/001_bullet``

using the file

    ``test_001_pycartcli.py``

.. literalinclude:: _test_001_pycartcli.py
    :caption: test_001_pycartcli.py
    :language: python

Test case: :func:`test_01_c`
----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pycartcli.py
    :caption: test_01_c
    :language: python
    :pyobject: test_01_c

PASS

Test case: :func:`test_02_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_001_pycartcli.py
    :caption: test_02_run
    :language: python
    :pyobject: test_02_run

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_02_run():
            # Instantiate
            cntl = cape.pycart.cntl.Cntl()
            # Run first case
    >       cntl.SubmitJobs(I="0")
    
    test/901_pycart/001_bullet/test_001_pycartcli.py:40: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1308: in SubmitJobs
        self.PrepareCase(i)
    cape/pycart/cntl.py:443: in PrepareCase
        self.PrepareMesh(i)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <cape.pycart.cntl.Cntl(nCase=4)>, i = 0
    
        def PrepareMesh(self, i):
            """Prepare the mesh for case *i* if necessary.
        
            :Call:
                >>> cntl.PrepareMesh(i)
            :Inputs:
                *cntl*: :class:`cape.pycart.cntl.Cntl`
                    Instance of control class containing relevant parameters
                *i*: :class:`int`
                    Index of the case to check (0-based)
            :Versions:
                * 2014-09-29 ``@ddalle``: First version
            """
            # ---------
            # Case info
            # ---------
            # Get the case name.
            frun = self.x.GetFullFolderNames(i)
            # Get name of group.
            fgrp = self.x.GetGroupFolderNames(i)
            # Check the mesh.
            if self.CheckMesh(i):
                return None
            # ------------------
            # Folder preparation
            # ------------------
            # Remember current location.
            fpwd = os.getcwd()
            # Go to root folder.
            os.chdir(self.RootDir)
            # Check for the group folder and make it if necessary.
            if not os.path.isdir(fgrp):
                self.mkdir(fgrp)
            # Check for groups with common meshes
            if self.opts.get_GroupMesh():
                # Get the group index.
                j = self.x.GetGroupIndex(i)
                # Status update
                print("  Group name: '%s' (index %i)" % (fgrp, j))
                # Go there.
                os.chdir(fgrp)
            else:
                # Check if the run folder exists.
                if not os.path.isdir(frun):
                    self.mkdir(frun)
                # Status update.
                print("  Case name: '%s' (index %i)" % (frun, i))
                # Go there.
                os.chdir(frun)
            # ----------
            # Copy files
            # ----------
            # Get the name of the configuration file.
            fxml = os.path.join(self.RootDir, self.opts.get_ConfigFile())
            fpre = os.path.join(self.RootDir, self.opts.get_preSpecCntl())
            fc3d = os.path.join(self.RootDir, self.opts.get_inputC3d())
            # Copy the config file.
            if os.path.isfile(fxml):
                shutil.copyfile(fxml, 'Config.xml')
            # Copy the preSpec file.
            if os.path.isfile(fpre):
                shutil.copyfile(fpre, 'preSpec.c3d.cntl')
            # Copy the cubes input file.
            if os.path.isfile(fc3d):
                shutil.copyfile(fc3d, 'input.c3d')
            # ------------------
            # Triangulation prep
            # ------------------
            # Status update
            print("  Preparing surface triangulation...")
            # Read the mesh.
            self.ReadTri()
            # Revert to initial surface.
            self.tri = self.tri0.Copy()
            # Apply rotations, translations, etc.
            self.PrepareTri(i)
            # Check intersection status.
            if self.opts.get_intersect():
                # Write the tri file as non-intersected; each volume is one CompID
                self.tri.WriteVolTri('Components.tri')
                # Write the existing triangulation with existing CompIDs.
                self.tri.Write('Components.c.tri')
            else:
                # Write the tri file.
                self.tri.Write('Components.i.tri')
            # --------------------
            # Volume mesh creation
            # --------------------
            # Get functions for mesh functions.
            keys = self.x.GetKeysByType('MeshFunction')
            # Loop through the mesh functions
            for key in keys:
                # Get the function for this *MeshFunction*
                func = self.x.defns[key]['Function']
                # Form args and kwargs
                a = (self, self.x[key][i])
                kw = dict(i=i)
                # Apply it
                self.exec_modfunction(func, a, kw, name="RunMatrixMeshFunction")
            # RunControl options (for consistency)
            rc = self.opts['RunControl']
            # Run autoInputs if necessary.
            if self.opts.get_PreMesh(0) or not os.path.isfile('preSpec.c3d.cntl'):
                # Run autoInputs (tests opts.get_autoInputs() internally)
                case.CaseAutoInputs(rc, j=0)
            # Read the resulting preSpec.c3d.cntl file
            self.PreSpecCntl = PreSpecCntl('preSpec.c3d.cntl')
            # Bounding box control...
            self.PreparePreSpecCntl()
            # Check for jumpstart.
            if self.opts.get_PreMesh(0) or self.opts.get_GroupMesh():
                # Run ``intersect`` if appropriate
                CaseIntersect(rc)
                # Run ``verify`` if appropriate
    >           case.CaseVerify(rc)
    E           AttributeError: module 'cape.pycart.case' has no attribute 'CaseVerify'
    
    cape/pycart/cntl.py:632: AttributeError

