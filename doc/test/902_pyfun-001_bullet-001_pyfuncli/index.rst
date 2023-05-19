
----------------------------------------------------------------
Test results: :mod:`test.902_pyfun.001_bullet.test_001_pyfuncli`
----------------------------------------------------------------

This page documents from the folder

    ``test/902_pyfun/001_bullet``

using the file

    ``test_001_pyfuncli.py``

.. literalinclude:: _test_001_pyfuncli.py
    :caption: test_001_pyfuncli.py
    :language: python

Test case: :func:`test_01_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyfuncli.py
    :caption: test_01_run
    :language: python
    :pyobject: test_01_run

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_01_run():
            # Instantiate
            cntl = cape.pyfun.cntl.Cntl()
            # Run first case
    >       cntl.SubmitJobs(I="8")
    
    test/902_pyfun/001_bullet/test_001_pyfuncli.py:29: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    cape/cntl.py:1168: in SubmitJobs
        self.PrepareCase(i)
    cape/pyfun/cntl.py:1435: in PrepareCase
        self.PrepareNamelist(i)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    self = <pyFun.Cntl(nCase=24)>, i = 8
    
        def PrepareNamelist(self, i):
            r"""
            Write :file:`fun3d.nml` for run case *i* in the appropriate
            folder and with the appropriate settings.
        
            :Call:
                >>> cntl.PrepareNamelist(i)
            :Inputs:
                *cntl*: :class:`cape.pyfun.cntl.Cntl`
                    Instance of FUN3D control class
                *i*: :class:`int`
                    Run index
            :Versions:
                * 2014-06-04 ``@ddalle``: Version 1.0
                * 2014-06-06 ``@ddalle``: Low-level functionality for grid
                                          folders
                * 2014-09-30 ``@ddalle``: Changed to write only a single
                                          case
                * 2018-04-19 ``@ddalle``: Moved flight conditions to new
                                          function
            """
            # Ensure case index is set
            self.opts.setx_i(i)
            # Read namelist file
            self.ReadNamelist()
            # Go safely to root folder.
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Set the flight conditions
            self.PrepareNamelistFlightConditions(i)
        
            # Get the case.
            frun = self.x.GetFullFolderNames(i)
            # Set up the component force & moment tracking
            self.PrepareNamelistConfig()
            # Set up boundary list
            self.PrepareNamelistBoundaryList()
            # Prepare Adiabatic walls
            self.PrepareNamelistAdiabaticWalls()
        
            # Set the surface BCs
            for k in self.x.GetKeysByType('SurfBC'):
                # Check option for auto flow initialization
                if self.x.defns[k].get("AutoFlowInit", True):
                    # Ensure the presence of the triangulation
                    self.ReadTri()
                # Apply the appropriate methods
                self.SetSurfBC(k, i)
            # Set the surface BCs that use thrust as input
            for k in self.x.GetKeysByType('SurfCT'):
                # Check option for auto flow initialization
                if self.x.defns[k].get("AutoFlowInit", True):
                    # Ensure the presence of the triangulation
                    self.ReadTri()
                # Apply the appropriate methods
                self.SetSurfBC(k, i, CT=True)
            # File name
            if self.opts.get_Dual():
                # Write in the 'Flow/' folder
                fout = os.path.join(
                    frun, 'Flow',
                    '%s.mapbc' % self.GetProjectRootName(0))
            else:
                # Main folder
                fout = os.path.join(frun, '%s.mapbc' % self.GetProjectRootName(0))
        
            # Prepare internal boundary conditions
            self.PrepareNamelistBoundaryConditions()
            # Write the BC file
            self.MapBC.Write(fout)
        
            # Make folder if necessary.
            if not os.path.isdir(frun):
                self.mkdir(frun)
            # Apply any namelist functions
            self.NamelistFunction(i)
            # Loop through input sequence
            for j in range(self.opts.get_nSeq()):
                # Set the "restart_read" property appropriately
                # This setting is overridden by *nopts* if appropriate
                if j == 0:
                    # First run sequence; not restart
                    self.Namelist.SetVar('code_run_control', 'restart_read', 'off')
                else:
                    # Later sequence; restart
                    self.Namelist.SetVar('code_run_control', 'restart_read', 'on')
                # Get the reduced namelist for sequence *j*
                nopts = self.opts.select_namelist(j)
                dopts = self.opts.select_dual_namelist(j)
    >           mopts = self.opts.select_moving_body_input(j)
    E           AttributeError: 'Options' object has no attribute 'select_moving_body_input'
    
    cape/pyfun/cntl.py:1548: AttributeError

