
------------------------------------------------------------------
Test results: :mod:`test.903_pyover.001_bullet.test_001_pyovercli`
------------------------------------------------------------------

This page documents from the folder

    ``test/903_pyover/001_bullet``

using the file

    ``test_001_pyovercli.py``

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_001_pyovercli.py
    :language: python

Test case: :func:`test_01_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_01_run
    :language: python
    :pyobject: test_01_run

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
        def test_01_run():
            # Instantiate
            cntl = cape.pyover.cntl.Cntl()
            # Run first case
    >       cntl.SubmitJobs(I="1")
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:29: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1286: in SubmitJobs
        sts = self.CheckCaseStatus(i, jobs, u=kw.get('u'))
    cape/cntl.py:1925: in CheckCaseStatus
        current_jobid=self.CheckBatch()
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <cape.pyover.cntl.Cntl(nCase=4)>
    
        def CheckBatch(self):
            r"""Check to see if we are running inside a batch job
        
            This looks for environment variables to see if this is running
            inside a batch job.  Currently supports slurm and PBS.
        
            :Call:
                >>> q = cntl.CheckBatch()
            :Inputs:
                *cntl*: :class:`cape.cntl.Cntl`
                    Overall CAPE control instance
            :Outputs:
                *jobid*: :class:`int`
                    ``0`` if no batch environment was detected
            :Versions:
                * 2023-12-13 ``@dvicker``: v1.0
            """
        
            # Assume this is not a batch job
            jobid=0
        
            if self.opts.get_slurm(0):
                jobid = int(os.environ.get('SLURM_JOB_ID', 0))
            else:
    >           pbsid = int(os.environ.get('PBS_JOBID', 0))
    E           ValueError: invalid literal for int() with base 10: '17816837.pbspl1.nas.nasa.gov'
    
    cape/cntl.py:2325: ValueError

Test case: :func:`test_02_c`
----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_02_c
    :language: python
    :pyobject: test_02_c

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_02_c():
            # Split command and add `-m` prefix
            cmdlist = [sys.executable, "-m", "cape.pyover", "-c", "-I", "1"]
            # Run the command
            stdout, _, _ = testutils.call_o(cmdlist)
            # Check outout
            result = testutils.compare_files(stdout, "test.02.out", ELLIPSIS=True)
    >       assert result.line1 == result.line2
    E       AssertionError: assert '1    powerof...           \n' == '1    powerof...0   .   ...\n'
    E         - 1    poweroff/m0.8a4.0b0.0 DONE    1500/1500   .   ...
    E         + 1    poweroff/m0.8a4.0b0.0 ---     /           .
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:41: AssertionError

Test case: :func:`test_03_fm`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_03_fm
    :language: python
    :pyobject: test_03_fm

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_03_fm():
            # Instantiate
            cntl = cape.pyover.cntl.Cntl()
            # Collect aero
            cntl.cli(fm=True, I="1")
            # Read databook
            cntl.ReadDataBook()
            # Get value
    >       CN = cntl.DataBook["bullet_no_base"]["CN"][0]
    E       IndexError: index 0 is out of bounds for axis 0 with size 0
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:54: IndexError

