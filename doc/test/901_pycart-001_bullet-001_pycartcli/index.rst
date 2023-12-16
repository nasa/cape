
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

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_01_c():
            # Split command and add `-m` prefix
            cmdlist = [sys.executable, "-m", "cape.pycart", "-c"]
            # Run the command
            stdout, _, _ = testutils.call_o(cmdlist)
            # Check outout
            result = testutils.compare_files(stdout, "test.01.out")
    >       assert result.line1 == result.line2
    E       AssertionError: assert '' == '0    powerof...           \n'
    E         - 0    poweroff/m1.5a0.0b0.0 ---     /           .
    
    test/901_pycart/001_bullet/test_001_pycartcli.py:31: AssertionError

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
    cape/cntl.py:1289: in SubmitJobs
        sts = self.CheckCaseStatus(i, jobs, u=kw.get('u'))
    cape/cntl.py:1928: in CheckCaseStatus
        current_jobid=self.CheckBatch()
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <cape.pycart.cntl.Cntl(nCase=4)>
    
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
    E           ValueError: invalid literal for int() with base 10: '17817902.pbspl1.nas.nasa.gov'
    
    cape/cntl.py:2328: ValueError

