
-----------------------------------------------------------------
Test results: :mod:`test.901_pycart.002_adapt.test_pycartadapt01`
-----------------------------------------------------------------

This page documents from the folder

    ``test/901_pycart/002_adapt``

using the file

    ``test_pycartadapt01.py``

.. literalinclude:: _test_pycartadapt01.py
    :caption: test_pycartadapt01.py
    :language: python

Test case: :func:`test_02_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_pycartadapt01.py
    :caption: test_02_run
    :language: python
    :pyobject: test_02_run

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_02_run():
            # Instantiate
            cntl = cape.pycart.cntl.Cntl()
            # Run first case
    >       cntl.SubmitJobs(I="0")
    
    test/901_pycart/002_adapt/test_pycartadapt01.py:29: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1286: in SubmitJobs
        sts = self.CheckCaseStatus(i, jobs, u=kw.get('u'))
    cape/cntl.py:1925: in CheckCaseStatus
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
    E           ValueError: invalid literal for int() with base 10: '17816837.pbspl1.nas.nasa.gov'
    
    cape/cntl.py:2325: ValueError

