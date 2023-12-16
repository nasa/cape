
--------------------------------------------------------------------
Test results: :mod:`test.902_pyfun.002_ellipsoid.test_001_aflr3mesh`
--------------------------------------------------------------------

This page documents from the folder

    ``test/902_pyfun/002_ellipsoid``

using the file

    ``test_001_aflr3mesh.py``

.. literalinclude:: _test_001_aflr3mesh.py
    :caption: test_001_aflr3mesh.py
    :language: python

Test case: :func:`test_01_aflr3`
--------------------------------
This test case runs the function:

.. literalinclude:: _test_001_aflr3mesh.py
    :caption: test_01_aflr3
    :language: python
    :pyobject: test_01_aflr3

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_01_aflr3():
            # Instantiate
            cntl = cape.pyfun.cntl.Cntl()
            # Run a case
    >       cntl.SubmitJobs(I="6")
    
    test/902_pyfun/002_ellipsoid/test_001_aflr3mesh.py:32: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1289: in SubmitJobs
        sts = self.CheckCaseStatus(i, jobs, u=kw.get('u'))
    cape/cntl.py:1928: in CheckCaseStatus
        current_jobid=self.CheckBatch()
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <cape.pyfun.cntl.Cntl(nCase=24)>
    
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

