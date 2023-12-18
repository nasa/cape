
--------------------------------------------------------------------
Test results: :mod:`test.007_pyfun.02_conditionals.test_conditional`
--------------------------------------------------------------------

This page documents from the folder

    ``test/007_pyfun/02_conditionals``

using the file

    ``test_conditional.py``

.. literalinclude:: _test_conditional.py
    :caption: test_conditional.py
    :language: python

Test case: :func:`test_conditionals01`
--------------------------------------
This test case runs the function:

.. literalinclude:: _test_conditional.py
    :caption: test_conditionals01
    :language: python
    :pyobject: test_conditionals01

PASS

Test case: :func:`test_conditionals2`
-------------------------------------
This test case runs the function:

.. literalinclude:: _test_conditional.py
    :caption: test_conditionals2
    :language: python
    :pyobject: test_conditionals2

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_conditionals2():
            # Instantiate
            cntl = cape.pyfun.cntl.Cntl()
            # Set up a case, but don't start
    >       cntl.SubmitJobs(I=0, start=False)
    
    test/007_pyfun/02_conditionals/test_conditional.py:44: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
<<<<<<< HEAD
    cape/cntl.py:1289: in SubmitJobs
        sts = self.CheckCaseStatus(i, jobs, u=kw.get('u'))
    cape/cntl.py:1928: in CheckCaseStatus
=======
    cape/cntl.py:1286: in SubmitJobs
        sts = self.CheckCaseStatus(i, jobs, u=kw.get('u'))
    cape/cntl.py:1925: in CheckCaseStatus
>>>>>>> 20b1c358a57cd52432a88c9786943b6a75785357
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
<<<<<<< HEAD
    E           ValueError: invalid literal for int() with base 10: '17817902.pbspl1.nas.nasa.gov'
    
    cape/cntl.py:2328: ValueError
=======
    E           ValueError: invalid literal for int() with base 10: '17816837.pbspl1.nas.nasa.gov'
    
    cape/cntl.py:2325: ValueError
>>>>>>> 20b1c358a57cd52432a88c9786943b6a75785357

