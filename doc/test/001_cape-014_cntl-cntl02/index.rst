
-------------------------------------------------------
Test results: :mod:`test.001_cape.014_cntl.test_cntl02`
-------------------------------------------------------

This page documents from the folder

    ``test/001_cape/014_cntl``

using the file

    ``test_cntl02.py``

.. literalinclude:: _test_cntl02.py
    :caption: test_cntl02.py
    :language: python

Test case: :func:`test_cntl01_multilevelcase`
---------------------------------------------
This test case runs the function:

.. literalinclude:: _test_cntl02.py
    :caption: test_cntl01_multilevelcase
    :language: python
    :pyobject: test_cntl01_multilevelcase

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_cntl01_multilevelcase():
            # Instatiate
            cntl = cape.cntl.Cntl()
            # Pick a case
            i = 14
            # Get name of case
            frun = cntl.x.GetFullFolderNames(i)
            # Make sure it's got a lot of / in it
            assert frun.count('/') == 3
            # Create a folder
            cntl.make_case_folder(i)
            # Make sure expected folder exists
            # (Tests making arbitrary-depth case folders)
            assert os.path.isdir(frun)
            # Get a report
            rep = cntl.opts.get_ReportList()[0]
            report = cntl.ReadReport(rep)
            # Update a case to check arbitrary-depth during report
    >       report.UpdateReport(I=i)
    
    test/001_cape/014_cntl/test_cntl02.py:40: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cfdx/report.py:560: in UpdateReport
        self.UpdateCases(I)
    cape/cfdx/report.py:677: in UpdateCases
        self.UpdateCase(i)
    cape/cfdx/report.py:915: in UpdateCase
        sts = self.cntl.CheckCaseStatus(i)
    cape/cntl.py:1925: in CheckCaseStatus
        current_jobid=self.CheckBatch()
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <cape.cntl.Cntl(nCase=20)>
    
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
    E           ValueError: invalid literal for int() with base 10: '17837858.pbspl1.nas.nasa.gov'
    
    cape/cntl.py:2325: ValueError

