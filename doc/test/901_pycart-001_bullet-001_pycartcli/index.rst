
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
            cntl.SubmitJobs(I="0")
            # Collect aero
    >       cntl.cli(fm=True, I="0")
    
    test/901_pycart/001_bullet/test_001_pycartcli.py:42: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    cape/pycart/cntl.py:178: in cli
        cmd = self.cli_cape(*a, **kw)
    cape/cntl.py:804: in cli_cape
        self.UpdateFM(**kw)
    cape/cntl.py:99: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/cntl.py:4073: in UpdateFM
        self.DataBook.UpdateDataBook(I, comp=comp)
    cape/cfdx/dataBook.py:733: in UpdateDataBook
        n += self.UpdateCaseComp(i, comp)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    self = <[AttributeError("'NoneType' object has no attribute 'n'") raised in repr()] DataBook object at 0x51fc7d0>, i = 0
    comp = 'bullet_no_base'
    
        def UpdateCaseComp(self, i, comp):
            r"""Update or add a case to a data book
        
            The history of a run directory is processed if either one of
            three criteria are met.
        
                1. The case is not already in the data book
                2. The most recent iteration is greater than the data book
                   value
                3. The number of iterations used to create statistics has
                   changed
        
            :Call:
                >>> n = DB.UpdateCaseComp(i, comp)
            :Inputs:
                *DB*: :class:`pyFun.dataBook.DataBook`
                    Instance of the data book class
                *i*: :class:`int`
                    RunMatrix index
                *comp*: :class:`str`
                    Name of component
            :Outputs:
                *n*: ``0`` | ``1``
                    How many updates were made
            :Versions:
                * 2014-12-22 ``@ddalle``: Version 1.0
                * 2017-04-12 ``@ddalle``: Modified to work one component
                * 2017-04-23 ``@ddalle``: Added output
            """
            # Read if necessary
            if comp not in self:
                self.ReadDBComp(comp)
            # Check if it's present
            if comp not in self:
                raise KeyError("No aero data book component '%s'" % comp)
            # Get the first data book component.
            DBc = self[comp]
            # Try to find a match existing in the data book.
            j = DBc.FindMatch(i)
            # Get the name of the folder.
            frun = self.x.GetFullFolderNames(i)
            # Status update.
            print(frun)
            # Go home.
            os.chdir(self.RootDir)
            # Check if the folder exists.
            if not os.path.isdir(frun):
                # Nothing to do.
                return 0
            # Go to the folder.
            os.chdir(frun)
            # Get the current iteration number.
            nIter = self.GetCurrentIter()
            # Get the number of iterations used for stats.
            nStats = self.opts.get_DataBookNStats()
            # Get the iteration at which statistics can begin.
    >       nMin = self.opts.get_nMin()
    E       AttributeError: 'Options' object has no attribute 'get_nMin'
    
    cape/cfdx/dataBook.py:902: AttributeError

