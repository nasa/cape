
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
            cntl.SubmitJobs(I="0")
            # Collect aero
            cntl.cli(fm=True, I="0")
            # Read databook
            cntl.ReadDataBook()
            # Get value
    >       CA = cntl.DataBook["bullet_no_base"]["CA"][0]
    E       IndexError: index 0 is out of bounds for axis 0 with size 0
    
    test/901_pycart/002_adapt/test_pycartadapt01.py:35: IndexError

