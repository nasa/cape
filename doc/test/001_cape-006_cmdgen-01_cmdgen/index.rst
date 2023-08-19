
------------------------------------------------------------
Test results: :mod:`test.001_cape.006_cmdgen.test_01_cmdgen`
------------------------------------------------------------

This page documents from the folder

    ``test/001_cape/006_cmdgen``

using the file

    ``test_01_cmdgen.py``

.. literalinclude:: _test_01_cmdgen.py
    :caption: test_01_cmdgen.py
    :language: python

Test case: :func:`test_01_aflr3`
--------------------------------
This test case runs the function:

.. literalinclude:: _test_01_cmdgen.py
    :caption: test_01_aflr3
    :language: python
    :pyobject: test_01_aflr3

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_testdir(__file__)
        def test_01_aflr3():
            # Get case runner
            runner = case.CaseRunner()
            # Read settings
            rc = runner.read_case_json()
            # Form command
            cmd1 = cmdgen.aflr3(rc)
            # Alternate form
            cmd2 = cmdgen.aflr3(
                i="pyfun.surf",
                o="pyfun.lb8.ugrid",
                blr=10,
                flags={"someflag": 2},
                keys={"somekey": 'c'})
            # Check
            assert cmd1[-1] == "somekey=c"
    >       assert cmd2 == [
                "aflr3",
                "-i", "pyfun.surf",
                "-o", "pyfun.lb8.ugrid",
                "-blr", "10",
                "-someflag", "2",
                "somekey=c"
            ]
    E       AssertionError: assert ['aflr3', '-i...-mdsblf', ...] == ['aflr3', '-i..., '-blr', ...]
    E         At index 5 diff: '-mdsblf' != '-blr'
    E         Left contains one more item: 'somekey=c'
    E         Use -v to get more diff
    
    test/001_cape/006_cmdgen/test_01_cmdgen.py:28: AssertionError

Test case: :func:`test_02_intersect`
------------------------------------
This test case runs the function:

.. literalinclude:: _test_01_cmdgen.py
    :caption: test_02_intersect
    :language: python
    :pyobject: test_02_intersect

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_testdir(__file__)
        def test_02_intersect():
            # Get case runner
            runner = case.CaseRunner()
            # Read settings
            rc = runner.read_case_json()
            # Form commands
            cmd1 = cmdgen.intersect(rc)
            cmd2 = cmdgen.intersect(T=True)
            # Check
    >       assert cmd1 == [
                "intersect",
                "-i", "Components.tri",
                "-o", "Components.i.tri",
                "-ascii", "-T"
            ]
    E       AssertionError: assert ['intersect',....i.tri', '-T'] == ['intersect',...'-ascii', ...]
    E         At index 5 diff: '-T' != '-ascii'
    E         Right contains one more item: '-T'
    E         Use -v to get more diff
    
    test/001_cape/006_cmdgen/test_01_cmdgen.py:49: AssertionError

