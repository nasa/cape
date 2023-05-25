
-------------------------------------------------------------
Test results: :mod:`test.005_cfdx.01_options.test_reportopts`
-------------------------------------------------------------

This page documents from the folder

    ``test/005_cfdx/01_options``

using the file

    ``test_reportopts.py``

.. literalinclude:: _test_reportopts.py
    :caption: test_reportopts.py
    :language: python

Test case: :func:`test_reportopts1`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_reportopts.py
    :caption: test_reportopts1
    :language: python
    :pyobject: test_reportopts1

PASS

Test case: :func:`test_sweepopts1`
----------------------------------
This test case runs the function:

.. literalinclude:: _test_reportopts.py
    :caption: test_sweepopts1
    :language: python
    :pyobject: test_sweepopts1

FAIL

Failure contents:

.. code-block:: none

    def test_sweepopts1():
            # Parse options
            opts = reportopts.ReportOpts(OPTS2)
            # Check it
    >       assert opts.get_SweepOpt("sweep1", "MinCases") == 5
    E       AssertionError: assert None == 5
    E        +  where None = <bound method SweepCollectionOpts.get_SweepOpt of {'Sweeps': {}, 'Figures': {}, 'Subfigures': {}}>('sweep1', 'MinCases')
    E        +    where <bound method SweepCollectionOpts.get_SweepOpt of {'Sweeps': {}, 'Figures': {}, 'Subfigures': {}}> = {'Sweeps': {}, 'Figures': {}, 'Subfigures': {}}.get_SweepOpt
    
    test/005_cfdx/01_options/test_reportopts.py:91: AssertionError

Test case: :func:`test_reportfigopts1`
--------------------------------------
This test case runs the function:

.. literalinclude:: _test_reportopts.py
    :caption: test_reportfigopts1
    :language: python
    :pyobject: test_reportfigopts1

PASS

Test case: :func:`test_reportfigopts2`
--------------------------------------
This test case runs the function:

.. literalinclude:: _test_reportopts.py
    :caption: test_reportfigopts2
    :language: python
    :pyobject: test_reportfigopts2

FAIL

Failure contents:

.. code-block:: none

    def test_reportfigopts2():
            # Initialize report with figures
            opts = reportopts.ReportOpts({"Figures": FIGOPTS1})
            # Test list of figures
    >       assert opts.get_FigList() == ["fig1", "fig2"]
    E       AssertionError: assert [] == ['fig1', 'fig2']
    E         Right contains 2 more items, first extra item: 'fig1'
    E         Use -v to get more diff
    
    test/005_cfdx/01_options/test_reportopts.py:109: AssertionError

Test case: :func:`test_subfigopts1`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_reportopts.py
    :caption: test_subfigopts1
    :language: python
    :pyobject: test_subfigopts1

FAIL

Failure contents:

.. code-block:: none

    def test_subfigopts1():
            # Initialize subfigure options
            opts = reportopts.ReportOpts({"Subfigures": SUBFIGOPTS1})
            # Test list of subfigures
    >       assert opts.get_SubfigList() == ["STACK", "STACK_CA"]
    E       AssertionError: assert [] == ['STACK', 'STACK_CA']
    E         Right contains 2 more items, first extra item: 'STACK'
    E         Use -v to get more diff
    
    test/005_cfdx/01_options/test_reportopts.py:116: AssertionError

