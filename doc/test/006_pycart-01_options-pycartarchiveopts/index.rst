
----------------------------------------------------------------------
Test results: :mod:`test.006_pycart.01_options.test_pycartarchiveopts`
----------------------------------------------------------------------

This page documents from the folder

    ``test/006_pycart/01_options``

using the file

    ``test_pycartarchiveopts.py``

.. literalinclude:: _test_pycartarchiveopts.py
    :caption: test_pycartarchiveopts.py
    :language: python

Test case: :func:`test_ArchiveOpts`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_pycartarchiveopts.py
    :caption: test_ArchiveOpts
    :language: python
    :pyobject: test_ArchiveOpts

FAIL

Failure contents:

.. code-block:: none

    def test_ArchiveOpts():
            # Full-folder archive
            opts1 = ArchiveOpts(OPTS1, Template="full")
            assert opts1.get_ArchivePreTarGroups() == VizGlob
            # Default
            opts2 = auto_Archive(OPTS1)
            assert opts2.get_ArchivePreTarGroups() == VizGlob
            opts1 = ArchiveOpts(OPTS1, ArchiveTemplate="nonsense")
            # None
            opts1 = ArchiveOpts(OPTS1, ArchiveTemplate="none")
    >       assert opts1 == dict(OPTS1, ArchiveTemplate="none")
    E       AssertionError: assert {'ArchiveForm...ateFiles': []} == {'ArchiveForm...es': '*.flow'}
    E         Omitting 3 identical items, use -vv to show
    E         Left contains 10 more items:
    E         {'PostDeletDirs': [],
    E          'PostDeleteDirs': [],
    E          'PostTarDirs': [],
    E          'PostTarGroups': [],
    E          'PostUpdateFiles': [],...
    E         
    E         ...Full output truncated (6 lines hidden), use '-vv' to show
    
    test/006_pycart/01_options/test_pycartarchiveopts.py:26: AssertionError

