
------------------------------------------------------------------------
Test results: :mod:`test.000_vendor.010_nmlfile.03_nml_io.test_nmlparts`
------------------------------------------------------------------------

This page documents from the folder

    ``test/000_vendor/010_nmlfile/03_nml_io``

using the file

    ``test_nmlparts.py``

.. literalinclude:: _test_nmlparts.py
    :caption: test_nmlparts.py
    :language: python

Test case: :func:`test_nml01`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml01
    :language: python
    :pyobject: test_nml01

PASS

Test case: :func:`test_nml02`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml02
    :language: python
    :pyobject: test_nml02

PASS

Test case: :func:`test_nml03`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml03
    :language: python
    :pyobject: test_nml03

PASS

Test case: :func:`test_nml04`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml04
    :language: python
    :pyobject: test_nml04

PASS

Test case: :func:`test_nml05`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml05
    :language: python
    :pyobject: test_nml05

PASS

Test case: :func:`test_nml06`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml06
    :language: python
    :pyobject: test_nml06

PASS

Test case: :func:`test_nml07`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml07
    :language: python
    :pyobject: test_nml07

PASS

Test case: :func:`test_nml08`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml08
    :language: python
    :pyobject: test_nml08

PASS

Test case: :func:`test_nml09`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml09
    :language: python
    :pyobject: test_nml09

PASS

Test case: :func:`test_nml10`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml10
    :language: python
    :pyobject: test_nml10

PASS

Test case: :func:`test_nml11`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml11
    :language: python
    :pyobject: test_nml11

PASS

Test case: :func:`test_nml12`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml12
    :language: python
    :pyobject: test_nml12

PASS

Test case: :func:`test_nml13`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml13
    :language: python
    :pyobject: test_nml13

PASS

Test case: :func:`test_nml14`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml14
    :language: python
    :pyobject: test_nml14

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, NML14)
        def test_nml14():
            # Read an escaped string
    >       nml = NmlFile(NML14)
    
    test/000_vendor/010_nmlfile/03_nml_io/test_nmlparts.py:176: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/nmlfile/__init__.py:90: in __init__
        self.read_nmlfile(a)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = {}, fname = 'invalid_str.nml'
    
        def read_nmlfile(self, fname: str):
            r"""Read a namelist file
        
            :Call:
                >>> nml.read_nmlfile(fname)
            :Inputs:
                *nml*: :class:`NmlFile`
                    Namelist index
                *fname*: :class:`str`
                    Name of file
            """
            # Save file name
            self.fname = fname
            # Open file
    >       with open(fname, 'r') as fp:
    E       FileNotFoundError: [Errno 2] No such file or directory: 'invalid_str.nml'
    
    cape/nmlfile/__init__.py:118: FileNotFoundError

Test case: :func:`test_nml15`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_nmlparts.py
    :caption: test_nml15
    :language: python
    :pyobject: test_nml15

PASS

