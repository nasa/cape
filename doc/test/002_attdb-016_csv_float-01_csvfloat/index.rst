
------------------------------------------------------------------
Test results: :mod:`test.002_attdb.016_csv_float.test_01_csvfloat`
------------------------------------------------------------------

This page documents from the folder

    ``test/002_attdb/016_csv_float``

using the file

    ``test_01_csvfloat.py``

.. literalinclude:: _test_01_csvfloat.py
    :caption: test_01_csvfloat.py
    :language: python

Test case: :func:`test_01_csvfloat`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_01_csvfloat.py
    :caption: test_01_csvfloat
    :language: python
    :pyobject: test_01_csvfloat

PASS

Test case: :func:`test_02_csvdtype`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_01_csvfloat.py
    :caption: test_02_csvdtype
    :language: python
    :pyobject: test_02_csvdtype

PASS

Test case: :func:`test_03_csvsimple`
------------------------------------
This test case runs the function:

.. literalinclude:: _test_01_csvfloat.py
    :caption: test_03_csvsimple
    :language: python
    :pyobject: test_03_csvsimple

PASS

Test case: :func:`test_04_csv_c`
--------------------------------
This test case runs the function:

.. literalinclude:: _test_01_csvfloat.py
    :caption: test_04_csv_c
    :language: python
    :pyobject: test_04_csv_c

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_testdir(__file__)
        def test_04_csv_c():
            # Read CSV file
            db = csvfile.CSVFile()
            # Read in C
    >       db.c_read_csv(CSVFILE)
    
    test/002_attdb/016_csv_float/test_01_csvfloat.py:66: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/attdb/ftypes/csvfile.py:304: in c_read_csv
        self.c_read_csv_data(f)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <csvfile.CSVFile(cols=['mach', 'alpha', 'beta'])>
    f = <_io.TextIOWrapper name='aeroenv.csv' mode='r' encoding='UTF-8'>
    
        def c_read_csv_data(self, f):
            r"""Read data portion of CSV file using C extension
        
            :Call:
                >>> db.c_read_csv_data(f)
            :Inputs:
                *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                    CSV file interface
                *f*: :class:`file`
                    Open file handle
            :Effects:
                *db.cols*: :class:`list`\ [:class:`str`]
                    List of column names
            :Versions:
                * 2019-11-25 ``@ddalle``: Version 1.0
            """
            # Test module
            if _ftypes is None:
    >           raise ImportError("No _ftypes extension module")
    E           ImportError: No _ftypes extension module
    
    cape/attdb/ftypes/csvfile.py:620: ImportError

Test case: :func:`test_05_csv_py`
---------------------------------
This test case runs the function:

.. literalinclude:: _test_01_csvfloat.py
    :caption: test_05_csv_py
    :language: python
    :pyobject: test_05_csv_py

PASS

