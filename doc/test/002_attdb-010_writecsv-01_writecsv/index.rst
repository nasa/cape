
-----------------------------------------------------------------
Test results: :mod:`test.002_attdb.010_writecsv.test_01_writecsv`
-----------------------------------------------------------------

This page documents from the folder

    ``test/002_attdb/010_writecsv``

using the file

    ``test_01_writecsv.py``

.. literalinclude:: _test_01_writecsv.py
    :caption: test_01_writecsv.py
    :language: python

Test case: :func:`test_01_csv_dense`
------------------------------------
This test case runs the function:

.. literalinclude:: _test_01_writecsv.py
    :caption: test_01_csv_dense
    :language: python
    :pyobject: test_01_csv_dense

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, MAT_FILE)
        def test_01_csv_dense():
            # Read DataKit from MAT file
            db = rdb.DataKit(MAT_FILE, DefaultWriteFormat="%.3f")
            # Write simple dense CSV file
    >       db.write_csv_dense(CSV_FILE1)
    
    test/002_attdb/010_writecsv/test_01_writecsv.py:28: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/attdb/rdb.py:1110: in write_csv_dense
        dbcsv.write_csv_dense(fname, cols=cols)
    cape/attdb/ftypes/csvfile.py:974: in write_csv_dense
        self._write_csv_dense(f, cols=cols)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <csvfile.CSVFile(cols=['alpha', 'beta', 'CN'])>
    fp = <_io.TextIOWrapper name='CN-dense.csv' mode='w' encoding='UTF-8'>
    cols = ['alpha', 'beta', 'CN']
    
        def _write_csv_dense(self, fp, cols=None):
            r"""Write dense CSV file using *WriteFlag* for each column
        
            :Call:
                >>> db._write_csv_dense(f, cols=None)
            :Inputs:
                *db*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                    CSV file interface
                *fp*: :class:`file`
                    File open for writing
                *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                    List of columns to write
            :Versions:
                * 2019-12-05 ``@ddalle``: Version 1.0
            """
            # Default column list
            if cols is None:
                cols = self.cols
            # Initialize dictionary of processed values
            vals = {}
            parsedcols = []
            # Initialize write flags
            wflags = []
            # Parse columns and values
            for col in cols:
                # Get value
                vj = self[col]
                # Check if array
                if isinstance(vj, np.ndarray):
                    # Normal case; fine
                    pass
                elif isinstance(vj, list):
                    # Convert to array
                    vj = np.array(vj)
                else:
                    # Skip
                    print(
                        ("  [write_csv_dense] Skipping col '%s' " % col) +
                        ("with type '%s'" % type(vj).__name__))
                # Get dimensions
                ndj = vj.ndim
                # Check dimension
                if ndj == 1:
                    # Save length
                    if len(parsedcols) == 0:
                        # Save length of first array
                        nx = vj.size
                    elif nx != vj.size:
                        # Skip mismatching array
                        print(
                            ("  [write_csv_dense] Skipping '%s' " % col) +
                            ("with size %i; expected %i" % (vj.size, nx)))
                    # Normal case; 1D array
                    parsedcols.append(col)
                    vals[col] = vj
                elif ndj > 2:
                    print(
                        ("  [write_csv_dense] Skipping %i-dimensional " % ndj) +
                        ("col '%s'" % col))
                    continue
                # Get dimensions
    >           nxj, nyj = vj.shape
    E           ValueError: not enough values to unpack (expected 2, got 1)
    
    cape/attdb/ftypes/csvfile.py:1041: ValueError

Test case: :func:`test_02_csv_write`
------------------------------------
This test case runs the function:

.. literalinclude:: _test_01_writecsv.py
    :caption: test_02_csv_write
    :language: python
    :pyobject: test_02_csv_write

PASS

