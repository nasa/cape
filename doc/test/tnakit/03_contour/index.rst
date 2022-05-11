
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``03_contour``: **FAIL** (command 1)
===========================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/tnakit/03_contour/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_contourplot.py
        $ python3 test01_contourplot.py

Command 1: Contour plot: Python 2 (**FAIL**)
---------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_contourplot.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.26 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_contourplot.py", line 11, in <module>
            import cape.attdb.dbfm as dbfm
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


:PNG:
    * **FAIL**
    * Actual:

        .. image:: PNG-00-00.png
            :width: 4.5in

    * Target:

        .. image:: PNG-target-00-00.png
            :width: 4.5in

