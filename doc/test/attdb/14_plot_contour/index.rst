
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``14_plot_contour``: **FAIL** (command 1)
================================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/attdb/14_plot_contour/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_mask.py
        $ python3 test01_mask.py
        $ python2 test02_levels.py
        $ python3 test02_levels.py
        $ python2 test03_response.py
        $ python3 test03_response.py

Command 1: Contours from indices: Python 2 (**FAIL**)
------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_mask.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.20 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_mask.py", line 11, in <module>
            import cape.attdb.rdb as rdb
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

