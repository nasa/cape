
.. This documentation written by TestDriver()
   on 2021-03-19 at 09:49 PDT

Test ``12_plot_png``
======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/attdb/12_plot_png/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_plot_png.py
        $ python3 test01_plot_png.py

Command 1: Line load plot with PNG: Python 2
---------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_plot_png.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 8.53635 seconds
    * Cumulative time: 8.53635 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:PNG:
    * **FAIL**
    * Difference fraction: 0.0197
    * Actual:

        .. image:: PNG-00-00.png
            :width: 4.5in

    * Target:

        .. image:: PNG-target-00-00.png
            :width: 4.5in

