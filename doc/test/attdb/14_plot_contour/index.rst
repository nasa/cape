
.. This documentation written by TestDriver()
   on 2021-03-19 at 09:49 PDT

Test ``14_plot_contour``
==========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/attdb/14_plot_contour/``

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

Command 1: Contours from indices: Python 2
-------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_mask.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 7.68875 seconds
    * Cumulative time: 7.68875 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:PNG:
    * **FAIL**
    * Difference fraction: 0.2943
    * Actual:

        .. image:: PNG-00-00.png
            :width: 4.5in

    * Target:

        .. image:: PNG-target-00-00.png
            :width: 4.5in

