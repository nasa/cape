
.. This documentation written by TestDriver()
   on 2022-05-13 at 15:17 PDT

Test ``12_plot_png``: PASS
============================

This test PASSED on 2022-05-13 at 15:17 PDT

This test is run in the folder:

    ``test/attdb/12_plot_png/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_plot_png.py
        $ python3 test01_plot_png.py

Command 1: Line load plot with PNG: Python 2 (PASS)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_plot_png.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 1.03 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:PNG:
    * **PASS**
    * Difference fraction: 0.0197
    * Target:

        .. image:: PNG-target-00-00.png
            :width: 4.5in

Command 2: Line load plot with PNG: Python 3 (PASS)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_plot_png.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 1.12 seconds
    * Cumulative time: 2.15 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:PNG:
    * **PASS**
    * Difference fraction: 0.0200
    * Target:

        .. image:: PNG-target-01-00.png
            :width: 4.5in

