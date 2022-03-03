
.. _install:

Requirements and Installation
=============================

CAPE tries not to have its own system requirements. The idea is that any
hardware that can run Cart3D (or FUN3D or OVERFLOW) can also run this software.
The *full* package is only supported on Linux, though support for portions of
the capabilities can be used on any machine that supports CPython 2.7 or 3.6+.

Installation is simple and done directly from a Python "wheel" file:

:Python 3.6+:

    (recommended)

    .. code-block:: console

        $ python3 -m pip install cape-1.0-cp38-cp38-linux_x86_64.whl

:Python 2.7:

    .. code-block:: console

        $ python2.7 -m pip install cape-1.0-cp27-cp27mu-linux_x86_64.whl

This package (including its C Python extensions) is supports either Python 2.7
or 3.6+ using the same source code. Some of the newer capabilities of Python 3
are therefore absent from the code base. Support for Python 2.7 will be dropped
with the next release.

    .. note::

        On a system on which you the user do not have elevated privileges, you
        may need to insert ``--user`` between ``install`` and ``cape``.

    .. note::

        This requires installing ``pip``, which is not universally installed
        with the default CPython installation. Usually **one** of the following
        commands will install it as long as Python itself is already installed
        on its system.

        .. code-block:: console

            $ easy_install --user pip
            $ easy_install-2.7 --user pip
            $ easy_install-3.6 --user pip

    .. note::

        The ``pip install`` command can still be executed on a system that
        doesn't directly match the specifications of the ``.whl`` file, but the
        C accelerator extension will not be installed in such a case.

    .. note::

        On Windows systems, the recommended ``pip install`` command is slightly
        different:

        .. code-block:: console

            $ py -m pip install cape-1.0-none-any.whl


Software Dependencies
---------------------
The following software is required to execute any of the pyCart functions.

    * Python version 2.7 or 3.6+
    * `NumPy <http://www.numpy.org>`_ version 1.4.1 or newer
    * `Matplotlib <https://matplotlib.org/>`_ version 2.0 or newer

Installation using the above ``pip`` commands will install or update these
packages if necessary.
        
Optional Software Dependencies
------------------------------

There are several more pieces of software that are required for some of the
advanced features of Cape.

    * Some version of PDFLaTeX (type ``pdflatex`` into a shell to test)

        .. note::

            There are various distributions of PDFLaTeX; the most common Linux
            distribution is ``texlive``, which is usually installed on high-
            performance computing systems.

    * `SciPy <https://pypi.org/project/scipy/>`_
    * `Vendorize <https://pypi.org/project/vendorize/>`_

For visualizing flow solutions in these automated reports, |tecplot| is
required, although ParaView support is also possible.

.. |tecplot| unicode:: Tecplot 0xAE

Finally, the `IPython <http://ipython.org/>`_ interactive shell is recommended
for advanced users and users of the API.

