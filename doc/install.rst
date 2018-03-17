
Requirements and Installation
=============================

Cape does not have strict system requirements. The idea is that any hardware
that can run Cart3D (or FUN3D or OVERFLOW) can also run this software. It is
only supported on Linux, though this may be extended to other operating
environments with minor changes.

Software Dependencies
---------------------
The following software is required to execute any of the pyCart functions.

    * Python version 2.6-2.7
    * `NumPy <http://www.numpy.org>`_ version 1.4.1 or newer
    
Support for Python 3.5+ is planned for near-future versions and is partially
functional already.  The software was written with some of the changes (such as
all :func:`print` statements using functions), but full support has not been
tested.
    
To test Python on your system, launch a Python shell by typing ``python`` into
a shell. A more accurate test would be to use the command ``/usr/bin/env
python``, which may point to a different version of Python.

    .. code-block:: none
    
        $ /usr/bin/env python
        Python 2.7.6 (default, Jun 22 2015, 17:58:13) 
        [GCC 4.8.2] on linux2
        Type "help", "copyright", "credits" or "license" for more information.
        >>>
        
Note that the shortcut ``Ctrl-D`` can be used to exit the Python shell.  The
command will report the version right after starting.  Once inside the shell,
use the following command to test NumPy support.

    .. code-block:: python
    
        >>> import numpy as np
        >>> np.__version__
        '1.8.2'
        
If NumPy is not present on your system, the result will look something like the
following.

    .. code-block:: python
    
        >>> import numpy as np
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        ImportError: No module named numpy
        
If you are using an enterprise system or a supercomputer, try loading an
environment module to see if NumPy is available but not in the default setup.
On NASA's *Pleiades*, the following system command is necessary and sufficient.

    .. code-block:: none
    
        $ module load python
        
If this fixes the problem, the module needs to be loaded each time Cape is
used. You can put it into your startup script (e.g. :file:`$HOME/.bashrc` or
:file:`$HOME/.cshrc`) or just remember to load it.

If NumPy is still not on your system, it can be found `here
<http://www.scipy.org/scipylib/download.html>`_, although it is present or
available on most standard Linux builds. A typical name for this package in
popular distributions such as Ubuntu is ``python-numpy``.

Optional Software Dependencies
------------------------------

There are several more pieces of software that are required for some of the
advanced features of Cape.  The following are required for generating various
plots in PDFs that can be used to track trends or focus in on the results of
individual cases.

    * Some version of PDFLaTeX (type ``pdflatex`` into a shell to test)
    * `Matplotlib <http://matplotlib.org/>`_ version 0.99 or newer (1.3+
      recommended)
    
There are various distributions of PDFLaTeX; the most common Linux distribution
is ``texlive``, which can be installed via your system's version of ``sudo
apt-get install texlive-full``.

To test Matplotlib, open a Python shell and try the following.

    .. code-block:: python
        
        >>> import matplotlib
        >>> matplotlib.__version__
        '1.3.1'
        
Installing Matplotlib is often tied together with NumPy, but on popular Linux
distributions, it can usually be found as a package ``python-matplotlib`` or
similar.

For visualizing flow solutions in these automated reports, |tecplot| is
required, although ParaView support is in development.

.. |tecplot| unicode:: Tecplot 0xAE

Finally, the `IPython <http://ipython.org/>`_ interactive shell is recommended
for advanced users and users of the API.

Optional Compiling
------------------
All features of Cape have at least a Python implementation, but some more
intensive functions are also written in C. To activate the faster versions of
these features (writing ``tri`` files is a key example), you will need a C
compiler and the NumPy libraries. In many distributions, the NumPy libraries
are already added to the path for any system with NumPy installed, but this is
not always true. The file :file:`$CAPE/config.cfg` contains settings that can
be edited if the compiler needs to be told where to find the NumPy libraries.

Installation is simple if the dependencies are present. In the ``$CAPE``
folder, run the command ``make``. If compilation is successful, it will create
the files ``$CAPE/pyCart/_pycart.so`` and ``$CAPE/cape/_cape.so``.
Otherwise, the compiler probably needs some help finding the file
``numpy/arrayobject.h``. For example, on *Pleiades*, I use the following
:file:`config.cfg` file.

    .. code-block:: cfg
    
        [python]
        exec = python2
        version = 2.7
        
        [compiler]
        cc = gcc
        extra_cflags = -Wall -Wno-unused-function -Wno-unused-variable
            -Wno-unused-but-set-variable -Wno-parentheses -Wno-format
            -Werror-implicit-function-declaration 
            -g -O2 -fPIC -fno-stack-protector
        extra_ldflags = 
        extra_include_dirs = /nasa/python/2.7.3/lib/python2.7/site-packages/numpy/core/include/

These settings differ slightly from those in the :file:`config.cfg` file that
is distributed with Cape, which is set up to work with a typical Red Hat
Enterprise Linux 7 build.

Setup and Typical Usage
-----------------------
The software is distributed as a tar archive, for example
:file:`cape_v0.8.tar.gz`.  Installation is a matter of untarring this archive in
your desired location, changing two environment variables, and optionally
compiling the C versions of some features as described in the previous section.

The following commands give an example of the first step.

    .. code-block:: bash
    
        $ tar -xzf cape_v0.8.tar.gz
        
We are using ``$CAPE`` as a variable to store the location of the folder that
gets created.  If this is unclear, run the following two commands in a BASH
environment.

    .. code-block:: bash
    
        $ CAPE=$PWD/pycart0.8
        $ echo $CAPE
        
Or, in a csh environment, the following will work.

    .. code-block:: csh
    
        $ setenv CAPE $PWD/pycart0.8
        $ echo $CAPE

The second part of the installation procedure is to edit two environment
variables.  The first is to add ``$CAPE/bin``, which contains the
executables, to the *PATH* variable.  Second, we add the Python source
directories to *PYTHONPATH* so that the pyCart/pyFun modules can be loaded by
any Python script.  The recommended way to do this is to use 
`environment modules <http://modules.sourceforge.net/>`_, but adding the
commands to the startup ``.bashrc`` or ``.cshrc`` file is also acceptable.

Using Startup Scripts
^^^^^^^^^^^^^^^^^^^^^
The following commands prepare your environment for using CAPE in a BASH
system.

    .. code-block:: bash
    
        export PATH=$PATH:$CAPE/scriptlib
        export PYTHONPATH=$CAPE:$PYTHONPATH
        
The following is appropriate for ``csh``.

    .. code-block:: csh
    
        setenv PATH $PATH:$CAPE/scriptlib
        setenv PYTHONPATH $CAPE:$PYTHONPATH
        
Add the appropriate set of commands to the appropriate ``.bashrc`` or
``.cshrc`` file, and pyCart, pyFun, and pyOver will be available for use in any
new shell.

Using Environment Modules
^^^^^^^^^^^^^^^^^^^^^^^^^
Environment modules are preferred because they reduce interference with other
software packages and because ``rc`` files are not automatically sourced in PBS
scripts (i.e. when running in a supercomputing environment).

A template module file is provided in :file:`$CAPE/modulefiles/pycart`. This
file contains a line

    .. code-block:: csh
    
        set  CAPE   $HOME/pycart0.8
        
Just edit this line so that it points to the appropriate location (i.e.
wherever you untarred the original file), and the module is ready for use. Then
to load the module, use

    .. code-block:: bash
    
        $ module load $CAPE/modulefiles/cape
        
If the module file is in one of the folders listed in *MODULEPATH*, this can be
shortened to just ``module load cape``.  Rather than explain this fully,
consider the following example that shows how to do this on Pleiades.

    .. code-block:: bash
    
        $ mkdir -p ~/share
        $ mkdir -p ~/share/modulefiles
        $ cp $CAPE/modulefiles/pycart ~/share/modulefiles
        $ module use -a ~/share/modulefiles
        $ module load pycart
        
It's a good idea to add the second command to ``.bashrc`` and/or ``.cshrc``.
Then the ``$HOME/share/modulefiles`` folder can be used as a home for local
environment modules, including this one.

