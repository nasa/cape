
Usage and Getting More Information
==================================

While the documentation you are presently reading is the most complete source of
information about the CAPE package, there are several more sources of
information.  The examples in ``$CAPE/examples/`` serve as tutorials for the
package, and these are also discussed in their own sections of this
documentation.

Interactive Documentation Strings
---------------------------------

For advanced usage of CAPE modules, especially :mod:`pyCart`, the functions and
classes that are part of the package have their own documentation that can be
accessed directly from a Python shell.  For those who are familiar with Python
terminology, one way to rephrase this is to say that the docstrings are quite
helpful.

    .. code-block:: none
    
        >>> import pyCart
        >>> print(pyCart.AlphaTPhi2AlphaBeta.__doc__)
        
        Convert total angle of attack and total roll angle to angle of attack and
        sideslip angle.
        
        :Call:
            >>> alpha, beta = pyCart.AlphaTPhi2AlphaBeta(alpha_t, beta)
        :Inputs:
            *alpha_t*: :class:`float` or :class:`numpy.array`
                Total angle of attack
            *phi*: :class:`float` or :class:`numpy.array`
                Total roll angle
        :Outputs:
            *alpha*: :class:`float` or :class:`numpy.array`
                Angle of attack
            *beta*: :class:`float` or :class:`numpy.array`
                Sideslip angle
        :Versions:
            * 2014-06-02 ``@ddalle``: First version
            
These documentation strings always list the inputs and outputs of a function and
assist the user in properly using functions.  The help messages are written in
the modified version of reStructuredText used in Sphinx documentation, which
contains extra information for the user.  There is more discussion of this
syntax below.

Another highly recommended tool for advanced scripting is 
`IPython <http://www.ipython.org>`_.  This package allows tab completion, which
reduces how many commands need to be memorized.  Another way to see these help
messages can be seen below.

    .. code-block:: none
    
        In[1]: ?pyCart.AlphaTPhi2AlphaBeta
        
        Convert total angle of attack and total roll angle to angle of attack and
        sideslip angle.
        
        :Call:
            >>> alpha, beta = pyCart.AlphaTPhi2AlphaBeta(alpha_t, beta)
        :Inputs:
            *alpha_t*: :class:`float` or :class:`numpy.array`
                Total angle of attack
            *phi*: :class:`float` or :class:`numpy.array`
                Total roll angle
        :Outputs:
            *alpha*: :class:`float` or :class:`numpy.array`
                Angle of attack
            *beta*: :class:`float` or :class:`numpy.array`
                Sideslip angle
        :Versions:
            * 2014-06-02 ``@ddalle``: First version
        

This documentation is built by the `Sphinx <http://www.sphinx-doc.org>`_
documentation software, which has certain features that may not be obvious to
all users of pyCart and the other parts of the CAPE package.  The docstrings
like the one shown above are read by Sphinx and turned into formatted text such
as :func:`pyCart.inputCntl.InputCntl.SetAlpha`.  The basic layout of these
docstrings, with optional portions in parentheses, is as follows.

    .. code-block:: none
    
        Short description of the function or class
        
        (More detailed description if necessary)
        
        :Call:
            >>> Example Python call showing how to use the function
            (>>> Alternate examples if function can be used multiple ways)
        :Inputs:
            *Name_input1*: :class:`variable_type`
                Description of first input
            *Name_input2*: :class:`variable_type`
                Description of second input
        :Outputs:
            *Name_output1*: :class:`variable_type`
                Description of first output
            *Name_output2*: :class:`variable_type`
                Description of second output
        :Versions:
            * Date ``@Author``: Short description
            
Sphinx automatically turns this text into the following format.

        Short description of the function or class
        
        (More detailed description if necessary)
        
        :Call:
            >>> Example Python call showing how to use the function
            (>>> Alternate examples if function can be used multiple ways)
        :Inputs:
            *Name_input1*: :class:`variable_type`
                Description of first input
            *Name_input2*: :class:`variable_type`
                Description of second input
        :Outputs:
            *Name_output1*: :class:`variable_type`
                Description of first output
            *Name_output2*: :class:`variable_type`
                Description of second output
        :Versions:
            * Date ``@Author``: Short description
            

Documentation Syntax Guide
--------------------------
Understanding this syntax can be somewhat helpful for reading the documentation
strings, and it provides a useful shorthand when discussing features of the
code.  A table of how various syntaxes are used is below.

========================   ===================   ==============================
Syntax                     Format                Description
========================   ===================   ==============================
\``f(a)``                  ``f(a)``              Raw text or source code
\``$PYCART``               ``$PYCART``           Environment variable
\*a\*                      *a*                   Variable name
\:Inputs:                  :Inputs:              Dictionary-style header
\:file:\`pyCart.json`      :file:`pyCart.json`   File name
\:mod:\`pyCart.tri`        :mod:`pyCart.tri`     Module name
\:class:\`int`             :class:`int`          Class or type of variable
\:func:\`SetAlpha`         :func:`SetAlpha`      Function name
========================   ===================   ==============================
                
The class, mod, and func keys generate links to their formatted documentation
when Sphinx can find it.

.. _kwarks:

Keyword Arguments
-----------------
One aspect of possible confusion to new or novice Python users is the so-called
keyword arguments.  For example, in the following example command, there are
regular arguments and keyword arguments.

    .. code-block:: python
    
        comp = 'CA'
        ylbl = 'CA (Axial force coefficient)'
        FM.PlotCoeff(comp, YLabel=ylbl)
        
In this case *comp* is a regular input, often called an "argument" in Python
jargon.  Then *YLabel* is a keyword input or keyword argument, which is
specified with an equal sign in the function call.  The advantage of keyword
arguments is that they can be given in any order, and many of them can be
skipped when default values should be used.  For example, the following two
commands are identical.

    .. code-block:: python
    
        FM.PlotCoeff('CA', XLabel='i', YLabel='CA')
        FM.PlotCoeff('CA', YLabel='CA', XLabel='i')
        
Another tool for using keywords is that a :class:`dict` can be used as a list of
keywords inputs.  The following two commands are identical.

    .. code-block:: python
    
        kw = {"d": 0.1, "k": 3.0, "YLabel": "Cx"}
        FM.PlotCoeff('CA', **kw)
        FM.PlotCoeff('CA', d=0.1, k=3.0, YLabel="Cx")
        