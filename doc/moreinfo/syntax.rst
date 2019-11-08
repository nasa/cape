
.. _doc-syntax:

Documentation Notation and Formatting
======================================

In addition to text with references to variables, functions, file names, etc.,
this documentation contains many descriptions of function inputs and outputs.
This is a semi-standard Python way of describing the user interface to a
function.

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
            
This documentation uses much of the same syntax and markup throughout the rest
of the document, and this quick guide may become useful in cases of confusion
or ambiguity.
            

Documentation Syntax Guide
--------------------------
Understanding this syntax can be somewhat helpful for reading the documentation
strings, and it provides a useful shorthand when discussing features of the
code.  A table of how various syntaxes are used is below.

========================   ===================   ==============================
Syntax                     Formatted             Description/Purpose
========================   ===================   ==============================
\``f(a)``                  ``f(a)``              Raw text or source code
\``$PYCART``               ``$PYCART``           Environment variable
\*a\*                      *a*                   Variable name
\*a>b\*                    *a>b*                 Key *b* of *a*, ``a["b"]``
\:Inputs:                  :Inputs:              Dictionary-style header
\:file:\`pyCart.json`      :file:`pyCart.json`   File name
\:mod:\`cape.tri`          :mod:`cape.tri`       Module name
\:class:\`int`             :class:`int`          Class or type of variable
\:func:\`SetAlpha`         :func:`SetAlpha`      Function name
========================   ===================   ==============================
                
The class, mod, and func keys generate links to their formatted documentation
when Sphinx can find it.

Some inputs and outputs can have multiple allowed types, which is also
represented in the docstrings.  In addition, lists, dictionaries, and NumPy
arrays often have a type, which is shown in parentheses.  For example, ``[0.1,
0.2, 0.34]`` might be listed as :class:`list` (:class:`float`), and ``{"name":
"a", "place": "b"}`` might be listed as :class:`dict` (:class:`str`).  Finally,
default values are sometimes placed in curly braces.  An example portion of a
document string is shown below.

    .. code-block:: none
    
        :Inputs:
            *a*: :class:`list` (:class:`float`)
                Description of a list of float values
            *b*: :class:`int` | ``None``
                Description of variable that is either an integer or nothing
            *c*: :class:`dict` | :class:`list`
                Description of multiple ways to represent some data
            *q*: {True} | False
                Description of Boolean variable that defaults to ``True``
                
The formatted version of the above looks like this.

    :Inputs:
        *a*: :class:`list` (:class:`float`)
            Description of a list of float values
        *b*: :class:`int` | ``None``
            Description of variable that is either an integer or nothing
        *c*: :class:`dict` | :class:`list`
            Description of multiple ways to represent some data
        *q*: {True} | False
            Description of Boolean variable that defaults to ``True``


.. _kwargs:

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
        
