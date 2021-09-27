
.. _doc-docstring:

Interactive Documentation Strings
=================================

In addition to this documentation and the examples demonstrated in
``$PYCART/examples/``, documentation for individual Python functions can be
accessed interactively within any Python shell. This is particularly helpful
for advanced users who would like to remember what are the inputs and outputs
to a function. Using standard Python nomenclature, the docstrings (short for
documentation strings) are generally informative.

    .. code-block:: pycon
    
        >>> import cape.convert
        >>> print(cape.convert.AlphaTPhi2AlphaBeta.__doc__)
        
        Convert total angle of attack and roll to alpha, beta
        
            :Call:
                >>> alpha, beta = cape.AlphaTPhi2AlphaBeta(alpha_t, beta)
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
                * 2014-06-02 ``@ddalle``: Version 1.0
            
These documentation strings always list the inputs and outputs of a function
and assist the user in properly using functions. The help messages are written
in the modified version of reStructuredText used in Sphinx documentation, which
contains extra information for the user. There is more discussion of this
syntax below.

An even more informative tool is the built-in Python function :func:`help`,
which can take as an argument a function, class, or module. For example

    .. code-block:: pycon
    
        >>> import cape.convert
        >>> help(cape.convert.AlphaTPhi2AlphaBeta)
        Help on function AlphaTPhi2AlphaBeta in module cape.convert:

        AlphaTPhi2AlphaBeta(alpha_t, phi)
            Convert total angle of attack and roll to alpha, beta
        
            :Call:
                >>> alpha, beta = cape.AlphaTPhi2AlphaBeta(alpha_t, beta)
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
                * 2014-06-02 ``@ddalle``: Version 1.0

Another highly recommended tool for advanced scripting is `IPython
<http://www.ipython.org>`_. This package allows fuller tab completion, which
reduces how many commands need to be memorized. Another way to see these help
messages can be seen below.

    .. code-block:: none
    
        In[1]: import cape.convert

        In[2]: ?cape.convert.AlphaTPhe2AlphaBeta
        Signature: cape.convert.AlphaTPhi2AlphaBeta(alpha_t, phi)
        Docstring:
        Convert total angle of attack and roll to alpha, beta
        
        :Call:
            >>> alpha, beta = cape.AlphaTPhi2AlphaBeta(alpha_t, beta)
        :Inputs:
            *alpha_t*: :class:`float` | :class:`numpy.array`
                Total angle of attack
            *phi*: :class:`float` | :class:`numpy.array`
                Total roll angle
        :Outputs:
            *alpha*: :class:`float` | :class:`numpy.array`
                Angle of attack
            *beta*: :class:`float` | :class:`numpy.array`
                Sideslip angle
        :Versions:
            * 2014-06-02 ``@ddalle``: Version 1.0

Of these three options, the :func:`help` is the most informative for exploring
totally unknown modules and classes. It will list, among other things, all of
the functions and/or classes within the module/class. So for example, the
:func:`help` on the module :mod:`cape.convert` looks like:

    .. code-block:: pycon

        >>> import cape.convert
        >>> help(cape.convert)
        Help on module cape.convert in cape:
        
        NAME
            cape.convert
        
        DESCRIPTION
            :mod:`cape.convert`: Unit and angle conversion utilities
            =========================================================
            
            Perform conversions such as (alpha total, phi) to (alpha, beta).  It
            also contains various utilities such as calculating Sutherland's law for
            estimating viscosity with the standard parameters, which are commonly
            needed tools for CFD solvers.
        
        FUNCTIONS
            AlphaBeta2AlphaMPhi(alpha, beta)
                Convert angle of attack and sideslip to maneuver axis
        ...

And it goes on much longer, using (at least on Mac and Linux systems) the
``more`` editor to provide a simple searchable interface to all of the
documentation for the module :mod:`cape.convert`.
