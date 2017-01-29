
.. _doc-docstring:

Interactive Documentation Strings
=================================

In addition to this documentation and the examples demonstrated in
``$PYCART/examples/``, documentation for individual Python functions can be
accessed interactively within any Python shell.  This is particularly helpful
for advanced users who would like to remember what are the inputs and outputs
to a function.  Using standard Python nomenclature, the docstrings (short for
documentation strings) are generally informative.

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
        
