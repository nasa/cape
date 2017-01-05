
Getting More Information
=========================

This section provides guidance on gaining information outside of the present
documentation and the example problems in ``$PYCART/examples/``.  This access
to additional information will mostly be useful to advanced users who want to
get interactive documentation for API functions.

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
        
