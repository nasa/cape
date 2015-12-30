
Usage and Getting More Information
==================================

While the documentation you are presently reading is the most complete source of
information about the CAPE package, there are several more sources of
information.  The examples in ``$CAPE/examples/`` serve as tutorials for the
package, and these are also discussed in their own sections of this
documentation.

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

This documentation is built by the `Sphinx <http://www.sphinx-doc.org>`_
documentation software, which has certain features that may not be obvious to
all users of pyCart and the other parts of the CAPE package.  