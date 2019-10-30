"""
:mod:`cape.tex`: Interface to LaTeX files
=============================================

This is a module built off of the :class:`cape.fileCntl.FileCntl` class
customized for manipulating Cape's automated reports.  In addition to containing
methods for reading, writing, and organizing lines of text, it contains a
:func:`Compile` method for compiling the PDF and sparing the user from
constructing the system command to do so.

See the :mod:`cape.cfdx.report` module or the
:ref:`JSON report page <cape-json-Report>` page for more information.

"""

# Import the base file control class.
from .fileCntl import FileCntl, _num, _float

# File interface
import os
import subprocess as sp

# Numerics
import numpy as np

# Base this class off of the main file control class.
class Tex(FileCntl):
    """
    File control class for LaTeX files
    
    :Call:
        >>> TX = cape.tex.Tex()
        >>> TX = cape.tex.Tex(fname='report.tex')
    :Inputs:
        *fname*: :class:`str`
            Name of LaTex report file to read/write
    :Outputs:
        *TX*: :class:`cape.tex.Tex` or derivative
            Instance of LaTeX report interface
    :Versions:
        * 2015-03-07 ``@ddalle``: Started
        * 2015-03-10 ``@ddalle``: First version
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname='report.tex'):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Save the location
        self.fdir = os.path.split(os.path.abspath(fname))[0]
        # Split into sections.
        self.SplitToSections(reg="%\$__([\w_]+)")
    
    # Display method
    def __repr__(self):
        """Display method for LaTeX class
        
        :Versions:
            * 2015-03-07 ``@ddalle``: First version
        """
        # Initialize the string.
        s = '<Tex("%s", %i lines' % (self.fname, len(self.lines))
        # Check for number of sections.
        if hasattr(self, 'SectionNames'):
            # Write the number of sections.
            s = s + ", %i sections)>" % len(self.SectionNames)
        else:
            # Just close the string.
            s = s + ")>"
        return s
        
    # Function to compile.
    def Compile(self, check=True):
        """Compile the LaTeX file
        
        :Call:
            >>> TX.Compile(check=True)
        :Inputs:
            *TX*: :class:`cape.tex.Tex` or derivative
                Instance of LaTeX report interface
            *check*: {``True``} | ``False``
                Whether or not to fail if there is a nonzero compile status
        :Versions:
            * 2015-03-07 ``@ddalle``: First version
            * 2017-03-07 ``@ddalle``: Added *check*
        """
        # Save current location.
        fpwd = os.getcwd()
        # Go to the file's location
        os.chdir(self.fdir)
        # Hide the output.
        f = open('/dev/null', 'w')
        # Compile.
        ierr = sp.call(['pdflatex', '-interaction=nonstopmode', self.fname],
            stdout=f)
        # Close the output "file".
        f.close()
        # Go back to original location
        os.chdir(fpwd)
        # Check compile status.
        if ierr and check:
            # Create an error
            raise SystemError("Compiling '%s' failed with status %i."
                % (self.fname, ierr))
        elif ierr:
            # Throw a warning
            print("Compiling '%s' failed with status %i."
                % (self.fname, ierr))
    
