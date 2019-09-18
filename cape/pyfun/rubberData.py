#!/usr/bin/env python
"""
:mod:`pyFun.rubberData`: FUN3D optimization/adaptation file ``rubber.data``
============================================================================

This module provides an interface to the fixed-name FUN3D file ``rubber.data``
that is used in FUN3D to provide inputs for optimization and or adjoint-based
mesh adaptation.  It provides a single class
:class:`pyFun.rubberData.RubberData` that interprets and sets individual lines
of this customized data file.

:See also:
    * :mod:`cape.fileCntl`
    * :mod:`pyFun.fun3d`
    * :func:`pyFun.fun3d.Fun3d.ReadRubberData`
    * :func:`pyFun.fun3d.Fun3d.PrepareRubberData`
"""

# Numerics
import numpy as np
# File control interface
from cape.fileCntl import FileCntl, re

# ``rubber.data`` class
class RubberData(FileCntl):
    
    
    # Initialization method
    def __init__(self, fname=None):
        # Read the file.
        if fname is not None:
            self.Read(fname)
        # Save the file name.
        self.fname = fname
        
    # Get the next line
    def GetNextLineIndex(self, i, n=1):
        """Get index of the next non-comment line after a specified line number
        
        :Call:
            >>> j = R.GetNextLineIndex(i)
            >>> j = R.GetNextLineIndex(i, n=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *i*: :class:`int`
                Line number
            *n*: :class:`int`
                Number of lines to read
        :Outputs:
            *j*: :class:`int` | ``None``
                Index of next non-comment line; if ``None``, no such line
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Initialize count
        m = 0
        # Loop through remaining lines
        for j in range(i+1, len(self.lines)):
            # Get the line
            line = self.lines[j].lstrip()
            # Check it
            if not line.startswith('#'):
                # Not a comment; good
                m += 1
                # Check if it's time to exit.
                if m >= n:
                    return j
        # If reaching this point, no following line
        return None
        
    # Get the contents of that line
    def GetNextLine(self, i, n=1):
        """Get index of the next non-comment line after a specified line number
        
        :Call:
            >>> line = R.GetNextLine(i)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *i*: :class:`int`
                Line number
            *n*: :class:`int`
                Number of lines to read
        :Outputs:
            *line*: :class:`str`
                Contents of next non-comment line, may be empty
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Get the index
        j = self.GetNextLineIndex(i, n)
        # Get the contents if applicable
        if j is None:
            # Return *fully* empty line, no newline character
            return ''
        else:
            # Read line *j*
            return self.lines[j]
        
    # Get the number of functions
    def GetNFunction(self):
        """Get the number of output/objective/constraint functions
        
        :Call:
            >>> n = R.GetNFunction()
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
        :Outputs:
            *n*: :class:`int` | ``None``
                Number of functions defined
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Get the line
        i = self.GetIndexStartsWith('Number of composite')
        # If no match, assume zero.
        if len(i) == 0: return 0
        # Try the next line
        line = self.GetNextLine(i[0])
        # Try to interpret this line
        try:
            # This should be a single-entry integer
            return int(line)
        except Exception:
            # Unknown
            return None
            
    # Set the number of functions
    def SetNFunction(self, n):
        """Set the number of output/objective/constraint functions
        
        :Call:
            >>> R.SetNFunction(n)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *n*: :class:`int` | ``None``
                Number of functions defined
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Get the line
        i = self.GetIndexStartsWith('Number of composite')
        # Must have a match
        if len(i) == 0:
            raise ValueError(
                ("File '%s' has no line defining number of functions.\n" %
                    self.fname) +
                ("Line must have following format:\n") +
                ("Number of composite functions for design problem statement"))
        # Get the index of the next line
        j = self.GetNextLineIndex(i[0])
        # Set it.
        self.lines[j] = "    %i\n" % n
        
    # Get number of components for a function
    def GetNComp(self, k=1):
        """Get number of components for function *k*
        
        :Call:
            >>> m = R.GetNComp(k)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Composite function number (almost always ``1``)
        :Outputs:
            *m*: :class:`int`
                Number of components for function *k*
        :Versions:
            * 2016-04-27 ``@ddalle``: First version
        """
        # Get the line
        I = self.GetIndexStartsWith('Number of components', k)
        # Line count
        if len(I) < k:
            raise ValueError("No component definitions for function %i" % k)
        # Get the next line
        line = self.GetNextLine(I[k-1])
        # Try to interpret this line
        try:
            # This should be a single-entry integer
            return int(line)
        except Exception:
            # Unknown
            return None
            
    # Set number of components for a function
    def SetNComp(self, k=1, m=1):
        """Set number of components for function *k*
        
        :Call:
            >>> R.SetNComp(k, m=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Composite function number (almost always ``1``)
            *m*: :class:`int`
                Number of components for function *k*
        :Versions:
            * 2016-04-27 ``@ddalle``: First version
        """
        # Get the line
        i = self.GetIndexStartsWith('Number of components', k)
        # Line count
        if len(i) < k:
            raise ValueError("No component definitions for function %i" % k)
        # Get the next line
        j = self.GetNextLineIndex(i[k-1])
        # Set the line.
        self.lines[j] = "    %i\n" % m
    
    # Add another component if necessry
    def AddCoeff(self, k=1, m=None):
        """Add one or more components to a function definition
        
        :Call:
            >>> R.AddCoeff(k)
            >>> R.AddCoeff(k, m)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Composite function number (almost always ``1``)
            *m*: {``None``} | :class:`int`
                Number of components; if ``None``, add one more component
        :Versions:
            * 2016-04-27 ``@ddalle``: First version
        """
        # Add sections if necessary
        for n in range(self.GetNFunction(),k):
            self.AddFunction()
        # Get the number of components
        n = self.GetNComp(k)
        # Default count
        if m is None: m = n+1
        # Do not add if not necessary.
        if m <= n: return
        # Set component count
        self.SetNComp(k, m)
        # Get the start of the component definition
        I = self.GetIndexStartsWith('Current value of function', k)
        # Get last line
        i = I[k-1]
        # Default line
        linedef = '   0 cd   0.00000   1.000   0.000   1.000\n'
        # Set of lines to add
        linesnew = [linedef] * (m-n)
        # Alter line set
        self.lines = self.lines[:i] + linesnew + self.lines[i:]
    
    # Add a new section
    def AddFunction(self):
        """Append an empty section (with default *CD* definition)
        
        :Call:
            >>> R.AddFunction()
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Number of functions
        n = self.GetNFunction()
        # Increase it
        self.SetNFunction(n+1)
        # Append the lines
        self.lines.append('Cost function (1) or constraint (2)\n')
        self.lines.append('    1\n')
        self.lines.append('If constraint, lower and upper bounds\n')
        self.lines.append('    %.15f    %.15f\n' % (0.1,0.5))
        self.lines.append('Number of components for function %3i\n' % (n+1))
        self.lines.append('    1\n')
        self.lines.append('Physical timestep interval where ')
        self.lines.append('function is defined\n')
        self.lines.append('    1     1\n')
        self.lines.append('Composite function weight, target, and power\n')
        self.lines.append('1.0 0.0 1.0\n')
        self.lines.append('Components of function %3i: ' % (n+1))
        self.lines.append('boundary id/name/value/weight/target/power\n')
        self.lines.append('    0 cd  -0.0   1.000   0.00000  1.00\n')
        self.lines.append('Current value of function %3i\n' % (n+1))
        self.lines.append('%12s%18.15f\n' % (" ", 0))
        self.lines.append('Current derivatives of function ')
        self.lines.append('wrt global design variables\n')
        # Write gradient
        for i in range(6):
            self.lines.append('%12s%18.15f\n' % (" ", 0))
    
    # Set get function type
    def GetFunctionType(self, k):
        """Get the type of function *k*
        
        :Call:
            >>> typ = R.GetFunctionType(k)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number
        :Outputs:
            *typ*: ``1`` | ``2``
                Function type index; ``1`` for function and ``2`` for constraint
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Get the lines marking this content
        I = self.GetIndexStartsWith('Cost function')
        # Check section count
        if k >= len(I):
            # Not enough sections
            raise ValueError("Cannot process function %i; only %i functions"
                % (k+1, len(I)))
        # Get the next line
        line = self.GetNextLine(I[k-1])
        # Get the value
        try:
            return int(line)
        except Exception:
            return None
    
    # Set the function type
    def SetFunctionType(self, k, typ):
        """Set the type of function *k*
        
        :Call:
            >>> R.SetFunctionType(k, typ)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number
            *typ*: ``1`` | ``2``
                Function type index; ``1`` for function and ``2`` for constraint
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Add sections if necessary
        for n in range(k,self.GetNFunction()+1):
            self.AddFunction()
        # Get the lines marking this content
        I = self.GetIndexStartsWith('Cost function')
        # Set the contents
        self.lines[I[k-1]] = '    %i\n' % typ
        
    # Get the function component
    def GetCoeffComp(self, k, j=1):
        """Get the component for function *k* component *j*
        
        :Call:
            >>> comp = R.GetFunctionComp(k, j=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (1-based)
            *j*: :class:`int`
                Component number (1-based)
        :Outputs:
            *comp*: :class:`int`
                Face ID; ``0`` for all components
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-27 ``@ddalle``: Added multiple components
        """
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Check number of components
        if self.GetNComp(k) < j:
            raise ValueError(
                ("Requested value from component %i of function %i\n" % (k,j))
                + ("Only %i components defined" % self.GetNComp(k)))
        # Get *j* comps later
        line = self.GetNextLine(I[k-1], j)
        # Get the value
        try:
            return int(line.split()[0])
        except Exception:
            return 0
    
    # Set the function component
    def SetCoeffComp(self, k, comp, j=1):
        """Set the component for function *k*
        
        :Call:
            >>> R.SetCoeffComp(k, comp)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
            *comp*: :class:`int`
                Face ID; ``0`` for all components
            *j*: :class:`int`
                Component number (1-based)
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-27 ``@ddalle``: Added multiple components
        """
        # Add component if necessary.
        self.AddCoeff(k, j)
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Get the next line
        i = self.GetNextLineIndex(I[k-1], j)
        # Get the line.
        line = self.lines[i]
        # Check contents
        V = line.split()
        # Alter the first.
        if len(V) < 1:
            # Initialize line and set component
            V = '%8s' % comp
        else:
            V[0] = '%8s' % comp
        # Save the line
        self.lines[i] = (' '.join(V) + '\n')
        
    # Get the function tag
    def GetCoeffType(self, k, j=1):
        """Get the keyword for coefficient *j* of function *k*
        
        :Call:
            >>> name = R.GetCoeffType(k, j=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (1-based)
            *j*: :class:`int`
                Component number (1-based)
        :Outputs:
            *name*: {``cd``} | :class:`int`
                Function keyword
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-28 ``@ddalle``: Added multiple components
        """
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Get the next line
        line = self.GetNextLine(I[k-1], j)
        # Get the value
        try:
            return line.spit()[1]
        except Exception:
            return 'cd'
        
    # Set the function type
    def SetCoeffType(self, k, name, j=1):
        """Set the keyword for function *k*
        
        :Call:
            >>> R.SetCoeffType(k, name, j=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (1-based)
            *name*: {``cd``} | :class:`int`
                Function keyword
            *j*: :class:`int`
                Component number (1-based)
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-28 ``@ddalle``: Added multiple components
        """
        # Add component if necessary.
        self.AddCoeff(k, j)
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Get the next line
        i = self.GetNextLineIndex(I[k-1], j)
        # Get the line
        line = self.lines[i]
        # Split into parts
        V = line.split()
        # Edit second entry
        if len(V) < 2:
            # Initialize list
            V0 = ['0', str(name)]
            V = V + V0[len(V):]
        else:
            # Edit second entry
            V[1] = str(name)
        # Use 8 chars for first entry
        V[0] = '%8s' % V[0]
        # Save the line.
        self.lines[i] = (' '.join(V) + '\n')
        
    # Get the function weight
    def GetCoeffWeight(self, k, j=1):
        """Get the component weight for coefficient *j* of function *k*
        
        :Call:
            >>> w = R.GetCoeffWeight(k, j=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (1-based)
            *j*: :class:`int`
                Component number (1-based)
        :Outputs:
            *w*: {``1.0``} | :class:`float`
                Function keyword
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-28 ``@ddalle``: Added multiple components
        """
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Get the next line
        line = self.GetNextLine(I[k-1], j)
        # Get the value
        try:
            return float(line.spit()[3])
        except Exception:
            return 1.0
        
    # Set the weight
    def SetCoeffWeight(self, k, w=1.0, j=1):
        """Set the weight for function *k*
        
        :Call:
            >>> R.SetCoeffWeight(k, w, j=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
            *w*: {``1.0``} | :class:`float`
                Function weight
            *j*: :class:`int`
                Component number (1-based)
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-28 ``@ddalle``: Added multiple components
        """
        # Add component if necessary.
        self.AddCoeff(k, j)
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Get the next line
        i = self.GetNextLineIndex(I[k-1], j)
        # Get the line
        line = self.lines[i]
        # Split into parts
        V = line.split()
        # Edit second entry
        if len(V) < 4:
            # Initialize list
            V0 = ['0', 'cd', '0.00000', '%f'%w]
            V = V + V0[len(V):]
        else:
            # Edit second entry
            V[3] = '%f'%w
        # Use 8 chars for first entry
        V[0] = '%8s' % V[0]
        # Save the line.
        self.lines[i] = (' '.join(V) + '\n')
    
    # Get the function weight
    def GetCoeffTarget(self, k, j=1):
        """Get the target value for coefficient *j* of function *k*
        
        :Call:
            >>> t = R.GetCoeffTarget(k, j=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (1-based)
            *j*: :class:`int`
                Component number (1-based)
        :Outputs:
            *t*: {``0.0``} | :class:`float`
                Function target
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-28 ``@ddalle``: Added multiple components
        """
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Get the next line
        line = self.GetNextLine(I[k-1], j)
        # Get the value
        try:
            return float(line.spit()[4])
        except Exception:
            return 0.0
    
    # Set the weight
    def SetCoeffTarget(self, k, t=0.0, j=1):
        """Set the weight for function *k*
        
        :Call:
            >>> R.SetCoeffTarget(k, t, j=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
            *t*: {``0.0``} | :class:`float`
                Function target
            *j*: :class:`int`
                Component number (1-based)
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-28 ``@ddalle``: Added multiple components
        """
        # Add component if necessary.
        self.AddCoeff(k, j)
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Get the next line
        i = self.GetNextLineIndex(I[k-1],j)
        # Get the line
        line = self.lines[i]
        # Split into parts
        V = line.split()
        # Edit second entry
        if len(V) < 5:
            # Initialize list
            V0 = ['0', 'cd', '0.00000', '1.000', '%f'%t]
            V = V + V0[len(V):]
        else:
            # Edit second entry
            V[4] = '%f' % t
        # Use 8 chars for first entry
        V[0] = '%8s' % V[0]
        # Save the line.
        self.lines[i] = (' '.join(V) + '\n')
    
    # Get the function power
    def GetCoeffPower(self, k, j=1):
        """Get the target value for coefficient *j* of function *k*
        
        :Call:
            >>> p = R.GetCoeffPower(k, j=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (1-based)
            *j*: :class:`int`
                Component number (1-based)
        :Outputs:
            *p*: {``1.0``} | :class:`float`
                Function power
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-28 ``@ddalle``: Added multiple components
        """
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Get the next line
        line = self.GetNextLine(I[k-1], j)
        # Get the value
        try:
            return float(line.spit()[5])
        except Exception:
            return 1.0
        
    # Set the power/exponent
    def SetCoeffPower(self, k, p=1.0, j=1):
        """Set the exponent for function *k*
        
        :Call:
            >>> R.SetCoeffPower(k, p, j=1)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
            *p*: {``1.0``} | :class:`float`
                Function exponent: ``w*(v-t)**p``
            *j*: :class:`int`
                Component number (1-based)
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
            * 2016-04-28 ``@ddalle``: Added multiple components
        """
        # Add component if necessary.
        self.AddCoeff(k, j)
        # Get the lines
        I = self.GetIndexStartsWith('Components of function')
        # Get the next line
        i = self.GetNextLineIndex(I[k-1], j)
        # Get the line
        line = self.lines[i]
        # Split into parts
        V = line.split()
        # Edit second entry
        if len(V) < 6:
            # Initialize list
            V0 = ['0', 'cd', '0.00000', '1.000', '0.000', '%f'%p]
            V = V + V0[len(V):]
        else:
            # Edit second entry
            V[5] = '%f' % p
        # Use 8 chars for first entry
        V[0] = '%8s' % V[0]
        # Save the line.
        self.lines[i] = (' '.join(V) + '\n')
    
# class RubberData
