#!/usr/bin/env python
"""
FUN3D optimization/adaptation file ``rubber.data``
==================================================

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
    def GetNextLineIndex(self, i):
        """Get index of the next non-comment line after a specified line number
        
        :Call:
            >>> j = R.GetNextLineIndex(i)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *i*: :class:`int`
                Line number
        :Outputs:
            *j*: :class:`int` | ``None``
                Index of next non-comment line; if ``None``, no such line
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Loop through remaining lines
        for j in range(i+1, len(self.lines)):
            # Get the line
            line = self.lines[j].lstrip()
            # Check it
            if not line.startswith('#'):
                # Not a comment; good
                return j
        # If reaching this point, no following line
        return None
        
    # Get the contents of that line
    def GetNextLine(self, i):
        """Get index of the next non-comment line after a specified line number
        
        :Call:
            >>> line = R.GetNextLine(i)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *i*: :class:`int`
                Line number
        :Outputs:
            *line*: :class:`str`
                Contents of next non-comment line, may be empty
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Get the index
        j = self.GetNextLineIndex(i)
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
        i = self.GetLineStartsWith('Number of composite')
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
        i = self.GetLineStartsWith('Number of composite')
        # Must have a match
        if len(i) == 0:
            raise ValueError(
                ("File '%s' has no line defining number of functions.\n" %
                    self.fname) +
                ("Line must have following format:\n") +
                ("Number of composite functions for design problem statement"))
        # Get the index of the next line
        j = self.GetNextLine(i[0])
        # Set it.
        self.lines[j] = "    %i" % n
    
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
        self.lines.append('If constraint, lowr and upper bounds\n')
        self.lines.append('    %.15f    %.15f\n' % (0.1,0.5))
        self.lines.append('Number of components for function %3i\n' % (n+1))
        self.lines.append('Physical timestep interval where ')
        self.lines.append('function is defined\n')
        self.lines.append('    1\n')
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
                Function number (0-based)
        :Outputs:
            *typ*: ``1`` | ``2``
                Function type index; ``1`` for function and ``2`` for constraint
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Get the lines marking this content
        I = self.GetLineStartsWith('Cost function')
        # Check section count
        if k >= len(I):
            # Not enough sections
            raise ValueError("Cannot process function %i; only %i functions"
                % (k+1, len(I)))
        # Get the next line
        line = self.GetNextLine(I[k])
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
                Function number (0-based)
            *typ*: ``1`` | ``2``
                Function type index; ``1`` for function and ``2`` for constraint
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Add sections if necessary
        for n in range(k,self.GetNFunction()+1):
            self.AddFunction()
        # Get the lines marking this content
        I = self.GetLineStartsWith('Cost function')
        # Set the contents
        self.lines[I[k]] = '    %i\n' % typ
        
    # Get the function component
    def GetFunctionComp(self, k):
        """Get the component for function *k*
        
        :Call:
            >>> comp = R.GetFunctionComp(k)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
        :Outputs:
            *comp*: :class:`int`
                Face ID; ``0`` for all components
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Get the lines
        I = self.GetLineStartsWith('Components of function')
        # Get the next line
        line = self.GetNextLine(I[k])
        # Get the value
        try:
            return int(line.split()[0])
        except Exception:
            return 0
    
    # Set the function component
    def SetFunctionComp(self, k, comp):
        """Set the component for function *k*
        
        :Call:
            >>> R.SetFunctionComp(k, comp)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
            *comp*: :class:`int`
                Face ID; ``0`` for all components
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Get the lines
        I = self.GetLineStartsWith('Components of function')
        # Get the next line
        j = self.GetNextLine(I[k])
        # Get the line.
        line = self.lines[j]
        # Check contents
        V = line.split()
        # Alter the first.
        if len(V) < 1:
            # Initialize line and set component
            V = str(comp)
        else:
            V[0] = str(comp)
        # Save the line
        self.lines[j] = (' '.join(V) + '\n')
        
    # Get the function tag
    def GetFunctionName(self, k):
        """Get the keyword for function *k*
        
        :Call:
            >>> kw = R.GetFunctionName(k)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
        :Outputs:
            *name*: {``cd``} | :class:`int`
                Function keyword
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Get the lines
        I = self.GetLineStartsWith('Components of function')
        # Get the next line
        line = self.GetNextLine(I[k])
        # Get the value
        try:
            return line.spit()[1]
        except Exception:
            return 'cd'
        
    # Set the function type
    def SetFunctionName(self, k, name):
        """Set the keyword for function *k*
        
        :Call:
            >>> R.SetFunctionName(k, name)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
            *name*: {``cd``} | :class:`int`
                Function keyword
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Add sections if necessary
        for n in range(k,self.GetNFunction()+1):
            self.AddFunction()
        # Get the lines
        I = self.GetLineStartsWith('Components of function')
        # Get the next line
        j = self.GetNextLine(I[k])
        # Get the line
        line = self.lines[j]
        # Split into parts
        V = line.split()
        # Edit second entry
        if len(V) < 2:
            # Initialize list
            V0 = ['0', str(kw)]
            V = V + V0[len(V):]
        else:
            # Edit second entry
            V[1] = str(kw)
        # Save the line.
        self.lines[j] = (' '.join(V) + '\n')
        
    # Set the weight
    def SetFunctionWeight(self, k, w=1.0):
        """Set the weight for function *k*
        
        :Call:
            >>> R.SetFunctionWeight(k, w)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
            *w*: {``1.0``} | :class:`float`
                Function weight
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Add sections if necessary
        for n in range(k,self.GetNFunction()+1):
            self.AddFunction()
        # Get the lines
        I = self.GetLineStartsWith('Components of function')
        # Get the next line
        j = self.GetNextLine(I[k])
        # Get the line
        line = self.lines[j]
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
        # Save the line.
        self.lines[j] = (' '.join(V) + '\n')
        
    # Set the weight
    def SetFunctionTarget(self, k, t=0.0):
        """Set the weight for function *k*
        
        :Call:
            >>> R.SetFunctionTarget(k, t)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
            *t*: {``0.0``} | :class:`float`
                Function target
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Add sections if necessary
        for n in range(k,self.GetNFunction()+1):
            self.AddFunction()
        # Get the lines
        I = self.GetLineStartsWith('Components of function')
        # Get the next line
        j = self.GetNextLine(I[k])
        # Get the line
        line = self.lines[j]
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
        # Save the line.
        self.lines[j] = (' '.join(V) + '\n')
        
    # Set the power/exponent
    def SetFunctionPower(self, k, p=1.0):
        """Set the exponent for function *k*
        
        :Call:
            >>> R.SetFunctionPower(k, p)
        :Inputs:
            *R*: :class:`pyFun.rubberData.RubberData`
                Interface to FUN3D function definition file
            *k*: :class:`int`
                Function number (0-based)
            *p*: {``1.0``} | :class:`float`
                Function exponent: ``w*(v-t)**p``
        :Versions:
            * 2016-04-22 ``@ddalle``: First version
        """
        # Add sections if necessary
        for n in range(k,self.GetNFunction()+1):
            self.AddFunction()
        # Get the lines
        I = self.GetLineStartsWith('Components of function')
        # Get the next line
        j = self.GetNextLine(I[k])
        # Get the line
        line = self.lines[j]
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
        # Save the line.
        self.lines[j] = (' '.join(V) + '\n')
# class RubberData
