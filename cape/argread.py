"""
Command-Line Argument Processor: :mod:`cape.argread`
====================================================

Parse command-line inputs based on one of two methods.  The first method counts
both "-" and "--" as prefixes for keyword names; this is common among many
advanced programs.  For example, the two following examples would be treated as
equivalent (assuming it is called by some script :file:`myScript.py`.

    .. code-block:: console
    
        $ myScript.py --v --i test.txt
        $ myScript.py -v -i test.txt

The second method assumes single-hyphen options are single-character flags that
can be combined.  This is common in many built-in Unix/Linux utilities.
Consider how ``ls -lh`` is interpreted.  The following two examples would be
interpreted equivalently.

    .. code-block:: console
    
        $ myScript.py -v -i
        $ myScript.py -vi

A third method is provided to have similar behavior to the Unix ``tar`` command.
In this case, the following two commands will be different.

    .. code-block:: console
    
        $ myScript.py -cf mytar.tar
        $ myScript.py --cf mytar.tar
        
The first example sets *c* to ``True`` and *f* to ``"mytar.tar"``; the second
command sets *cf* to ``"mytar.tar"``.
"""

# Process options using any dash as keyword
def readkeys(argv):
    """
    Read list of strings from ``sys.argv`` with any hyphen or double-hyphen as
    an indicator of a keyword.
    
    :Call:
        >>> (args, kwargs) = argread.readkeys(argv)
    
    :Inputs:
        *argv*: :class:`list` (:class:`str`)
            List of string inputs; first entry is ignored (from ``sys.argv``)
    
    :Outputs:
        *args*: :class:`list`
            List of general inputs with no keyword names
        *kwargs*: :class:`dict`
            Dictionary of inputs specified with option flags
    
    :Examples:
        The following shows an example with only general inputs and no options
        
            >>> (a, kw) = readkeys(['ex.sh', 'a.1', '1'])
            >>> a
            ['a.1', '1']
            >>> kw
            {}
            
        This example shows one general input followed by two options.  One of
        the options has an argument associated with it, and the other does not.
            
            >>> (a, kw) = readkeys(['ex.sh', 'a.1', '-i', 'in.tri', '-v'])
            >>> a
            ['a.1']
            >>> kw
            {'i': 'in.tri', 'v': True}
            
        Double-hyphens are interpreted as hyphens.
        
            >>> (a, kw) = readkeys(['ex.sh', '--h', '--i', 'in.tri', 'a.1'])
            >>> a
            ['a.1']
            >>> kw
            {'h': True, 'i': 'in.tri'}
            
    :Versions:
        * 2014.06.10 ``@ddalle``: First version
    """
    # Check the input.
    if type(argv) is not list:
        raise TypeError('Input must be a list of strings.')
    # Initialize outputs.
    args = []
    kwargs = {}
    # Get the number of arguments.
    argc = len(argv)
    # Argument counter (don't process argv[0]).
    iarg = 1
    # Loop until the last argument has been reached.
    for i in range(argc):
        # Check for last input.
        if iarg >= argc: break
        # Read the argument. (convert to str if needed)
        a = str(argv[iarg])
        # Check for hyphens.
        if not a.startswith('-'):
            # General input.
            args.append(a)
            # Go to the next input.
            iarg += 1
        else:
            # Key name starts after '-'s
            k = a.lstrip('-')
            # Increase the arg count.
            iarg += 1
            # Check for more arguments.
            if iarg >= argc:
                # No option value.
                kwargs[k] = True
            else:
                # Read the next argument.
                v = argv[iarg]
                # Check if it's another option.
                if v.startswith('-'):
                    # No option value.
                    kwargs[k] = True
                else:
                    # Store the option value.
                    kwargs[k] = v
                    # Go to the next argument.
                    iarg += 1
    # Return the args and kwargs
    return (args, kwargs)
    
    
# Process options using any dash as keyword
def readflags(argv):
    """
    Read list of strings from ``sys.argv`` with double-hyphen as an indicator 
    of a keyword and a single-hyphen as a list of stackable flags
    
    :Call:
        >>> (args, kwargs) = argread.readflags(argv)
    
    :Inputs:
        *argv*: :class:`list` (:class:`str`)
            List of string inputs; first entry is ignored (from ``sys.argv``)
    
    :Outputs:
        *args*: :class:`list`
            List of general inputs with no keyword names
        *kwargs*: :class:`dict`
            Dictionary of inputs specified with option flags
    
    :Examples:
        The following shows an example with a stacked flag.
        
            >>> (a, kw) = readflags(['ex.sh', 'arg.file', '-lvi']
            >>> a
            ['arg.file']
            >>> kw
            {'l': True, 'v': True, 'i': True}
            
        This example shows the difference between single- and double-hyphens.
            
            >>> (a, kw) = readflags(['ex.sh', '-lv', '--in', 'i.tri'])
            >>> a
            []
            >>> kw
            {'l': True, 'v': True, 'in': 'i.tri'}
            
        The following shows a stacked flag followed by a raw input.
        
            >>> a, kw = readflagstar(['ex.sh', '-tvf', 'fname.dat'])
            >>> a
            ['fname.dat']
            >>> kw
            {'t': True, 'v': True, 'f': True}
            
    :Versions:
        * 2014.06.10 ``@ddalle``: First version
    """
    # Check the input.
    if type(argv) is not list:
        raise TypeError('Input must be a list of strings.')
    # Initialize outputs.
    args = []
    kwargs = {}
    # Get the number of arguments.
    argc = len(argv)
    # Argument counter (don't process argv[0]).
    iarg = 1
    # Loop until the last argument has been reached.
    for i in range(argc):
        # Check for last input.
        if iarg >= argc: break
        # Read the argument. (convert to str if needed)
        a = str(argv[iarg])
        # Check for hyphens.
        if a.startswith('--'):
            # Key name starts after '-'s
            k = a.lstrip('-')
            # Increase the arg count.
            iarg += 1
            # Check for more arguments.
            if iarg >= argc:
                # No option value.
                kwargs[k] = True
            else:
                # Read the next argument.
                v = argv[iarg]
                # Check if it's another option.
                if v.startswith('-'):
                    # No option value.
                    kwargs[k] = True
                else:
                    # Store the option value.
                    kwargs[k] = v
                    # Go to the next argument.
                    iarg += 1
        elif a.startswith('-'):
            # List of flags starts after '-'.
            f = a[1:]
            # Move to next input.
            iarg += 1
            # Check the length.
            if len(f) == 0:
                # Empty flag.
                kwargs[''] = True
            else:
                # List of flags.
                for j in range(len(f)):
                    kwargs[f[j]] = True
        else:
            # General input.
            args.append(a)
            # Go to the next input.
            iarg += 1
        # Check for last input.
        if iarg >= argc: break
    # Return the args and kwargs
    return (args, kwargs)

# Process options using any dash as keyword
def readflagstar(argv):
    """
    Read list of strings from ``sys.argv`` with double-hyphen as an indicator 
    of a keyword and a single-hyphen as a list of stackable flags.  This version
    behaves like `tar` in that the last flag in a group can be used with a
    following value.
    
    :Call:
        >>> (args, kwargs) = argread.readflagstar(argv)
    
    :Inputs:
        *argv*: :class:`list` (:class:`str`)
            List of string inputs; first entry is ignored (from ``sys.argv``)
    
    :Outputs:
        *args*: :class:`list`
            List of general inputs with no keyword names
        *kwargs*: :class:`dict`
            Dictionary of inputs specified with option flags
    
    :Examples:
        The following shows an example with a stacked flag.
        
            >>> (a, kw) = readflagstar(['ex.sh', 'arg.file', '-lvi']
            >>> a
            ['arg.file']
            >>> kw
            {'l': True, 'v': True, 'i': True}
            
        This example shows the difference between single- and double-hyphens.
            
            >>> (a, kw) = readflagstar(['ex.sh', '-lv', '--in', 'i.tri'])
            >>> a
            []
            >>> kw
            {'l': True, 'v': True, 'in': 'i.tri'}
            
        The following shows a stacked flag with a value for the last option
        
            >>> a, kw = readflagstar(['ex.sh', '-tvf', 'fname.dat'])
            >>> a
            []
            >>> kw
            {'t': True, 'v': True, 'f': 'fname.dat'}
            
    :Versions:
        * 2014.10.10 ``@ddalle``: First version
    """
    # Check the input.
    if type(argv) is not list:
        raise TypeError('Input must be a list of strings.')
    # Initialize outputs.
    args = []
    kwargs = {}
    # Get the number of arguments.
    argc = len(argv)
    # Argument counter (don't process argv[0]).
    iarg = 1
    # Loop until the last argument has been reached.
    for i in range(argc):
        # Check for last input.
        if iarg >= argc: break
        # Read the argument. (convert to str if needed)
        a = str(argv[iarg])
        # Check for hyphens.
        if a.startswith('--'):
            # Key name starts after '-'s
            k = a.lstrip('-')
            # Increase the arg count.
            iarg += 1
            # Check for more arguments.
            if iarg >= argc:
                # No option value.
                kwargs[k] = True
            else:
                # Read the next argument.
                v = argv[iarg]
                # Check if it's another option.
                if v.startswith('-'):
                    # No option value.
                    kwargs[k] = True
                else:
                    # Store the option value.
                    kwargs[k] = v
                    # Go to the next argument.
                    iarg += 1
        elif a.startswith('-'):
            # List of flags starts after '-'.
            f = a[1:]
            # Move to next input.
            iarg += 1
            # Check for a blank following command.
            if iarg < argc and (not str(argv[iarg]).startswith('-')):
                # Read the next argument.
                a = str(argv[iarg])
                # Increase the argument count again.
                iarg += 1
                # Check the length.
                if len(f) == 0:
                    # Empty flag.
                    kwargs[''] = a
                else:
                    # Save the last flag with a value
                    #  Example: "tar -xf f.tar"
                    #      ==>   {"x":True, "f":'f.tar'}
                    kwargs[f[-1]] = a
                    # List of flags.
                    for j in range(len(f)-1):
                        kwargs[f[j]] = True
            else:
                # Check the length.
                if len(f) == 0:
                    # Empty flag.
                    kwargs[''] = True
                else:
                    # List of flags.
                    for j in range(len(f)):
                        kwargs[f[j]] = True
        else:
            # General input.
            args.append(a)
            # Go to the next input.
            iarg += 1
        # Check for last input.
        if iarg >= argc: break
    # Return the args and kwargs
    return (args, kwargs)

