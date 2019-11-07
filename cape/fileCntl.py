"""
:mod:`cape.fileCntl`: File control base module
===============================================

This provides common methods to control objects for various specific
files.  This includes various methods for reading files, splitting it
into sections, and replacing lines based on patterns or regular
expressions.

File manipulation classes for specific files (such as
:class:`pyCart.inputCntl.InputCntl` for Cart3D ``input.cntl`` files) are built
off of this module and its main class. A possibly incomplete list of class
built on this class is given below.

    * :class:`cape.pycart.inputCntl.InputCntl`
    * :class:`cape.namelist.Namelist`
    * :class:`cape.namelist2.Namelist2`
    * :class:`cape.tex.Tex`
    * :class:`cape.pyfun.namelist.Namelist`
    * :class:`cape.pyfun.mapbc.MapBC`
    * :class:`caep.pyover.overNamelist.OverNamelist`

"""

# Standard library
import os
import re

   
# Minor function to convert strings to numbers
def _float(s):
    """Convert string to float when possible
    
    Otherwise the string is returned without raising an exception.
    
    :Call:
        >>> x = _float(s)
    :Inputs:
        *s*: :class:`str`
            String representation of the value to be interpreted
    :Outputs:
        *x*: :class:`float` | :class:`str`
            String converted to float if possible
    :Examples:
        >>> _float('1')
        1.0
        >>> _float('a')
        'a'
        >>> _float('1.1')
        1.1
    :Versions:
        * 2014-06-10 ``@ddalle``: First version
    """
    # Attempt the conversion.
    try:
        # Use built-in converter
        x = float(s)
    except Exception:
        # Return the original string.
        x = s
    # Output
    return x


# Minor function to convert strings to numbers
def _int(s):
    """Convert string to integer when possible
    
    Otherwise the string is returned without raising an exception.
    
    :Call:
        >>> x = _int(s)
    :Inputs:
        *s*: :class:`str`
            String representation of the value to be interpreted
    :Outputs:
        *x*: :class:`int` | :class:`str`
            String converted to int if possible
    :Examples:
        >>> _int('1')
        1
        >>> _int('a')
        'a'
        >>> _int('1.')
        '1.'
    :Versions:
        * 2014-06-10 ``@ddalle``: First version
    """
    # Attempt the conversion.
    try:
        # Use built-in converter
        x = int(s)
    except Exception:
        # Return the original string.
        x = s
    # Output
    return x


# Minor function to convert strings to numbers
def _num(s):
    """Convert string to numeric value when possible

    Otherwise the string is returned without raising an exception.
    
    :Call:
        >>> x = _num(s)
    :Inputs:
        *s*: :class:`str`
            String representation of the value to be interpreted
    :Outputs:
        *x*: :class:`float`, :class:`int`, or :class:`str`
            String converted to int or float if possible
    :Examples:
        >>> _num('1')
        1
        >>> _num('a')
        'a'
        >>> _num('1.')
        1.0
    :Versions:
        * 2014-06-10 ``@ddalle``: First version
    """
    # Attempt the conversion.
    try:
        # Use built-in converter
        x = int(s)
    except Exception:
        # Try again with the float converter.
        try:
            x = float(s)
        except Exception:
            # Return the original string.
            x = s
    # Output
    return x


# File control class
class FileCntl(object):
    """Base file control class
    
    The lines of the file can be split into sections based on a regular
    expression (see :func:`cape.fileCntl.FileCntl.SplitToSections`);
    most methods will keep the overall line list and the section
    breakout consistent.
    
    :Call:
        >>> FC = cape.fileCntl.FileCntl(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read from and manipulate
    :Data members:
        *FC.fname*: :class:`str`
            Name of file instance was read from
        *FC.lines*: :class:`list` (:class:`str`)
            List of all lines in the file (to use for replacement)
        *FC.SectionNames*: :class:`list` (:class:`str`)
            List of section titles if present
        *FC.Section*: :class:`dict` (:class:`list` (:class:`str`))
            Dictionary of the lines in each section, if present
    """
    
    # Initialization method; not useful for derived classes
    def __init__(self, fname=None):
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname

    # Display method
    def __repr__(self):
        """Display method for file control class
        """
        # Versions:
        #  2014.06.03 @ddalle  : First version
        
        # Initialize the string.
        s = '<%s("%s", %i lines' % (
            self.__class__.__name__,
            self.fname,
            len(self.lines))
        # Check for number of sections.
        if hasattr(self, 'SectionNames'):
            # Count non trivial section names
            nsec = sum([not sec.startswith("_") for sec in self.SectionNames])
            # Write the number of sections
            s = s + ", %i sections)>" % nsec
        else:
            # Just close the string.
            s = s + ")>"
        return s

    # Read the file.
    def Read(self, fname):
        """Read text from file
        
        :Call:
            >>> FC.Read(fname)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *fname*: :class:`str`
                Name of file to read from
        :Effects:
            *FC.lines*: :class:`list`
                List of lines in file is created
            *FC._updated_sections*: :class:`bool`
                Whether section breakouts have been updated
            *FC._updated_lines*: :class:`bool`
                Flag for update status of global lines
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Read the file.
        if fname is None or not os.path.isfile(fname):
            # No file: initialize empty content
            self.lines = []
        else:
            # Open the file and read the lines.
            self.lines = open(fname).readlines()
        # Initialize update statuses.
        self._updated_sections = False
        self._updated_lines = False

    # Function to split into sections
    def SplitToSections(self, reg=r"\$__([\w_]+)", ngr=1, begin=True):
        r"""Split into sections based on starting regular expression
        
        :Call:
            >>> FC.SplitToSections()
            >>> FC.SplitToSections(reg="\$__([\w_]+)", ngr=1, **kw)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *reg*: :class:`str`
                Regular expression for recognizing the start of a new
                section. By default this looks for sections that start
                with ``"$__"`` as in Cart3D ``input.cntl`` files.  The
                regular expression must also include a group (meaning
                content between parentheses) to capture the *name* of
                the section.  Thus the default value of
                ``"\$__([\w_]+)"`` finds any name that consists of word
                characters and/or underscores.
            *ngr*: {``1``} | :class:`int` | :class:`str`
                Group number from which to take name of section.  This
                is always ``1`` unless the section-starting regular
                expression has more than one explicit group.  Note that
                using ``1`` instead of ``0`` means that an explicit
                group using parentheses is required.  A string can be
                used if the groups have names in the regular expression
                *reg*.
            *begin*: {``True``} | ``False``
                Whether section regular expression must begin line
            *endreg*: {``None``} | :class:`str`
                Optional regular expression for end of section.  If
                used, some lines will end up in sections called
                 ``"_inter1"``, ``"_inter2"``, etc.
        :Effects:
            *FC.SectionNames*: :class:`list`
                List of section names is created (includes "_header")
            *FC.Section*: :class:`dict`
                Dictionary of section line lists is created
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Initial section name
        sec = "_header"
        # Initialize the sections.
        self.SectionNames = [sec]
        self.Section = {sec: []}
        # Compile regular expression
        regexa = re.compile(reg)
        for line in self.lines:
            # Search for the new-section regular expression
            if begin:
                # Search from beginning of line (neglecting whitespace)
                m = regexa.match(line.lstrip())
            else:
                # Search anywhere in line
                m = regexa.search(line)
            # Check if there was a match.
            if m:
                # Get the new section name
                sec = m.group(ngr)
                # Start the new section
                self.SectionNames.append(sec)
                self.Section[sec] = [line]
            else:
                # Append the line to the current section.
                self.Section[sec].append(line)

    # Function to split into sections with ends
    def SplitToBlocks(
            self, reg=r"\$__([\w_]+)", ngr=1, **kw):
        r"""Split lines into sections based on start and end
        
        :Call:
            >>> FC.SplitToBlocks()
            >>> FC.SplitToBlocks(reg="\$__([\w_]+)", ngr=1, **kw)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *reg*: :class:`str`
                Regular expression for recognizing the start of a new
                section. By default this looks for sections that start
                with ``"$__"`` as inCart3D ``input.cntl`` files.  The
                regular expression must also include a group (meaning
                content between parentheses) to capture the *name* of
                the section.  Thus the default value of
                ``"\$__([\w_]+)"`` finds any name that consists of word
                characters and/or underscores.
            *ngr*: {``1``} | :class:`int` | :class:`str`
                Group number from which to take name of section.  This
                is always ``1`` unless the section-starting regular
                expression has more than one explicit group.  Note that
                using ``1`` instead of ``0`` means that an explicit
                group using parentheses is required. A string can be
                used if the groups have names in the regular expression
                *reg*.
            *begin*: {``True``} | ``False``
                Whether section regular expression must begin line
            *endreg*: {``None``} | :class:`str`
                Optional regular expression for end of section.  If
                used, some lines will end up in sections called
                ``"_inter1"``, ``"_inter2"``, etc.
            *endbegin*: {*begin*} | ``True`` | ``False``
                Whether section-end regular expression must begin line
            *endngr*: {*ngr*} | :class:`int` | :class:`str`
                Group number of name for title of end-of-section regex
        :Effects:
            *FC.SectionNames*: :class:`list`
                List of section names is created (includes "_header")
            *FC.Section*: :class:`dict`
                Dictionary of section line lists is created
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Process inputs
        begin  = kw.pop("begin", True)
        endreg = kw.pop("endreg", None)
        ngrb   = kw.pop("endngr", ngr)
        endbeg = kw.pop("endbegin", begin)
        # Check for unprocessed keywords
        if kw:
            # Get first key
            k, v = kw.popitem()
            raise IOError("Received unrecognized keyword '%s'" % k)
        # Initial section name
        sec = "_header"
        # Number of intermediate sections
        nint = 0
        # Initialize the sections.
        self.SectionNames = [sec]
        self.Section = {sec: []}
        # Compile regular expression
        regexa = re.compile(reg)
        # End of line
        if endreg is None:
            # No end-of-section
            regexb = None
        else:
            # Compile end-of-section regular expression
            regexb = re.compile(endreg)
        # Loop through the lines.
        for line in self.lines:
            # Search for the new-section regular expression
            if begin:
                # Search from beginning of line (neglecting whitespace)
                m = regexa.match(line.lstrip())
            else:
                # Search anywhere in line
                m = regexa.search(line)
            # Check if there was a match.
            if m:
                # Get the new section name
                grp = m.group(ngr)
                # Very special check for formats with section ends
                if (regexb is None) or (grp != sec):
                    # Start the new section
                    sec = grp
                    self.SectionNames.append(sec)
                    self.Section[sec] = [line]
                    # Do not allow begin section to also end section
                    continue
                else:
                    # This is still part of previous section
                    self.Section[sec].append(line)
            else:
                # Append the line to the current section.
                self.Section[sec].append(line)
            # Check for end-of-section check
            if regexb:
                # Check the end-of-section regex
                if endbeg:
                    # Search from beginning of line
                    m = regexb.match(line.lstrip())
                else:
                    # Search anywhere in line
                    m = regexb.search(line)
                # Check if there was a match
                if m:
                    # Try to check the section name
                    try:
                        # Get group name
                        grp = m.group(ngrb)
                        # Check name
                        if sec != grp:
                            raise ValueError(
                                "Section '%s' ends with '%s'" % (sec, grp))
                    except IndexError:
                        # End-of-section marker probably doesn't have group
                        pass
                    # Move to next intermediate section
                    nint += 1
                    sec = "_inter%i" % nint
                    # Start the section
                    self.SectionNames.append(sec)
                    self.Section[sec] = []

    # Function to update the text based on the section content.
    def UpdateLines(self):
        """Update the global file text list from current section content
        
        :Call:
            >>> FC.UpdateLines()
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl` | derivative
                File control instance
        :Effects:
            *FC.lines*: :class:`list`
                Lines are rewritten to match the sequence of lines from
                the sections
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Check for lines
        if not self._updated_sections:
            # No updates.
            return None
        # Reinitialize the lines.
        self.lines = []
        # Loop through the sections.
        for sec in self.SectionNames:
            # Join the lines in that section.
            self.lines.extend(self.Section[sec])
        # The lines are now up-to-date.
        self._updated_sections = False
        # Done.
        return None
        
    # Function to update the text based on the section content.
    def UpdateSections(self):
        """Remake the section split if necessary

        This runs :func:`SplitToSections()` is run if
        *FC._updated_lines* is ``True``.
        
        :Call:
            >>> FC.UpdateSections()
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Check for lines
        if not self._updated_lines:
            # No updates.
            return None
        # Redo the split.
        self.SplitToSections()
        self._updated_lines = False
        # Done.
        return None
        
    # Method to ensure that an instance has a certain section
    def AssertSection(self, sec):
        """Assert that a certain section is present
        
        :Call:
            >>> FC.AssertSection(sec)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance, defaults to *FC.fname*
            *sec*: :class:`str`
                Name of section to check for
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Update sections.
        self.UpdateSections()
        # Check for the section.
        if sec not in self.SectionNames:
            raise KeyError(
                "File control instance does not have section '%s'" % sec)
        # Done
        return None
        
    # Method to write the file.
    def _Write(self, fname=None):
        """Write to text file
        
        :Call:
            >>> FC._Write()
            >>> FC._Write(fname)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl` | derivative
                File control instance, defaults to *FC.fname*
            *fname*: :class:`str`
                Name of file to write to
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Update the lines if appropriate.
        self.UpdateLines()
        # Default file name.
        if fname is None: fname = self.fname
        # Open the new file.
        f = open(fname, 'w')
        # Write the joined text.
        f.write("".join(self.lines))
        # Close the file and exit.
        f.close()
        
    # Method to write the file.
    def Write(self, fname=None):
        """Write to text file
        
        :Call:
            >>> FC.Write()
            >>> FC.Write(fname)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance, defaults to *FC.fname*
            *fname*: :class:`str`
                Name of file to write to
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
            * 2015-11-16 ``@ddalle``: Moved contents to :func:`_Write`
        """
        # Update the lines if appropriate.
        self._Write(fname)
        
    # Method to write the file as an executable.
    def WriteEx(self, fname=None):
        """Write to text file as an executable script
        
        :Call:
            >>> FC.WriteEx()
            >>> FC.WriteEx(fname)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance, defaults to *FC.fname*
            *fname*: :class:`str`
                Name of file to write to
        :Versions:
            * 2014-06-23 ``@ddalle``: First version
        """
        # Write the file.
        self._Write(fname)
        # Get the mode of the file
        fmod = os.stat(fname).st_mode & 0o7777
        # Make sure the user-executable bit is set.
        fmod = fmod | 0o700
        # Check for group-readable and universe-readable
        if fmod & 0o040: fmod = fmod | 0o070
        if fmod & 0o004: fmod = fmod | 0o007
        # Change the mode.
        os.chmod(fname, fmod)

    # Method to replace a line that starts with a given string
    def ReplaceLineStartsWith(self, start, line, imin=0, nmax=None):
        """Replace lines starting with fixed text

        Find all lines that begin with a certain string and replace them
        with another string.  Note that the entire line is replaced, not
        just the initial string.
        
        Leading spaces are ignored during the match tests.
        
        :Call:
            >>> n = FC.ReplaceLineStartsWith(start, line, imin=0, nmax=None)
            >>> n = FC.ReplaceLineStartsWith(start, lines, imin=0, nmax=None)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *start*: :class:`str`
                String to test as literal match for beginning of line
            *line*: :class:`str`
                String to replace every match with
            *lines*: :class:`list`
                List of strings for replacements
            *imin*: {``0``} | :class:`int` >= 0
                Do not make replacements for matches with index < *imin*
            *nmax*: {``None``} | :class:`int` > 0
                Make at most *nmax* substitutions
        :Outputs:
            *n*: :class:`int`
                Number of matches found
        :Effects:
            *FC.lines*: Some of the lines may be affected
            *FC._updated_lines*: Set to ``True``
        :Examples:
            Suppose that *FC* has the following two lines.
            
                ``Mach      8.00   # some comment\n``
                
                ``Mach      Mach_TMP\n``
            
            Then this example will replace *both* lines with the string
             ``Mach 4.0``
            
                >>> FC.ReplaceLineStartsWith('Mach', 'Mach 4.0')
                
            This example replaces each line with a different value for
            the Mach number.
            
                >>> FC.ReplaceLineStartsWith(
                    'Mach', ['Mach 2.0', 'Mach 4.0']
                
            Finally, this example is different from the first example in
            that it will replace the first line and then quit before it
            can find the second match.
            
                >>> FC.ReplaceLineStartsWith('Mach', ['Mach 4.0'])
                
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
            * 2019-06-19 ``@ddalle``: Added *imin* and *nmax*
        """
        # Set the update status.
        self.UpdateLines()
        self._updated_lines = True
        # Number of matches.
        n = 0
        # Loop through the lines.
        for i in range(len(self.lines)):
            # Get the line.
            L = self.lines[i]
            # Check for a match.
            if L.startswith(start):
                # Check *imin* index
                if i < imin:
                    # Increase count but don't substitute
                    n += 1
                    continue
                # Check for the replacement type.
                if type(line) is str:
                    # Replace the line.
                    self.lines[i] = line
                    n += 1
                else:
                    # Replace the line based on the list.
                    self.lines[i] = line[n]
                    # Increase the match count.
                    n += 1
                    # Check for end of matches.
                    if n >= len(line):
                        return len(line)
                # Check maximum substitution count
                if (nmax is not None) and (n >= nmax):
                    break
        # Done
        return n - max(imin, 0)
        
    # Method to replace a line only in a certain section
    def ReplaceLineInSectionStartsWith(
            self, sec, start, line, imin=0, nmax=None):
        """Make replacements within section based on starting string

        Find all lines in a certain section that start with a specified
        literal string and replace the entire line with the specified text.
        
        :Call:
            >>> n = FC.ReplaceLineInSectionStartsWith(
                , start, line, **kw)
            >>> n = FC.ReplaceLineInSectionStartsWith(
                sec, start, lines, **kw)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *start*: :class:`str`
                String to test as literal match for line start
            *line*: :class:`str`
                String to replace every match with
            *lines*: :class:`list`
                List of replacement strings
            *imin*: {``0``} | :class:`int` >= 0
                Do not make replacements for matches with index < *imin*
            *nmax*: {``None``} | :class:`int` > 0
                Make at most *nmax* substitutions
        :Outputs:
            *n*: :class:`int`
                Number of matches found
        :Effects:
            Some lines in *FC.Section[sec]* may be replaced.
        :See also:
            This function is similar to
            :func:`cape.fileCntl.FileCntl.ReplaceLineStartsWith` except
            that the search is restricted to a specified section.
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Number of matches.
        n = 0
        # Update the sections.
        self.UpdateSections()
        # Check if the section exists.
        if sec not in self.SectionNames: return n
        # Set the update status.
        self._updated_sections = True
        # Loop through the lines.
        for i in range(len(self.Section[sec])):
            # Get the line.
            L = self.Section[sec][i]
            # Check for a match.
            if L.startswith(start):
                # Check *imin* index
                if i < imin:
                    # Increase count but don't substitute
                    n += 1
                    continue
                # Check for the replacement type.
                if type(line) is str:
                    # Replace the line.
                    self.Section[sec][i] = line
                    n += 1
                else:
                    # Replace the line based on the match count.
                    self.Section[sec][i] = line[n]
                    # Increase the match count.
                    n += 1
                    # Check for end of matches.
                    if n >= len(line):
                        return len(line)
                # Check maximum substitution count
                if (nmax is not None) and (n >= nmax):
                    break
        # Done.
        return n - max(imin, 0)
        
    # Method to insert a line somewhere
    def InsertLine(self, i, line):
        """Insert a line of text somewhere into the text
        
        :Call:
            >>> FC.InsertLine(i, line)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *i*: :class:`int`
                Index to which to insert the line
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Set the update flags.
        self.UpdateLines()
        self._updated_lines = True
        # Insert the line.
        self.lines.insert(i, line)
        
    # Method to append a line
    def AppendLine(self, line):
        """Append a line of text to *FC.lines*
        
        :Call:
            >>> FC.AppendLine(line)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Set the update flag.
        self.UpdateLines()
        self._updated_lines = True
        # Insert the line.
        self.lines.append(line)
        
    # Method to append a line
    def PrependLine(self, line):
        """Prepend a line of text to *FC.lines*
        
        :Call:
            >>> FC.PrependLine(line)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Set the update flag.
        self.UpdateLines()
        self._updated_lines = True
        # Insert the line.
        self.lines.prepend(line)

    # Method to insert a line somewhere
    def InsertLineToSection(self, sec, i, line):
        """Insert a line of text somewhere into the text of a section
        
        :Call:
            >>> FC.InsertLineToSection(sec, i, line)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to update
            *i*: :class:`int`
                Index to which to insert the line
            *line*: :class:`str`
                String to add
        :Effects:
            A line is inserted to *FC.Section[sec]*
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Set the update flags.
        self.UpdateSections()
        self._updated_sections = True
        # Check for the section
        self.AssertSection(sec)
        # Insert the line.
        self.Section[sec].insert(i, line)
        
    # Method to append a line somewhere
    def AppendLineToSection(self, sec, line):
        """Append a line of text to a section
        
        :Call:
            >>> FC.AppendLineToSection(sec, line)
        :Inputs:
            *FC*: :class:`pyCart.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to update
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Set the update flags.
        self.UpdateSections()
        self._updated_sections = True
        # Check for the section
        self.AssertSection(sec)
        # Insert the line.
        self.Section[sec].append(line)
        
    # Method to prepend a line somewhere
    def PrependLineToSection(self, sec, line):
        """Prepend a line of text to a section
        
        :Call:
            >>> FC.PrependLineToSection(sec, line)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to update
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Set the update flags.
        self.UpdateSections()
        self._updated_sections = True
        # Check for the section
        self.AssertSection(sec)
        # Insert the line.
        self.Section[sec].insert(1, line)
        
    # Method to delete a line that starts with a certain literal
    def DeleteLineStartsWith(self, start, count=1):
        """Delete lines that start with given text up to *count* times
        
        :Call:
            >>> n = FC.DeleteLineStartsWith(start)
            >>> n = FC.DeleteLineStartsWith(start, count)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *start*: :class:`str`
                Line-starting string to search for
            *count*: {``1``} | :class:`int`
                Maximum number of lines to delete
        :Outputs:
            *n*: :class:`int`
                Number of deletions made
        :Effects:
            Lines in *FC.lines* that start with *start* are removed
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Initialize the deletion count.
        n = 0
        # Update the text.
        self.UpdateLines()
        # Line number
        i = 0
        # Loop backward through the lines.
        while i < len(self.lines):
            # Get the line.
            L = self.lines[i]
            # Check it.
            if L.startswith(start):
                # Increase the count.
                n += 1
                self._updated_lines = True
                # Delete the line.
                self.lines.__delitem__(i)
                # Check for limit.
                if n >= count:
                    return n
            else:
                # Increase line number.
                i += 1
        # Done.
        return n
        
    # Method to delete a line from a section that starts with a certain literal
    def DeleteLineInSectionStartsWith(self, sec, start, count=1):
        """Delete lines based on start text and section name
        
        :Call:
            >>> n = FC.DeleteLineInSectionStartsWith(sec, start)
            >>> n = FC.DeleteLineInSectionStartsWith(sec, start, count)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search
            *start*: :class:`str`
                Line-starting string to search for
            *count*: {``1``} | :class:`int`
                Maximum number of lines to delete
        :Outputs:
            *n*: :class:`int`
                Number of deletions made
        :Effects:
            Lines in *FC.Section[sec]* may be removed if they start with
            *start*.
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Initialize the deletion count.
        n = 0
        # Update the sections.
        self.UpdateSections()
        # Check for the section.
        if sec not in self.SectionNames: return n
        # Line number
        i = 0
        # Loop backward through the lines.
        while i < len(self.Section[sec]):
            # Get the line.
            L = self.Section[sec][i]
            # Check it.
            if L.startswith(start):
                # Increase the count.
                n += 1
                self._updated_sections = True
                # Delete the line.
                self.Section[sec].__delitem__(i)
                # Check for limit.
                if (count) and (n >= count):
                    return n
            else:
                # Increase the line number.
                i += 1
        # Done.
        return n
        
    # Replace a line or add it if not found
    def ReplaceOrAddLineStartsWith(self, start, line, i=None, **kw):
        """Replace a line or add a new one

        Replace a line that starts with a given literal string or add
        the line if no matches are found.
        
        :Call:
            >>> FC.ReplaceOrAddLineStartsWith(start, line, **kw)
            >>> FC.ReplaceOrAddLineStartsWith(start, line, i, **kw)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *start*: :class:`str`
                Target beginning of line to replace
            *line*: :class:`str`
                String to replace every match with
            *i*: {``None``} | :class:`int`
                Location to add line, negative ok, (default is append)
            *imin*: {``0``} | :class:`int` >= 0
                Do not make replacements for matches with index < *imin*
            *nmax*: {``None``} | :class:`int` > 0
                Make at most *nmax* substitutions
        :Effects:
            Replaces line in section *FC.lines* or adds it if not found
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Call the replace method (only perform once).
        n = self.ReplaceLineStartsWith(start, [line], **kw)
        # Check for a match.
        if not n:
            # Check where to add the line.
            if i is None:
                # Append.
                self.lines.append(line)
            else:
                # Get the section
                lines = self.lines
                # Correct for empty lines
                if i < 0:
                    # Count empty lines at the end
                    j = self.CountEmptyEnd(lines)
                    # Insert at specified location
                    lines.insert(i-j, line)
                else:
                    # Insert at specified location.
                    lines.insert(i, line)
        
    # Replace a line or add (from one section) if not found
    def ReplaceOrAddLineToSectionStartsWith(
            self, sec, start, line, i=None, **kw):
        """Replace a line or add a new one (within section)

        Replace a line in a specified section that starts with a given
        literal  string or add the line to the section if no matches
        are found.
        
        :Call:
            >>> FC.ReplaceOrAddLineToSectionStartsWith(sec, start, line)
            >>> FC.ReplaceOrAddLineToSectionStartsWith(sec, start, line,
                i=None, **kw)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *start*: :class:`str`
                Target line start for replacement
            *line*: :class:`str`
                String to replace every match with
            *i*: {``None``} | :class:`int`
                Location to add line (by default it is appended)
            *imin*: {``0``} | :class:`int` >= 0
                Do not make replacements for matches with index < *imin*
            *nmax*: {``None``} | :class:`int` > 0
                Make at most *nmax* substitutions
        :Effects:
            Replaces line in *FC.Section[sec]* or adds it if not found
        :Versions:
            * 2014-06-03 ``@ddalle``: First version
        """
        # Call the replace method (only perform once).
        n = self.ReplaceLineInSectionStartsWith(sec, start, [line], **kw)
        # Must have the section.
        self.AssertSection(sec)
        # Check for a match.
        if not n:
            # Check where to add the line.
            if i is None:
                # Append.
                self.Section[sec].append(line)
            else:
                # Get the section
                lines = self.Section[sec]
                # Correct for empty lines
                if i < 0:
                    # Count empty lines at the end
                    j = self.CountEmptyEnd(lines)
                    # Insert at specified location
                    lines.insert(i-j, line)
                else:
                    # Insert at specified location.
                    lines.insert(i, line)
        # Done
        return None
        
    # Method to replace a line that starts with a regular expression
    def ReplaceLineSearch(self, reg, line, imin=0, nmax=None):
        r"""Replace lines based on initial regular expression

        Find all lines that begin with a certain regular expression and
        replace them with another string.  Note that the entire line is
        replaced, not just the regular expression.
        
        Leading spaces are ignored during the match tests.
        
        :Call:
            >>> n = FC.ReplaceLineSearch(reg, line, imin=0, nmax=None)
            >>> n = FC.ReplaceLineSearch(reg, lines, imin=0, nmax=None)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *reg*: :class:`str`
                Target regular expression for line starts
            *line*: :class:`str`
                String to replace every match with
            *lines*: :class:`list`
                Multiple replacements
            *imin*: {``0``} | :class:`int` >= 0
                Do not make replacements for matches with index < *imin*
            *nmax*: {``None``} | :class:`int` > 0
                Make at most *nmax* substitutions
        :Outputs:
            *n*: :class:`int`
                Number of matches found
        :Effects:
            *FC.lines*: Some of the lines may be affected
            *FC._updated_lines*: Set to ``True``
        :Examples:
            Suppose that *FC* has the following two lines.
            
                ``Mach      8.00   # some comment\n``
                
                ``Mach    4\n``
            
            Then this example will replace *both* lines with
            ``Mach 2.0``
            
                >>> FC.ReplaceLineSearch('Mach\s+[0-9.]+', 'Mach 2.0')
                
            This example replaces each line with a different value for
            the Mach number.
            
                >>> FC.ReplaceLineSearch('Mach\s+[0-9.]+',
                    ['Mach 2.0', 'Mach 2.5'])
                
            Finally, this example is different from the first example in
            that it will replace the first line and then quit before it
            can find the second match.
            
                >>> FC.ReplaceLineSearch('Mach\s+[0-9.]+', ['Mach 2.0'])
                
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
        # Set the update status.
        self.UpdateLines()
        self._updated_lines = True
        # Number of matches.
        n = 0
        # Loop through the lines.
        for i in range(len(self.lines)):
            # Get the line.
            L = self.lines[i]
            # Check for a match.
            if re.search(reg, L):
                # Check *imin* index
                if i < imin:
                    # Increase count but don't substitute
                    n += 1
                    continue
                # Check for the replacement type.
                if type(line) is str:
                    # Replace the line.
                    self.lines[i] = line
                    n += 1
                else:
                    # Replace the line based on the list.
                    self.lines[i] = line[n]
                    # Increase the match count.
                    n += 1
                    # Check for end of matches.
                    if n >= len(line):
                        return len(line)
                # Check maximum substitution count
                if (nmax is not None) and (n >= nmax):
                    break
        # Done
        return n - max(0, imin)
        
    # Method to replace a line only in a certain section
    def ReplaceLineInSectionSearch(self, sec, reg, line, imin=0, nmax=None):
        """
        Find all lines in a certain section that start with a specified regular
        expression and replace the entire lines with the specified text.
        
        :Call:
            >>> n = FC.ReplaceLineInSectionSearch(sec, reg, line, **kw)
            >>> n = FC.ReplaceLineInSectionSearch(sec, reg, lines, **kw)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *reg*: :class:`str`
                Regular expression to search for at beginning of each line
            *line*: :class:`str`
                String to replace every match with
            *lines*: :class:`list`
                List of strings to match first ``len(lines)`` matches with
            *imin*: {``0``} | :class:`int` >= 0
                Do not make replacements for matches with index < *imin*
            *nmax*: {``None``} | :class:`int` > 0
                Make at most *nmax* substitutions
        :Outputs:
            *n*: :class:`int`
                Number of matches found
        :Effects:
            Some lines in *FC.Section[sec]* may be replaced.
        :See also:
            This function is similar to
            :func:`pyCart.fileCntl.FileCntl.ReplaceLineSearch` except that
            the search is restricted to a specified section.
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
        # Number of matches.
        n = 0
        # Update the sections.
        self.UpdateSections()
        # Check if the section exists.
        if sec not in self.SectionNames: return n
        # Set the update status.
        self._updated_sections = True
        # Loop through the lines.
        for i in range(len(self.Section[sec])):
            # Get the line.
            L = self.Section[sec][i]
            # Check for a match.
            if re.search(reg, L):
                # Check *imin* index
                if i < imin:
                    # Increase count but don't substitute
                    n += 1
                    continue
                # Check for the replacement type.
                if type(line) is str:
                    # Replace the line.
                    self.Section[sec][i] = line
                    n += 1
                else:
                    # Replace the line based on the match count.
                    self.Section[sec][i] = line[n]
                    # Increase the match count.
                    n += 1
                    # Check for end of matches.
                    if n >= len(line):
                        return len(line)
                # Check maximum substitution count
                if (nmax is not None) and (n >= nmax):
                    break
        # Done.
        return n
        
    # Replace a line or add it if not found
    def ReplaceOrAddLineSearch(self, reg, line, i=None, **kw):
        """
        Replace a line that starts with a given regular expression or add the
        line if no matches are found.
        
        :Call:
            >>> FC.ReplaceOrAddLineSearch(reg, line, **kw)
            >>> FC.ReplaceOrAddLineSearch(reg, line, i, **kw)
        :Inputs:
            *FC*: :class:`pyCart.fileCntl.FileCntl`
                File control instance
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *line*: :class:`str`
                String to replace first match with
            *i*: :class:`int`
                Location to add line (by default it is appended)
            *imin*: {``0``} | :class:`int` >= 0
                Do not make replacements for matches with index < *imin*
            *nmax*: {``None``} | :class:`int` > 0
                Make at most *nmax* substitutions
        :Effects:
            Replaces line in section *FC.lines* or adds it if not found
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
        # Call the replace method (only perform once).
        n = self.ReplaceLineSearch(reg, [line], **kw)
        # Check for a match.
        if not n:
            # Check where to add the line.
            if i is None:
                # Append.
                self.lines.append(line)
            else:
                # Get the section
                lines = self.lines
                # Correct for empty lines
                if i < 0:
                    # Count empty lines at the end
                    j = self.CountEmptyEnd(lines)
                    # Insert at specified location
                    lines.insert(i-j, line)
                else:
                    # Insert at specified location.
                    lines.insert(i, line)
        # Done
        return None
        
    # Count empty lines at the end of a section
    def CountEmptyEnd(self, lines):
        """Count empty lines at the end of a list of lines
        
        :Call:
            >>> n = FC.CountEmptyEnd(lines)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *lines*: :class:`list` (:class:`str`)
                List of lines in section or file
        :Outputs:
            *n*: :class:`int`
                Number of trailing empty lines
        :Versions:
            * 2016-04-18 ``@ddalle``: First version
        """
        # Initialize count
        n = 0
        # Loop through lines
        for line in lines[-1::-1]:
            # Check if it's empty
            if len(line.strip()) != 0:
                break
            # Count the line
            n += 1
        # Output
        return n
        
    # Count empty lines at the end of a section
    def CountEmptyStart(self, lines):
        """Count empty lines at the start of a list of lines
        
        :Call:
            >>> n = FC.CountEmptyStart(lines)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *lines*: :class:`list` (:class:`str`)
                List of lines in section or file
        :Outputs:
            *n*: :class:`int`
                Number of trailing empty lines
        :Versions:
            * 2016-04-18 ``@ddalle``: First version
        """
        # Initialize count
        n = 0
        # Loop through lines
        for line in lines:
            # Check if it's empty
            if len(line.strip()) != 0:
                break
            # Count the line
            n += 1
        # Output
        return n

    # Replace a line or add (from one section) if not found
    def ReplaceOrAddLineToSectionSearch(self, sec, reg, line, i=None):
        """Replace a line in a specified section

        Replace a line in a specified section that starts with a given
        regular  expression or add the line to the section if no matches
        are found.
        
        :Call:
            >>> FC.ReplaceOrAddLineToSectionStartsWith(sec, reg, line)
            >>> FC.ReplaceOrAddLineToSectionStartsWith(sec, reg, line,
                i=None, **kw)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *line*: :class:`str`
                String to replace every match with
            *i*: {```None``} | :class:`int`
                Location to add line (by default it is appended)
        :Effects:
            Replaces line in *FC.Section[sec]* or adds it if not found
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
        # Call the replace method (only perform once).
        n = self.ReplaceLineInSectionSearch(sec, reg, [line])
        # Must have the section.
        self.AssertSection(sec)
        # Check for a match.
        if not n:
            # Check where to add the line.
            if i is None:
                # Append.
                self.Section[sec].append(line)
            else:
                # Get the section
                lines = self.Section[sec]
                # Correct for empty lines
                if i < 0:
                    # Count empty lines at the end
                    j = self.CountEmptyEnd(lines)
                    # Insert at specified location
                    lines.insert(i-j, line)
                else:
                    # Insert at specified location.
                    lines.insert(i, line)
        # Done
        return None
        
    # Get a line that starts with a literal
    def GetLineStartsWith(self, start, n=None):
        """Find lines that start with a given literal pattern
        
        :Call:
            >>> lines = FC.GetLineStartsWith(start)
            >>> lines = FC.GetLineStartsWith(start, n)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *start*: :class:`str`
                String to test as match for beginning of each line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *lines*: :class:`list` (:class:`str`)
                List of lines that match pattern
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Set the update status.
        self.UpdateLines()
        # Initialize matches
        lines = []
        # Number of matches
        m = 0
        # Loop through the lines.
        for L in self.lines:
            # Check for maximum matches.
            if n and (m >= n):
                break
            # Check for a match.
            if L.startswith(start):
                # Add to match list
                lines.append(L)
                # Increase count
                m += 1
        # Done
        return lines
        
    # Get a line that starts with a literal
    def GetLineSearch(self, reg, n=None):
        """Find lines that start with a given regular expression
        
        :Call:
            >>> lines = FC.GetLineSearch(reg)
            >>> lines = FC.GetLineSearch(reg, n)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *lines*: :class:`list` (:class:`str`)
                List of lines that match pattern
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Set the update status.
        self.UpdateLines()
        # Initialize matches
        lines = []
        # Number of matches
        m = 0
        # Loop through the lines.
        for L in self.lines:
            # Check for maximum matches.
            if n and (m >= n):
                break
            # Check for a match.
            if re.search(reg, L):
                # Add to match list
                lines.append(L)
                # Increase count
                m += 1
        # Done
        return lines
        
    # Get a line that starts with a literal
    def GetLineInSectionStartsWith(self, sec, start, n=None):
        """Find lines in a given section that start specified target
        
        :Call:
            >>> lines = FC.GetLineInSectionStartsWith(sec, start)
            >>> lines = FC.GetLineInSectionStartsWith(sec, start, n)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *start*: :class:`str`
                Target line start
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *lines*: :class:`list` (:class:`str`)
                List of lines that match pattern
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Set the update status.
        self.UpdateSections()
        # Initialize matches
        lines = []
        # Number of matches
        m = 0
        # Check if the section exists.
        if sec not in self.SectionNames: return lines
        # Loop through the lines.
        for L in self.Section[sec]:
            # Check for maximum matches.
            if n and (m >= n):
                break
            # Check for a match.
            if L.startswith(start):
                # Add to match list
                lines.append(L)
                # Increase count
                m += 1
        # Done
        return lines
        
    # Get a line that starts with a literal
    def GetLineInSectionSearch(self, sec, reg, n=None):
        """Find lines in a given section that start specified regex
        
        :Call:
            >>> lines = FC.GetLineInSectionSearch(sec, reg)
            >>> lines = FC.GetLineInSectionSearch(sec, reg, n)
        :Inputs:
            *FC*: :class:`pyCart.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *lines*: :class:`list` (:class:`str`)
                List of lines that match pattern
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Set the update status.
        self.UpdateSections()
        # Initialize matches
        lines = []
        # Number of matches
        m = 0
        # Check if the section exists.
        if sec not in self.SectionNames: return lines
        # Loop through the lines.
        for L in self.Section[sec]:
            # Check for maximum matches.
            if n and (m >= n):
                break
            # Check for a match.
            if re.search(reg, L):
                # Add to match list
                lines.append(L)
                # Increase count
                m += 1
        # Done
        return lines
        
    # Get index of a line that starts with a literal
    def GetIndexStartsWith(self, start, n=None):
        r"""Find indices of lines that start with a given literal pattern
        
        :Call:
            >>> i = FC.GetIndexStartsWith(start)
            >>> i = FC.GetIndexStartsWith(start, n)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *start*: :class:`str`
                Line start target
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *i*: :class:`list`\ [:class:`int`]
                List of lines that match pattern
        :Versions:
            * 2015-02-28 ``@ddalle``: First version
        """
        # Set the update status.
        self.UpdateLines()
        # Initialize matches
        i = []
        # Number of matches
        m = 0
        # Loop through the lines.
        for k in range(len(self.lines)):
            # Check for maximum matches.
            if n and (m >= n):
                break
            # Check for a match.
            if self.lines[k].startswith(start):
                # Add to match list
                i.append(k)
                # Increase count
                m += 1
        # Done
        return i
        
    # Get index of a line that starts with a literal
    def GetIndexSearch(self, reg, n=None):
        r"""Find lines that start with a given regular expression
        
        :Call:
            >>> i = FC.GetIndexSearch(reg)
            >>> i = FC.GetIndexSearch(reg, n)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *i*: :class:`list`\ [:class:`int`]
                List of lines that match pattern
        :Versions:
            * 2014-02-28 ``@ddalle``: First version
        """
        # Set the update status.
        self.UpdateLines()
        # Initialize matches
        i = []
        # Number of matches
        m = 0
        # Loop through the lines.
        for k in range(len(self.lines)):
            # Check for maximum matches.
            if n and (m >= n):
                break
            # Check for a match.
            if re.search(reg, self.lines[k]):
                # Add to match list
                i.append(k)
                # Increase count
                m += 1
        # Done
        return i
        
    # Get index a line that starts with a literal
    def GetIndexInSectionStartsWith(self, sec, start, n=None):
        """Find lines in a given section with given start string
        
        :Call:
            >>> i = FC.GetIndexInSectionStartsWith(sec, start)
            >>> i = FC.GetIndexInSectionStartsWith(sec, start, n)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *start*: :class:`str`
                Line start target
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *i*: :class:`list` (:class:`int`)
                List of indices of lines in section that match pattern
        :Versions:
            * 2014-02-28 ``@ddalle``: First version
        """
        # Set the update status.
        self.UpdateSections()
        # Initialize matches
        i = []
        # Number of matches
        m = 0
        # Check if the section exists.
        if sec not in self.SectionNames: return i
        # Loop through the lines.
        for k in range(len(self.Section[sec])):
            # Check for maximum matches.
            if n and (m >= n):
                break
            # Check for a match.
            if self.Section[sec][k].startswith(start):
                # Add to match list
                i.append(k)
                # Increase count
                m += 1
        # Done
        return i
        
    # Get index of a line that starts with a literal
    def GetIndexInSectionSearch(self, sec, reg, n=None):
        """Find lines in a given section that start with a regex
        
        :Call:
            >>> i = FC.GetIndexInSectionSearch(sec, reg)
            >>> i = FC.GetIndexInSectionSearch(sec, reg, n)
        :Inputs:
            *FC*: :class:`cape.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *i*: :class:`list` (:class:`int`)
                List of indices of lines in section that match pattern
        :Versions:
            * 2014-02-28 ``@ddalle``: First version
        """
        # Set the update status.
        self.UpdateSections()
        # Initialize matches
        i = []
        # Number of matches
        m = 0
        # Check if the section exists.
        if sec not in self.SectionNames: return i
        # Loop through the lines.
        for k in range(len(self.Section[sec])):
            # Check for maximum matches.
            if n and (m >= n):
                break
            # Check for a match.
            if re.search(reg, self.Section[sec][k]):
                # Add to match list
                i.append(k)
                # Increase count
                m += 1
        # Done
        return i

