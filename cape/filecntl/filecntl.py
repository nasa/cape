r"""
This provides common methods to control objects for various specific
files.  This includes various methods for reading files, splitting it
into sections, and replacing lines based on patterns or regular
expressions.

File manipulation classes for specific files (such as
:class:`pyCart.inputCntl.InputCntl` for Cart3D ``input.cntl`` files) are built
off of this module and its main class. A possibly incomplete list of class
built on this class is given below.

    * :class:`cape.pycart.inputCntl.InputCntl`
    * :class:`cape.filecntl.namelist.Namelist`
    * :class:`cape.filecntl.namelist2.Namelist2`
    * :class:`cape.tex.Tex`
    * :class:`cape.pyfun.namelist.Namelist`
    * :class:`cape.pyfun.mapbc.MapBC`
    * :class:`caep.pyover.overNamelist.OverNamelist`

"""

# Standard library
import os
import re


# Regular expression for integers
REGEX_INT = re.compile(r"[+-]?[0-9]+")
# Regular expression for loats
_re1 = r"[0-9]+\.?[0-9]*"
_re2 = r"[0-9]*\.[0-9]+"
REGEX_FLOAT = re.compile(rf"[+-]?({_re1}|{_re2})([EDed][+-]?[0-9]+)?")


# Minor function to convert strings to numbers
def _float(s: str):
    r"""Convert string to float when possible

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
        * 2014-06-10 ``@ddalle``: v1.0
        * 2023-12-29 ``@ddalle``: v2.0; use regex
    """
    # Check string
    if REGEX_FLOAT.fullmatch(s):
        # Convert
        return float(s.replace("D", "e").replace("d", "e"))
    # Otherwise return the string
    return s


# Minor function to convert strings to numbers
def _int(s: str):
    r"""Convert string to integer when possible

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
        * 2014-06-10 ``@ddalle``: v1.0
        * 2023-12-29 ``@ddalle``: v2.0; use regex
    """
    # Check string
    if REGEX_INT.fullmatch(s):
        # Convert
        return int(s)
    # Otherwise return original string
    return s


# Minor function to convert strings to numbers
def _num(s: str):
    r"""Convert string to numeric value when possible

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
        * 2014-06-10 ``@ddalle``: v1.0
        * 2023-12-29 ``@ddalle``: v2.0; use regex
    """
    # Check string
    if REGEX_INT.fullmatch(s):
        # Convertible to int
        return int(s)
    elif REGEX_FLOAT.fullmatch(s):
        # Convertible to float
        return float(s.replace("D", "e").replace("d", "e"))
    # Otherwise return string
    return s


# Convert string -> list but leave list alone
def _listify(str_or_list) -> list:
    # Check type
    if isinstance(str_or_list, (list, tuple)):
        # Return a copy
        return list(str_or_list)
    else:
        # Return singleton list
        return [str_or_list]


# File control class
class FileCntl(object):
    r"""Base file control class

    The lines of the file can be split into sections based on a regular
    expression (see :func:`cape.filecntl.FileCntl.SplitToSections`);
    most methods will keep the overall line list and the section
    breakout consistent.

    :Call:
        >>> fc = FileCntl(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read from and manipulate
    :Data members:
        *fc.fname*: :class:`str`
            Name of file instance was read from
        *fc.lines*: :class:`list`\ [:class:`str`]
            List of all lines in the file (to use for replacement)
        *fc.SectionNames*: :class:`list`\ [:class:`str`]
            List of section titles if present
        *fc.Section*: :class:`dict` (:class:`list`\ [:class:`str`])
            Dictionary of the lines in each section, if present
    """

    # Initialization method; not useful for derived classes
    def __init__(self, fname=None):
        # Read the file
        self.Read(fname)
        # Save the file name
        self.fname = fname

    # Display method
    def __repr__(self):
        r"""Display method for file control class

        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2023-12-29 ``@ddalle``: v1.1; support empty fname
        """

        # Initialize the string
        s = f"<{self.__class__.__name__}("
        # Add name of file (if any)
        if self.fname is not None:
            s += f"'{self.fname}'"
        # Add number of lines
        s += f")[{len(self.lines)} lines"
        # Check for sections
        if len(self.SectionNames):
            s += f", {len(self.SectionNames)} sections"
        # End the string
        s += "]>"
        # Output
        return s

    # Read the file.
    def Read(self, fname: str):
        r"""Read text from file

        :Call:
            >>> fc.Read(fname)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *fname*: :class:`str`
                Name of file to read from
        :Effects:
            *fc.lines*: :class:`list`
                List of lines in file is created
            *fc._updated_sections*: :class:`bool`
                Whether section breakouts have been updated
            *fc._updated_lines*: :class:`bool`
                Flag for update status of global lines
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Read the file.
        if fname is None or not os.path.isfile(fname):
            # No file: initialize empty content
            self.lines = []
        else:
            # Open the file and read the lines
            self.lines = open(fname).readlines()
        # Initialize sections
        self.Section = {}
        self.SectionNames = []
        # Initialize update statuses
        self._updated_sections = False
        self._updated_lines = False
        # Section regex
        self._section_regex = None

    # Function to split into sections
    def SplitToSections(self, reg=r"\$__([\w_]+)", ngr=1, begin=True):
        r"""Split into sections based on starting regular expression

        :Call:
            >>> fc.SplitToSections()
            >>> fc.SplitToSections(reg="\$__([\w_]+)", ngr=1, **kw)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *reg*: {``"\$__([\w_]+)"``} | :class:`str`
                Regular expression for recognizing the start of a new
                section. By default this looks for sections that start
                with ``"$__"`` as in Cart3D ``input.cntl`` files. The
                regular expression must also include a group (meaning
                content between parentheses) to capture the *name* of
                the section. Thus the default value of
                ``"\$__([\w_]+)"`` finds any name that consists of word
                characters and/or underscores.
            *ngr*: {``1``} | :class:`int` | :class:`str`
                Group number from which to take name of section. This
                is always ``1`` unless the section-starting regular
                expression has more than one explicit group. Note that
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
        :Effects:
            *fc.SectionNames*: :class:`list`
                List of section names is created (includes "_header")
            *fc.Section*: :class:`dict`
                Dictionary of section line lists is created
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2024-01-02 ``@ddalle``: v1.1; save regex used
        """
        # Initial section name
        sec = "_header"
        # Initialize the sections.
        self.SectionNames = [sec]
        self.Section = {sec: []}
        # Save regular expression
        self._section_regex = reg
        # Compile regular expression
        regexa = re.compile(reg)
        # Loop through lines
        for line in self.lines:
            # Search from beginning of line or anywhere in line
            matchfunc = regexa.match if begin else regexa.search
            # Search for the new-section regular expression
            m = matchfunc(line.strip())
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
            >>> fc.SplitToBlocks()
            >>> fc.SplitToBlocks(reg="\$__([\w_]+)", ngr=1, **kw)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
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
                Group number from which to take name of section. This
                is always ``1`` unless the section-starting regular
                expression has more than one explicit group. Note that
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
            *fc.SectionNames*: :class:`list`
                List of section names is created (includes "_header")
            *fc.Section*: :class:`dict`
                Dictionary of section line lists is created
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2024-01-02 ``@ddalle``: v1.1; updates after testing
        """
        # Process inputs
        ngrb = kw.pop("endngr", ngr)
        begin = kw.pop("begin", True)
        endreg = kw.pop("endreg", None)
        endbeg = kw.pop("endbegin", begin)
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
            matchfuncb = None
        else:
            # Compile end-of-section regular expression
            regexb = re.compile(endreg)
            # Search from beginning of line or anywhere in line
            matchfuncb = regexb.match if endbeg else regexb.search
        # Search from beginning of line or anywhere in line
        matchfunca = regexa.match if begin else regexa.search
        # Loop through the lines.
        for line in self.lines:
            # Search for the new-section regular expression
            m = matchfunca(line.strip())
            # Check if there was a match.
            if m is None:
                # Append the line to the current section.
                self.Section[sec].append(line)
            else:
                # Get the new section name
                grp = m.group(ngr)
                # Very special check for formats with section ends
                if (regexb is None) or (grp != sec):
                    # Check for empty section
                    if len(self.Section[sec]) == 0:
                        # Remove it
                        self.Section.pop(sec)
                        self.SectionNames.remove(sec)
                    # Start the new section
                    sec = grp
                    self.SectionNames.append(sec)
                    self.Section[sec] = [line]
                    # Do not allow begin section to also end section
                    continue
                else:
                    # This is still part of previous section
                    self.Section[sec].append(line)
            # Check for end-of-section check
            if regexb is None:
                continue
            # Check the end-of-section regex
            m = matchfuncb(line.strip())
            # Check if there was a match
            if m is None:
                continue
            # Try to check the section name
            try:
                # Get group name
                grp = m.group(ngrb)
                # Check name
                if sec != grp:
                    raise ValueError(
                        f"Section '{sec}' of file '{self.fname}' " +
                        f"ends with '{grp}'")
            except IndexError:
                # End-of-section marker probably doesn't have group
                pass
            # Move to next intermediate section
            nint += 1
            sec = "_inter%i" % nint
            # Start the section
            self.SectionNames.append(sec)
            self.Section[sec] = []
        # Check for empty final section
        if len(self.Section[sec]) == 0:
            # Remove it
            self.Section.pop(sec)
            self.SectionNames.remove(sec)

    # Function to update the text based on the section content.
    def UpdateLines(self):
        r"""Update the global file text from current section content

        :Call:
            >>> fc.UpdateLines()
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
        :Effects:
            *fc.lines*: :class:`list`
                Lines are rewritten to match the sequence of lines from
                the sections
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Check for lines
        if not self._updated_sections:
            # No updates
            return
        # Reinitialize the lines
        self.lines = []
        # Loop through the sections
        for sec in self.SectionNames:
            # Join the lines in that section.
            self.lines.extend(self.Section[sec])
        # The lines are now up-to-date
        self._updated_sections = False

    # Function to update the text based on the section content.
    def UpdateSections(self):
        r"""Remake the section split if necessary

        This runs :func:`SplitToSections()` is run if
        *fc._updated_lines* is ``True``.

        :Call:
            >>> fc.UpdateSections()
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2024-01-02 ``@ddalle``: v1.1; fix for generic sec regex
        """
        # Check for lines
        if not self._updated_lines:
            # No updates
            return
        # Redo the split
        self.SplitToSections(reg=self._section_regex)
        self._updated_lines = False

    # Method to ensure that an instance has a certain section
    def AssertSection(self, sec: str):
        r"""Assert that a certain section is present

        :Call:
            >>> fc.AssertSection(sec)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance, defaults to *fc.fname*
            *sec*: :class:`str`
                Name of section to check for
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Update sections.
        self.UpdateSections()
        # Check for the section.
        if sec not in self.SectionNames:
            raise KeyError(
                "File control instance does not have section '%s'" % sec)

    # Method to write the file.
    def _Write(self, fname=None):
        r"""Write to text file

        :Call:
            >>> fc._Write()
            >>> fc._Write(fname)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance, defaults to *fc.fname*
            *fname*: :class:`str`
                Name of file to write to
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Update the lines if appropriate
        self.UpdateLines()
        # Default file name
        if fname is None:
            fname = self.fname
        # Open the new file
        with open(fname, 'w') as fp:
            # Write the joined text.
            fp.write("".join(self.lines))

    # Method to write the file.
    def Write(self, fname=None):
        r"""Write to text file

        :Call:
            >>> fc.Write()
            >>> fc.Write(fname)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance, defaults to *fc.fname*
            *fname*: :class:`str`
                Name of file to write to
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2015-11-16 ``@ddalle``: v1.1; use :func:`_Write`
        """
        # Update the lines if appropriate.
        self._Write(fname)

    # Method to write the file as an executable.
    def WriteEx(self, fname=None):
        r"""Write to text file as an executable script

        :Call:
            >>> fc.WriteEx()
            >>> fc.WriteEx(fname)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance, defaults to *fc.fname*
            *fname*: :class:`str`
                Name of file to write to
        :Versions:
            * 2014-06-23 ``@ddalle``: v1.0
            * 2024-01-02 ``@ddalle``: v1.1
                - remove two 'if' statements using bit-shift
                - test if Windows
        """
        # Default file name
        if fname is None:
            fname = self.fname
        # Write the file
        self._Write(fname)
        # No effect in windows
        if os.name != "posix":
            return  # pragma no cover
        # Get the mode of the file
        fmod = os.stat(fname).st_mode
        # Make sure the user-executable bit is set
        fmod = fmod | 0o100
        # If (group|others) readable, also make executable
        fmod = fmod | ((fmod & 0o040) >> 2)
        fmod = fmod | ((fmod & 0o004) >> 2)
        # Change the mode
        os.chmod(fname, fmod)

    # Method to insert a line somewhere
    def InsertLine(self, i: int, line: str):
        r"""Insert a line of text somewhere into the text

        :Call:
            >>> fc.InsertLine(i, line)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *i*: :class:`int`
                Index to which to insert the line
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Set the update flags.
        self.UpdateLines()
        self._updated_lines = True
        # Insert the line
        _insert_line(self.lines, line, i)

    # Method to append a line
    def AppendLine(self, line: str):
        r"""Append a line of text to *fc.lines*

        :Call:
            >>> fc.AppendLine(line)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Set the update flag.
        self.UpdateLines()
        self._updated_lines = True
        # Add the line
        _insert_line(self.lines, line)

    # Method to append a line
    def PrependLine(self, line: str):
        r"""Prepend a line of text to *fc.lines*

        :Call:
            >>> fc.PrependLine(line)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Set the update flag.
        self.UpdateLines()
        self._updated_lines = True
        # Insert the line. at beginning
        _insert_line(self.lines, line, 0)

    # Method to insert a line somewhere
    def InsertLineToSection(self, sec: str, i: int, line: str):
        r"""Insert a line of text somewhere into the text of a section

        :Call:
            >>> fc.InsertLineToSection(sec, i, line)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to update
            *i*: :class:`int`
                Index to which to insert the line
            *line*: :class:`str`
                String to add
        :Effects:
            A line is inserted to *fc.Section[sec]*
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Set the update flags.
        self.UpdateSections()
        self._updated_sections = True
        # Check for the section
        self.AssertSection(sec)
        # Insert the line
        _insert_line(self.Section[sec], line, i)

    # Method to append a line somewhere
    def AppendLineToSection(self, sec: str, line: str):
        r"""Append a line of text to a section

        :Call:
            >>> fc.AppendLineToSection(sec, line)
        :Inputs:
            *fc*: :class:`pyCart.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to update
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Set the update flags.
        self.UpdateSections()
        self._updated_sections = True
        # Check for the section
        self.AssertSection(sec)
        # Insert the line
        _insert_line(self.Section[sec], line)

    # Method to prepend a line somewhere
    def PrependLineToSection(self, sec: str, line: str):
        r"""Prepend a line of text to a section

        :Call:
            >>> fc.PrependLineToSection(sec, line)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to update
            *line*: :class:`str`
                String to add
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
        """
        # Set the update flags.
        self.UpdateSections()
        self._updated_sections = True
        # Check for the section
        self.AssertSection(sec)
        # Insert the line
        _insert_line(self.Section[sec], line, 0)

    # Method to delete a line that starts with a certain literal
    def DeleteLineStartsWith(self, start: str, imin=0, count=1) -> int:
        r"""Delete lines that start with given text up to *count* times

        :Call:
            >>> n = fc.DeleteLineStartsWith(start)
            >>> n = fc.DeleteLineStartsWith(start, count)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *start*: :class:`str`
                Line-starting string to search for
            *imin*: {``0``} | :class:`int`
                Index of first line from which to start search
            *count*: {``1``} | :class:`int`
                Maximum number of lines to delete
        :Outputs:
            *n*: :class:`int`
                Number of deletions made
        :Effects:
            Lines in *fc.lines* that start with *start* are removed
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2023-12-29 ``@ddalle``: v2.0; use _delete_line()
        """
        # Update the text
        self.UpdateLines()
        # Apply generic function
        n = _delete_line_startswith(self.lines, start, imin, count)
        # Mark line updated
        self._updated_lines = True
        # Output
        return n

    # Method to delete a line from a section that starts with a certain literal
    def DeleteLineInSectionStartsWith(
            self, sec: str, start: str, imin=0, count=1) -> int:
        r"""Delete lines based on start text and section name

        :Call:
            >>> n = fc.DeleteLineInSectionStartsWith(sec, start)
            >>> n = fc.DeleteLineInSectionStartsWith(sec, start, count)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search
            *start*: :class:`str`
                Line-starting string to search for
            *imin*: {``0``} | :class:`int`
                Index of first line from which to start search
            *count*: {``1``} | :class:`int`
                Maximum number of lines to delete
        :Outputs:
            *n*: :class:`int`
                Number of deletions made
        :Effects:
            Lines in *fc.Section[sec]* may be removed if they start with
            *start*.
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2023-12-29 ``@ddalle``: v2.0; use _delete_line()
        """
        # Initialize the deletion count
        n = 0
        # Update the sections
        self.UpdateSections()
        # Check for the section
        if sec not in self.SectionNames:
            return n
        # Apply generic function
        n = _delete_line_startswith(self.Section[sec], start, imin, count)
        # Mark line updated
        self._updated_sections = True
        # Output
        return n

    # Method to delete a line that contains regular expression
    def DeleteLineSearch(self, reg: str, imin=0, count=1) -> int:
        r"""Delete lines that start with given text up to *count* times

        :Call:
            >>> n = fc.DeleteLineSearch(reg, imin=0, count=1)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *reg*: :class:`str`
                Regular expression search for
            *imin*: {``0``} | :class:`int`
                Index of first line from which to start search
            *count*: {``1``} | :class:`int`
                Maximum number of lines to delete
        :Outputs:
            *n*: :class:`int`
                Number of deletions made
        :Effects:
            Lines in *fc.lines* that match *reg* are removed
        :Versions:
            * 2023-12-30 ``@ddalle``: v1.0
        """
        # Update the text
        self.UpdateLines()
        # Apply generic function
        n = _delete_line_search(self.lines, reg, imin, count)
        # Mark line updated
        self._updated_lines = True
        # Output
        return n

    # Method to delete a line in a section that contains regex
    def DeleteLineInSectionSearch(
            self, sec: str, reg: str, imin=0, count=1) -> int:
        r"""Delete lines that start with given text up to *count* times

        :Call:
            >>> n = fc.DeleteLineInSectionSearch(sec, reg, **kw)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search
            *reg*: :class:`str`
                Regular expression search for
            *imin*: {``0``} | :class:`int`
                Index of first line from which to start search
            *count*: {``1``} | :class:`int`
                Maximum number of lines to delete
        :Outputs:
            *n*: :class:`int`
                Number of deletions made
        :Effects:
            Lines in *fc.lines* that match *reg* are removed
        :Versions:
            * 2023-12-30 ``@ddalle``: v1.0
        """
        # Initialize the deletion count
        n = 0
        # Update the sections
        self.UpdateSections()
        # Check for the section
        if sec not in self.SectionNames:
            return n
        # Apply generic function
        n = _delete_line_search(self.Section[sec], reg, imin, count)
        # Mark line updated
        self._updated_sections = True
        # Output
        return n

    # Method to replace a line that starts with a given string
    def ReplaceLineStartsWith(
            self, start: str, line: str, imin=0, nmax=None) -> int:
        r"""Replace lines starting with fixed text

        Find all lines that begin with a certain string and replace them
        with another string.  Note that the entire line is replaced, not
        just the initial string.

        Leading spaces are ignored during the match tests.

        :Call:
            >>> n = fc.ReplaceLineStartsWith(start, line, imin=0, **kw)
            >>> n = fc.ReplaceLineStartsWith(start, lines, imin=0, **kw)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
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
            *fc.lines*: Some of the lines may be affected
            *fc._updated_lines*: Set to ``True``
        :Examples:
            Suppose that *fc* has the following two lines.

                ``Mach      8.00   # some comment\n``

                ``Mach      Mach_TMP\n``

            Then this example will replace *both* lines with the string
            ``Mach 4.0``

                >>> fc.ReplaceLineStartsWith('Mach', 'Mach 4.0')

            This example replaces each line with a different value for
            the Mach number.

                >>> fc.ReplaceLineStartsWith(
                    'Mach', ['Mach 2.0', 'Mach 4.0']

            Finally, this example is different from the first example in
            that it will replace the first line and then quit before it
            can find the second match.

                >>> fc.ReplaceLineStartsWith('Mach', ['Mach 4.0'])

        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2019-06-19 ``@ddalle``: v1.1; add *imin* and *nmax*
            * 2023-12-26 ``@ddalle``: v1.2; fix output w/ *imin*
            * 2023-12-29 ``@ddalle``: v2.0; use _replace_line()
        """
        # Set the update status
        self.UpdateLines()
        self._updated_lines = True
        # Call standalone function
        n = _replace_line_startswith(self.lines, line, start, imin, nmax)
        # Output
        return n

    # Method to replace a line only in a certain section
    def ReplaceLineInSectionStartsWith(
            self, sec: str, start: str, line, imin=0, nmax=None) -> int:
        r"""Make replacements within section based on starting string

        Find all lines in a certain section that start with a specified
        literal string and replace the entire line with the specified text.

        :Call:
            >>> n = fc.ReplaceLineInSectionStartsWith(
                , start, line, **kw)
            >>> n = fc.ReplaceLineInSectionStartsWith(
                sec, start, lines, **kw)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
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
            Some lines in *fc.Section[sec]* may be replaced.
        :See also:
            This function is similar to
            :func:`cape.filecntl.FileCntl.ReplaceLineStartsWith` except
            that the search is restricted to a specified section.
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2023-12-26 ``@ddalle``: v1.1; reduce lines; fix *imin*
            * 2023-12-29 ``@ddalle``: v2.0; use _replace_line()
        """
        # Number of matches
        n = 0
        # Update the sections
        self.UpdateSections()
        self._updated_sections = True
        # Check if the section exists
        if sec not in self.SectionNames:
            return n
        # Use generic function
        n = _replace_line_startswith(
            self.Section[sec], line, start, imin, nmax)
        # Output
        return n

    # Method to replace a line that starts with a regular expression
    def ReplaceLineSearch(self, reg: str, line, imin=0, nmax=None) -> int:
        r"""Replace lines based on initial regular expression

        Find all lines that begin with a certain regular expression and
        replace them with another string.  Note that the entire line is
        replaced, not just the regular expression.

        Leading spaces are ignored during the match tests.

        :Call:
            >>> n = fc.ReplaceLineSearch(reg, line, imin=0, nmax=None)
            >>> n = fc.ReplaceLineSearch(reg, lines, imin=0, nmax=None)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
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
            *fc.lines*: Some of the lines may be affected
            *fc._updated_lines*: Set to ``True``
        :Examples:
            Suppose that *fc* has the following two lines.

                ``Mach      8.00   # some comment\n``

                ``Mach    4\n``

            Then this example will replace *both* lines with
            ``Mach 2.0``

                >>> fc.ReplaceLineSearch('Mach\s+[0-9.]+', 'Mach 2.0')

            This example replaces each line with a different value for
            the Mach number.

                >>> fc.ReplaceLineSearch('Mach\s+[0-9.]+',
                    ['Mach 2.0', 'Mach 2.5'])

            Finally, this example is different from the first example in
            that it will replace the first line and then quit before it
            can find the second match.

                >>> fc.ReplaceLineSearch('Mach\s+[0-9.]+', ['Mach 2.0'])

        :Versions:
            * 2014-06-04 ``@ddalle``: v1.0
            * 2023-12-29 ``@ddalle``: v2.0; use _replace_line()
        """
        # Set the update status
        self.UpdateLines()
        self._updated_lines = True
        # Call standalone function
        n = _replace_line_search(self.lines, line, reg, imin, nmax)
        # Output
        return n

    # Method to replace a line only in a certain section
    def ReplaceLineInSectionSearch(
            self, sec: str, reg: str, line, imin=0, nmax=None) -> int:
        r"""
        Find all lines in a certain section that start with a specified regular
        expression and replace the entire lines with the specified text.

        :Call:
            >>> n = fc.ReplaceLineInSectionSearch(sec, reg, line, **kw)
            >>> n = fc.ReplaceLineInSectionSearch(sec, reg, lines, **kw)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
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
            Some lines in *fc.Section[sec]* may be replaced.
        :See also:
            This function is similar to
            :func:`pyCart.fileCntl.FileCntl.ReplaceLineSearch` except that
            the search is restricted to a specified section.
        :Versions:
            * 2014-06-04 ``@ddalle``: v1.0
            * 2023-12-29 ``@ddalle``: v2.0; use _replace_line()
        """
        # Number of matches.
        n = 0
        # Update the sections.
        self.UpdateSections()
        # Set the update status.
        self._updated_sections = True
        # Check if the section exists.
        if sec not in self.SectionNames:
            return n
        # Call standalone function
        n = _replace_line_search(self.Section[sec], line, reg, imin, nmax)
        # Output
        return n

    # Replace a line or add it if not found
    def ReplaceOrAddLineStartsWith(self, start: str, line: str, i=None, **kw):
        r"""Replace a line or add a new one

        Replace a line that starts with a given literal string or add
        the line if no matches are found.

        :Call:
            >>> fc.ReplaceOrAddLineStartsWith(start, line, **kw)
            >>> fc.ReplaceOrAddLineStartsWith(start, line, i, **kw)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
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
            Replaces line in section *fc.lines* or adds it if not found
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2023-12-28 ``@ddalle``: v1.1; use _insert_line()
        """
        # Call the replace method (only perform once)
        n = self.ReplaceLineStartsWith(start, [line], **kw)
        # Check for a match
        if n:
            return
        # Append/insert the line
        _insert_line(self.lines, line, i)

    # Replace a line or add (from one section) if not found
    def ReplaceOrAddLineToSectionStartsWith(
            self, sec: str, start: str, line: str, i=None, **kw):
        r"""Replace a line or add a new one (within section)

        Replace a line in a specified section that starts with a given
        literal  string or add the line to the section if no matches
        are found.

        :Call:
            >>> fc.ReplaceOrAddLineToSectionStartsWith(sec, start, line)
            >>> fc.ReplaceOrAddLineToSectionStartsWith(
                sec, start, line, i=None, **kw)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
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
            Replaces line in *fc.Section[sec]* or adds it if not found
        :Versions:
            * 2014-06-03 ``@ddalle``: v1.0
            * 2023-12-28 ``@ddalle``: v1.1; use _insert_line()
        """
        # Call the replace method (only perform once)
        n = self.ReplaceLineInSectionStartsWith(sec, start, [line], **kw)
        # Check if found
        if n:
            return
        # Must have the section
        self.AssertSection(sec)
        # Append/insert line
        _insert_line(self.Section[sec], line, i)

    # Replace a line or add it if not found
    def ReplaceOrAddLineSearch(self, reg: str, line: str, i=None, **kw):
        r"""Replace a line identified by regex, or add new line

        Replace a line that starts with a given regular expression or
        add the line if no matches are found.

        :Call:
            >>> fc.ReplaceOrAddLineSearch(reg, line, **kw)
            >>> fc.ReplaceOrAddLineSearch(reg, line, i, **kw)
        :Inputs:
            *fc*: :class:`pyCart.fileCntl.FileCntl`
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
            Replaces line in section *fc.lines* or adds it if not found
        :Versions:
            * 2014-06-04 ``@ddalle``: v1.0
            * 2023-12-30 ``@ddalle``: v1.1; use _insert_line()
        """
        # Call the replace method (only perform once).
        n = self.ReplaceLineSearch(reg, [line], **kw)
        # Check for a match
        if n:
            return
        # Append/insert the line
        _insert_line(self.lines, line, i)

    # Replace a line or add (from one section) if not found
    def ReplaceOrAddLineToSectionSearch(
            self, sec: str, reg: str, line: str, i=None, **kw):
        r"""Replace a line in a specified section

        Replace a line in a specified section that starts with a given
        regular  expression or add the line to the section if no matches
        are found.

        :Call:
            >>> fc.ReplaceOrAddLineToSectionStartsWith(sec, reg, line)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *line*: :class:`str`
                String to replace every match with
            *i*: {```None``} | :class:`int`
                Location to add line (by default it is appended)
            *imin*: {``0``} | :class:`int` >= 0
                Do not make replacements for matches with index < *imin*
            *nmax*: {``None``} | :class:`int` > 0
                Make at most *nmax* substitutions
        :Effects:
            Replaces line in *fc.Section[sec]* or adds it if not found
        :Versions:
            * 2014-06-04 ``@ddalle``: v1.0
        """
        # Call the replace method (only perform once).
        n = self.ReplaceLineInSectionSearch(sec, reg, [line], **kw)
        # Check if found
        if n:
            return
        # Must have the section
        self.AssertSection(sec)
        # Append/insert line
        _insert_line(self.Section[sec], line, i)

    # Get a line that starts with a literal
    def GetLineStartsWith(self, start, n=None):
        r"""Find lines that start with a given literal pattern

        :Call:
            >>> lines = fc.GetLineStartsWith(start)
            >>> lines = fc.GetLineStartsWith(start, n)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *start*: :class:`str`
                String to test as match for beginning of each line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                List of lines that match pattern
        :Versions:
            * 2014-06-10 ``@ddalle``: v1.0
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
        r"""Find lines that start with a given regular expression

        :Call:
            >>> lines = fc.GetLineSearch(reg)
            >>> lines = fc.GetLineSearch(reg, n)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                List of lines that match pattern
        :Versions:
            * 2014-06-10 ``@ddalle``: v1.0
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
        r"""Find lines in a given section that start specified target

        :Call:
            >>> lines = fc.GetLineInSectionStartsWith(sec, start)
            >>> lines = fc.GetLineInSectionStartsWith(sec, start, n)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *start*: :class:`str`
                Target line start
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                List of lines that match pattern
        :Versions:
            * 2014-06-10 ``@ddalle``: v1.0
        """
        # Set the update status.
        self.UpdateSections()
        # Initialize matches
        lines = []
        # Number of matches
        m = 0
        # Check if the section exists.
        if sec not in self.SectionNames:
            return lines
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
        r"""Find lines in a given section that start specified regex

        :Call:
            >>> lines = fc.GetLineInSectionSearch(sec, reg)
            >>> lines = fc.GetLineInSectionSearch(sec, reg, n)
        :Inputs:
            *fc*: :class:`pyCart.fileCntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                List of lines that match pattern
        :Versions:
            * 2014-06-10 ``@ddalle``: v1.0
        """
        # Set the update status.
        self.UpdateSections()
        # Initialize matches
        lines = []
        # Number of matches
        m = 0
        # Check if the section exists.
        if sec not in self.SectionNames:
            return lines
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
            >>> i = fc.GetIndexStartsWith(start)
            >>> i = fc.GetIndexStartsWith(start, n)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *start*: :class:`str`
                Line start target
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *i*: :class:`list`\ [:class:`int`]
                List of lines that match pattern
        :Versions:
            * 2015-02-28 ``@ddalle``: v1.0
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
            >>> i = fc.GetIndexSearch(reg)
            >>> i = fc.GetIndexSearch(reg, n)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *i*: :class:`list`\ [:class:`int`]
                List of lines that match pattern
        :Versions:
            * 2014-02-28 ``@ddalle``: v1.0
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
        r"""Find lines in a given section with given start string

        :Call:
            >>> i = fc.GetIndexInSectionStartsWith(sec, start)
            >>> i = fc.GetIndexInSectionStartsWith(sec, start, n)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *start*: :class:`str`
                Line start target
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *i*: :class:`list`\ [:class:`int`]
                List of indices of lines in section that match pattern
        :Versions:
            * 2014-02-28 ``@ddalle``: v1.0
        """
        # Set the update status.
        self.UpdateSections()
        # Initialize matches
        i = []
        # Number of matches
        m = 0
        # Check if the section exists.
        if sec not in self.SectionNames:
            return i
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
        r"""Find lines in a given section that start with a regex

        :Call:
            >>> i = fc.GetIndexInSectionSearch(sec, reg)
            >>> i = fc.GetIndexInSectionSearch(sec, reg, n)
        :Inputs:
            *fc*: :class:`cape.filecntl.FileCntl`
                File control instance
            *sec*: :class:`str`
                Name of section to search in
            *reg*: :class:`str`
                Regular expression to match beginning of line
            *n*: :class:`int`
                Maximum number of matches to search for
        :Outputs:
            *i*: :class:`list`\ [:class:`int`]
                List of indices of lines in section that match pattern
        :Versions:
            * 2014-02-28 ``@ddalle``: v1.0
        """
        # Set the update status.
        self.UpdateSections()
        # Initialize matches
        i = []
        # Number of matches
        m = 0
        # Check if the section exists.
        if sec not in self.SectionNames:
            return i
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


# Add line to list of lines
def _insert_line(lines: list, line: str, i=None):
    # Check where to add the line
    if i is None:
        # Append
        lines.append(line)
    else:
        # Correct for empty lines
        if i < 0:
            # Count empty lines at the end
            j = count_trailing_blanklines(lines)
            # Insert at specified location
            lines.insert(i - j, line)
        else:
            # Insert at specified location.
            lines.insert(i, line)


# Replace a line based on literal start
def _replace_line_startswith(
        lines: list, line, start: str, imin=0, nmax=None) -> int:
    # Create function
    func = _genr8_startswith(start)
    # Call general function
    return _replace_line(lines, line, func, imin, nmax)


def _replace_line_search(
        lines: list, line, reg: str, imin=0, nmax=None) -> int:
    # Create function
    func = _genr8_search(reg)
    # Call general function
    return _replace_line(lines, line, func, imin, nmax)


# Replace a line based on arbitrary test function
def _replace_line(lines: list, line, func, imin=0, nmax=None) -> int:
    # Ensure we have a list of lines
    replacements = _listify(line)
    # Number of matches
    n = 0
    # Loop through the lines
    for i in range(imin, len(lines)):
        # Get the line
        li = lines[i]
        # Check for a match
        if not func(li):
            continue
        # Replace the line based on the match count.
        lines[i] = replacements[n]
        # Increase the match count.
        n += 1
        # Check for end of matches.
        if n >= len(replacements) or (nmax and (n >= nmax)):
            break
    # Done
    return n


# Count empty lines at the end of a section
def count_leading_blanklines(lines: list) -> int:
    r"""Count empty lines at the start of a list of lines

    :Call:
        >>> n = count_leading_blanklines(lines)
    :Inputs:
        *lines*: :class:`list`\ [:class:`str`]
            List of lines in section or file
    :Outputs:
        *n*: :class:`int`
            Number of trailing empty lines
    :Versions:
        * 2016-04-18 ``@ddalle``: v1.0 (method of FileCntl)
        * 2023-12-23 ``@ddalle``: v1.1
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


# Count empty lines at the end of a section
def count_trailing_blanklines(lines: list) -> int:
    r"""Count empty lines at the end of a list of lines

    :Call:
        >>> n = count_trailing_blanklines(lines)
    :Inputs:
        *lines*: :class:`list`\ [:class:`str`]
            List of lines in section or file
    :Outputs:
        *n*: :class:`int`
            Number of trailing empty lines
    :Versions:
        * 2016-04-18 ``@ddalle``: v1.0 (method of FileCntl)
        * 2023-12-28 ``@ddalle``: v1.1
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


# Delete a line based on start
def _delete_line_search(lines: list, reg: str, imin=0, nmax=None) -> int:
    r"""Delete line(s) from a list based on regular expression

    :Call:
        >>> n = _delete_line_search(lines, reg, imin=0, nmax=None)
    :Inputs:
        *lines*: :class:`list`\ [:class:`str`]
            List of lines (from which to delete)
        *reg*: :class:`str`
            Regular expression string
        *imin*: {``0``} | :class:`int`
            Index of first line in *lines* to considere
        *nmax*: {``None``} | :class:`int`
            Maximum number of deletions to make
    :Outputs:
        *n*: :class:`int`
            Number of deletions
    """
    # Create function
    func = _genr8_search(reg)
    # Apply main function
    n = _delete_line(lines, func, imin, nmax)
    # Output
    return n


# Delete a line based on start
def _delete_line_startswith(lines: list, start: str, imin=0, nmax=None) -> int:
    r"""Delete line(s) from a list based on literal start text

    :Call:
        >>> n = _delete_line_startswith(lines, start, imin=0, nmax=None)
    :Inputs:
        *lines*: :class:`list`\ [:class:`str`]
            List of lines (from which to delete)
        *start*: :class:`str`
            Delete all lines starting with this text
        *imin*: {``0``} | :class:`int`
            Index of first line in *lines* to considere
        *nmax*: {``None``} | :class:`int`
            Maximum number of deletions to make
    :Outputs:
        *n*: :class:`int`
            Number of deletions
    """
    # Create function
    func = _genr8_startswith(start)
    # Apply main function
    n = _delete_line(lines, func, imin, nmax)
    # Output
    return n


# Delete a line
def _delete_line(lines: list, func, imin=0, nmax=None) -> int:
    r"""Delete line(s) from a list based on a test function

    :Call:
        >>> n = _delete_line(lines, func, imin=0, nmax=None)
    :Inputs:
        *lines*: :class:`list`\ [:class:`str`]
            List of lines (from which to delete)
        *func*: **callable** (:class:`str`,) -> :class:`bool`
            Function that takes in a single line and retunrs whether or
            not to delete the line
        *imin*: {``0``} | :class:`int`
            Index of first line in *lines* to considere
        *nmax*: {``None``} | :class:`int`
            Maximum number of deletions to make
    :Outputs:
        *n*: :class:`int`
            Number of deletions
    """
    # Line number
    i = imin
    # Initialize count
    n = 0
    # Loop backward through the lines
    while i < len(lines):
        # Get the line.
        li = lines[i]
        # Check it
        if func(li):
            # Increase the count
            n += 1
            # Delete the line
            lines.__delitem__(i)
            # Check for limit
            if nmax and (n >= nmax):
                return n
        else:
            # Increase line number
            i += 1
    # Done
    return n


def _genr8_startswith(start: str):
    r"""Create a function that tests if a line starts with *start*

    :Call:
        >>> func = _genr8_startswith(start)
    :Inputs:
        *start*: :class:`str`
            Text to test for at beginning of line
    :Outputs:
        *func*: **callablle** (:class:`str`,) -> :class:`bool`
            Function which tests if a string begins with *start*
    """
    # Create subfunction
    def func(line: str) -> bool:
        return line.startswith(start)
    # Return the subfunction
    return func


def _genr8_search(reg: str):
    r"""Create a function that tests contains regular expression *reg*

    :Call:
        >>> func = _genr8_search(reg)
    :Inputs:
        *reg*: :class:`str`
            Regular expression string
    :Outputs:
        *func*: **callablle** (:class:`str`,) -> :class:`bool`
            Function which tests if a string contains match for *reg*
    """
    # Compile the regular expression
    regex = re.compile(reg)

    # Create subfunction
    def func(line: str):
        # Search
        return regex.search(line)
    # Return the subfunction
    return func
