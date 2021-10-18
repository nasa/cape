#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.jobxml`: Interface to Kestrel main XML control file
=====================================================================

This module provides the class :class:`JobXML`, which reads, edits, and
writes the main XML file that sets the main inputs for Kestrel jobs.

"""

# Standard library
import sys

# Third-party


# Local imports
from .. import xmlfile


# Faulty *unicode* type for Python 3
if sys.version_info.major > 2:
    unicode = str


# Main class
class JobXML(xmlfile.XMLFile):
    r"""Interface to XML files

    :Call:
        >>> xml = JobXML(fxml)
        >>> xml = JobXML(xml1)
        >>> xml = JobXML(et)
        >>> xml = JobXML(e)
        >>> xml = JobXML(txt)
        >>> xml = JobXML()
    :Inputs:
        *fxml*: :class:`str`
            Name of an XML file
        *et*: :class:`xml.etree.ElementTree.ElementTree`
            An XML element tree
        *e*: :class:`xml.etree.ElementTree.Element`
            An XML root element
        *txt*: :class:`str`
            XML text to parse directly
        *xml1*: :class:`XMLFile`
            Another instance (of parent class)
    :Outputs:
        *xml*: :class:`JobXML`
            Instance of Kestrel job XML file interface
    :Attributes:
        *xml.tree*: :class:`xml.etree.ElementTree.ElementTree`
            An XML element tree interface to contents
        *xml.root*: :class:`xml.etree.ElementTree.Element`
            The root element of the XML element tree
        *xml.fname*: ``None`` | :class:`str`
            Name of file read or default file name to write
    :Versions:
        * 2021-10-18 ``@ddalle``: Version 0.0: Started
    """
   # --- __dunder__ ---
   # --- Specific values: get ---
    def get_mach(self):
        return self.get_input("Mach")

    def get_alpha(self):
        return self.get_input("Alpha")

    def get_beta(self):
        return self.get_input("Beta")

    def get_restart(self):
        return self.get_input("Restart")

    def get_kcfd_iters(self):
        return self.get_kcfd("Iterations")

    def get_kcfd_subiters(self):
        return self.get_kcfd("Subiterations")

    def get_kcfd_timestep(self):
        return self.get_kcfd("TimeStep")

   # --- Specific values: set ---
    def set_mach(self, mach):
        return self.set_input("Mach", mach)

    def set_alpha(self, alpha):
        return self.set_input("Alpha", alpha)

    def set_beta(self, beta):
        return self.set_input("Beta", beta)

    def set_restart(self, restart=True):
        return self.set_input("Restart", restart)

    def set_kcfd_iters(self, iters):
        return self.set_kcfd("Iterations", iters)

    def set_kcfd_subiters(self, subiters):
        return self.set_kcfd("Subiterations", subiters)

    def set_kcfd_timestep(self, timestep):
        return self.set_kcfd("TimeStep", timestep)
            
   # --- Sections: find ---
    def find_input(self, name):
        r"""Get an *InputList.Input* XML element by *name* attrib

        :Call:
            >>> elem = xml.find_input(name)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *name*: :class:`str`
                Name of input attribute to query
        :Outputs:
            *elem*: ``None`` | :class:`Element`
                Element matching all criteria
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        return self.find("InputList.Input", attrib={"name": name})

    def find_kcfd(self, tag):
        r"""Find an XML element from the *KCFD* settings

        :Call:
            >>> elem = xml.find_kcfd(tag)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *tag*: :class:`str`
                Element tag in *KCFD* parent
        :Outputs:
            *elem*: ``None`` | :class:`Element`
                Element matching all criteria
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        # Full path to section
        tags = [
            "BodyHierarchy",
            "ActionList",
            "FVMCFD",
            tag
        ]
        # Constraints at each level
        attribs = [
            None,
            None,
            {"name": "KCFD"}
        ]
        # Find the element
        return self.find(tags, attribs=attribs)

   # --- Sections: set value --- 
    def set_input(self, name, v):
        r"""Set the text of an *InputList.Input* element

        :Call:
            >>> xml.set_input(name, v)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *name*: :class:`str`
                Name of input attribute to query
            *v*: ``None`` | **any**
                Python value to save to element as text
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        self.set_elem("InputList.Input", v, attrib={"name": name})

    def set_kcfd(self, tag, v):
        r"""Set the text of a *KCFD* element

        :Call:
            >>> xml.set_kcfd(tag, v)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *tag*: :class:`str`
                Element tag in *KCFD* parent
            *v*: ``None`` | **any**
                Python value to save to element as text
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        # Full path to section
        tags = [
            "BodyHierarchy",
            "ActionList",
            "FVMCFD",
            tag
        ]
        # Constraints at each level
        attribs = [
            None,
            None,
            {"name": "KCFD"}
        ]
        # Edit or add requested element
        self.set_elem(tags, v, attribs=attribs)

   # --- Sections: get value ---
    def get_input(self, name):
        r"""Get the converted text of an *InputList.Input* element

        :Call:
            >>> v = xml.get_input(name)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *name*: :class:`str`
                Name of input attribute to query
        :Outputs:
            *v*: ``None`` | **any**
                (Converted) text of *InputList.Input* element with
                attribute *name* matching
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        # Get the element
        elem = self.find_input(name)
        # Check if found
        if elem is None:
            return
        # Convert *text* to value
        return self.text2val(elem.text)

    def get_kcfd(self, tag):
        r"""Get converted text from the *KCFD* settings

        :Call:
            >>> v = xml.get_kcfd(tag)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *tag*: :class:`str`
                Element tag in *KCFD* parent
        :Outputs:
            *v*: ``None`` | **any**
                Converted *text* from found element
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        # Get the element
        elem = self.find_kcfd(tag)
        # Check if found
        if elem is None:
            return
        # Convert *text* to value
        return self.text2val(elem.text)

   # --- Sections: get text ---
    def gettext_input(self, name):
        r"""Get the ext of an *InputList.Input* element

        :Call:
            >>> txt = xml.gettext_input(name)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *name*: :class:`str`
                Name of input attribute to query
        :Outputs:
            *txt*: ``None`` | :class:`str`
                Text of *InputList.Input* element with
                attribute *name* matching
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        # Get the element
        elem = self.find_input(name)
        # Check if found
        if elem is None:
            return
        # Return the *text*
        return elem.text

    def gettext_kcfd(self, tag):
        r"""Get text from the *KCFD* settings

        :Call:
            >>> v = xml.gettext_kcfd(tag)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *tag*: :class:`str`
                Element tag in *KCFD* parent
        :Outputs:
            *txt*: ``None`` | :class:`str`
                *text* from found element
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        # Get the element
        elem = self.find_kcfd(tag)
        # Check if found
        if elem is None:
            return
        # Convert *text* to value
        return elem.text

   # --- Text <--> value ---
    def text2val(self, txt):
        r"""Convert XML text to Python value

        :Call:
            >>> v = xml.text2val(txt)
        :Inputs:
            *xml*: :class:`JobXML`
                XML file interface
            *txt*: :class:`str`
                Text to convert
        :Outputs:
            *v*: ``None`` | |xml2py-types|
                Converted value
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0

        .. |xml2py-types| replace:
            :class:`bool` | :class:`int` | :class:`float` | :class:`str`
        """
        # Allow None
        if txt is None:
            return
        # Check if we were given something other than a string
        if not isinstance(txt, (str, unicode)):
            raise TypeError("Expected a 'str'; got '%s'" % type(txt).__name__)
        # Strip white space
        txt = txt.strip()
        # Check for two literal values
        if txt.lower() in {"true", "yes"}:
            return True
        elif txt.lower() in {"false", "no"}:
            return False
        # Attempt conversions: int first
        try:
            return int(txt)
        except ValueError:
            pass
        # Attempt conversions: float second
        try:
            return float(txt)
        except ValueError:
            pass
        # Weird case, hex?
        if re.fullmatch("0x[0-9A-Fa-f]+", txt):
            # Convert hex literal to int
            return eval(txt)
        else:
            # Unable to convert; use string
            return txt

    def val2text(self, v):
        r"""Convert Python value to XML text

        :Call:
            >>> txt = xml.val2text(v)
        :Inputs:
            *xml*: :class:`JobXML`
                XML file interface
            *v*: **any**
                Python value to convert
        :Outputs:
            *txt*: :class:`str`
                Converted text
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        # Check for recognized literals
        if v is None:
            return ""
        elif v is True:
            return "Yes"
        elif v is False:
            return "No"
        # Otherwise just convert to text using str()
        return str(v)


# Common *InputList.Input* keys
INPUTLIST_KEYS = {
    "alpha": "angle of attack",
    "beta": "sideslip angle",
    "mach": "Mach number",
    "rey": "Reynolds number",
    "relen": "Reynolds length",
    "restart": "restart flag",
}
KCFD_KEYS = {
    "iters": "number of iterations",
    "subiters": "number of subiterations",
    "timestep": "non-dimensional time step",
}


# Template docstrings
_DOCSTRING_GET_INPUT = r"""Get %(descr)s from *InputList* section

        :Call:
            >>> %(key)s = xml.get_%(key)s()
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
        :Outputs:
            *%(key)s*: ``None`` | :class:`float` | :class:`str`
                Converted value of XML element text
        """
_DOCSTRING_SET_INPUT = r"""Set %(descr)s in *InputList* section

        :Call:
            >>> xml.set_%(key)s(%(key)s)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *%(key)s*: ``None`` | :class:`float` | :class:`str`
                Value to set in XML file
        """
_DOCSTRING_GET_KCFD = r"""Get %(descr)s from *KCFD* section

        :Call:
            >>> %(key)s = xml.get_kcfd_%(key)s()
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
        :Outputs:
            *%(key)s*: ``None`` | :class:`float` | :class:`str`
                Converted value of XML element text
        """
_DOCSTRING_SET_KCFD = r"""Set %(descr)s in *KCFD* section

        :Call:
            >>> xml.set_kcfd_%(key)s(%(key)s)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *%(key)s*: ``None`` | :class:`float` | :class:`str`
                Converted value of XML element text
        """


# Set doc strings: InputList getters
for _key, _desc in INPUTLIST_KEYS.items():
    # Try to get function
    _func = getattr(JobXML, "get_%s" % _key, None)
    # Check if found
    if _func is None:
        continue
    # Format mapping
    _fmt = {"key": _key, "descr": _desc}
    # Save updated doscstring
    _func.__doc__ = _DOCSTRING_GET_INPUT % _fmt
# Set doc strings: InputList setters
for _key, _desc in INPUTLIST_KEYS.items():
    # Try to get function
    _func = getattr(JobXML, "set_%s" % _key, None)
    # Check if found
    if _func is None:
        continue
    # Format mapping
    _fmt = {"key": _key, "descr": _desc}
    # Save updated doscstring
    _func.__doc__ = _DOCSTRING_SET_INPUT % _fmt
# Set doc strings: *KCFD* getters
for _key, _desc in KCFD_KEYS.items():
    # Try to get function
    _func = getattr(JobXML, "get_kcfd_%s" % _key, None)
    # Check if found
    if _func is None:
        continue
    # Format mapping
    _fmt = {"key": _key, "descr": _desc}
    # Save updated doscstring
    _func.__doc__ = _DOCSTRING_GET_KCFD % _fmt
# Set doc strings: *KCFD* setters
for _key, _desc in KCFD_KEYS.items():
    # Try to get function
    _func = getattr(JobXML, "set_kcfd_%s" % _key, None)
    # Check if found
    if _func is None:
        continue
    # Format mapping
    _fmt = {"key": _key, "descr": _desc}
    # Save updated doscstring
    _func.__doc__ = _DOCSTRING_SET_KCFD % _fmt

