#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.jobxml`: Interface to Kestrel main XML control file
=====================================================================

This module provides the class :class:`JobXML`, which reads, edits, and
writes the main XML file that sets the main inputs for Kestrel jobs.

"""

# Standard library
import ast
import re

# Third-party

# Local imports
from .. import xmlfile


# Constants
FS_TAG = "BodyHierarchy.Freestream"


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
        * 2021-10-18 ``@ddalle``: v0.0: Started
    """
   # --- __dunder__ ---
   # --- Overall ---
    def get_job_name(self) -> str:
        return self.get_value("BodyHierarchy.Simulation.Name")

    def set_job_name(self, job_name: str):
        return self.set_section_item(
            tag="Name", value=job_name, section="Simulation")

   # --- Specific values: get ---
    def get_mach(self):
        return self.get_freestream("Mach")

    def get_alpha(self):
        return self.get_freestream("Alpha")

    def get_beta(self):
        return self.get_freestream("Beta")

    def get_pressure(self):
        return self.get_freestream("StaticPressure")

    def get_relen(self):
        return self.get_freestream("ReynoldsLength")

    def get_rey(self):
        return self.get_freestream("Reynolds")

    def get_temperature(self):
        return self.get_freestream("StaticTemperature")

    def get_velocity(self):
        return self.get_freestream("Velocity")

    def get_restart(self):
        return self.get_value("BodyHierarchy.Simulation.Restart")

    def get_kcfd_iters(self):
        return self.get_fvmcfd("Iterations")

    def get_kcfd_subiters(self):
        return self.get_fvmcfd("SubIterations")

    def get_kcfd_timestep(self):
        return self.get_fvmcfd("TimeStep")

   # --- Specific values: set ---
    def set_mach(self, mach):
        return self.set_freestream("Mach", mach)

    def set_alpha(self, alpha):
        return self.set_freestream("Alpha", alpha)

    def set_beta(self, beta):
        return self.set_freestream("Beta", beta)

    def set_pressure(self, pressure):
        return self.set_freestream("StaticPressure", pressure)

    def set_relen(self, relen):
        return self.set_freestream("ReynoldsLength", relen)

    def set_rey(self, rey):
        return self.set_freestream("Reynolds", rey)

    def set_temperature(self, temperature):
        return self.set_freestream("StaticTemperature", temperature)

    def set_velocity(self, velocity):
        return self.set_freestream("Velocity", velocity)

    def set_restart(self, restart=True):
        return self.set_section_item(
            tag="Restart", value=restart, section="BodyHierarchy.Simulation")

    def set_kcfd_iters(self, iters):
        return self.set_fvmcfd("Iterations", iters)

    def set_kcfd_subiters(self, subiters):
        return self.set_fvmcfd("Subiterations", subiters)

    def set_kcfd_timestep(self, timestep):
        return self.set_fvmcfd("TimeStep", timestep)

   # --- Sections: general ---
    def _prep_section_item(self, **kw):
        r"""Prepare :class:`dict` item descriptor for :func:`set_elem`

        :Call:
            >>> tags, v, xmlitem = xml._prep_section_item(**kw)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *tag*: :class:`str`
                Element tag using full path or shortcut
            *section*: {``None``} | ``Input`` | ``KCFD`` | :class:`str`
                Name of special section
            *value*: {``None``} | :class:`str`
                Value to set, if any
            *attrib*: {``None``} | :class:`dict`
                Requirements to match for *elem.attrib*
            *attribs*: {``None``} | :class:`list`\ [*attrib*]
                Target *attrib* for each level of *tags*
            *tail*: {``None``} | :class:`str`
                Target *elem.tail*, ignoring head/tail white space
            *exacttext*: {``None``} | :class:`str`
                Target *elem.text*, exact match
            *exacttail*: {``None``} | :class:`str`
                Target *elem.tail*, exact match
        :Outputs:
            *tags*: :class:`list`\ [:class:`str`]
                Full path to sought XML item
            *v*: ``None`` | :class:`str`
                Text from *value* above
            *xmlitem*: :class:`dict`
                Modified search parameters for section *type*
        :Versions:
            * 2021-10-18 ``@ddalle``: v1.0
            * 2024-04-17 ``@ddalle``: v1.1; Kestrel moved many tags
        """
        # Get final text
        v = kw.pop("value", None)
        # Initial *tag* specification
        tag = kw.pop("tag", None)
        # Check for special type
        itemtype = kw.pop("section", None)
        # Check for special type
        if itemtype == "KCFD":
            # Full path to <KCFD><*tag*>
            tags = [
                "BodyHierarchy",
                "KCFD",
                tag,
            ]
        elif itemtype == "FVMCFD":
            # Full path to <FVMCFD><*tag*>
            tags = [
                "BodyHierarchy",
                "ActionList",
                "FVMCFD",
                tag,
            ]
        elif itemtype == "Freestream":
            # Full path to <Freestream><*tag*> tag
            tags = [
                "BodyHierarchy",
                "Freestream",
                tag,
            ]
        elif itemtype == "Simulation":
            # Full path to <Simulation><*tag*>
            tags = [
                "BodyHierarchy",
                "Simulation",
                tag,
            ]
        elif itemtype:
            # Unknown... prepend tags
            tags = itemtype.split(".") + tag.split('.')
        else:
            # Use specified tags and options (manual option)
            tags = tag.split(".")
        # Output
        return tags, v, kw

   # --- Sections: find ---
    def find_section_item(self, **kw):
        r"""Prepare :class:`dict` item descriptor for :func:`set_elem`

        :Call:
            >>> elem = xml.find_section_item(**kw)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *tag*: :class:`str`
                Element tag using full path or shortcut
            *section*: {``None``} | ``Input`` | ``KCFD`` | :class:`str`
                Name of special section
            *value*: {``None``} | :class:`str`
                Value to set, if any
            *attrib*: {``None``} | :class:`dict`
                Requirements to match for *elem.attrib*
            *attribs*: {``None``} | :class:`list`\ [*attrib*]
                Target *attrib* for each level of *tags*
            *tail*: {``None``} | :class:`str`
                Target *elem.tail*, ignoring head/tail white space
            *exacttext*: {``None``} | :class:`str`
                Target *elem.text*, exact match
            *exacttail*: {``None``} | :class:`str`
                Target *elem.tail*, exact match
        :Outputs:
            *elem*: ``None`` | :class:`Element`
                Element matching all criteria
        :Versions:
            * 2021-10-18 ``@ddalle``: v1.0
        """
        # Prepare descriptor
        tags, _, xmlitem = self._prep_section_item(**kw)
        # Find item
        return self.find(tags, **xmlitem)

    def find_input(self, name: str, parent: str):
        r"""Get XML element of freestream or other section by tag name

        :Call:
            >>> elem = xml.find_input(name)
            >>> elem = xml.find_input(name, parent)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *name*: :class:`str`
                Name of freestream input attribute to query
            *parent*: {``"BodyHierarchy.Freestream"``} | :class:`str`
                Option to use non-default path to flow quantities
        :Outputs:
            *elem*: ``None`` | :class:`Element`
                Element matching all criteria
        :Versions:
            * 2021-10-18 ``@ddalle``: v1.0
            * 2024-04-17 ``@ddalle``: v2.0; Kestrel removed <InputList>
        """
        return self.find(f"{parent}.{name}")

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
            * 2021-10-18 ``@ddalle``: v1.0
        """
        # Full path to section
        tags = [
            "BodyHierarchy",
            "ActionList",
            "FVMCFD",
            tag
        ]
        # Find the element
        return self.find(tags)

   # --- Sections: set value ---
    def set_section_item(self, **kw):
        r"""Set value by dictionary of options

        :Call:
            >>> xml.set_section_item(**kw)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *tag*: :class:`str`
                Element tag using full path or shortcut
            *section*: {``None``} | ``Input`` | ``KCFD`` | :class:`str`
                Name of special section
            *value*: {``None``} | :class:`str`
                Value to set, if any
            *attrib*: {``None``} | :class:`dict`
                Requirements to match for *elem.attrib*
            *attribs*: {``None``} | :class:`list`\ [*attrib*]
                Target *attrib* for each level of *tags*
            *insert*: {``True``} | ``False``
                Option to insert new element(s) if not found
            *indent*: {``2``} | :class:`int` >= 0
                Number of spaces in an indent
            *tab*: {``indent * " "``} | :class:`str`
                Override *indent* with a specific string
            *text*: {``None``} | :class:`str`
                Target *elem.text* for searching
            *tail*: {``None``} | :class:`str`
                Target *elem.tail* for searching
            *exacttext*: {``None``} | :class:`str`
                Target *elem.text*, exact match
            *exacttail*: {``None``} | :class:`str`
                Target *elem.tail*, exact match
            *newattrib*: {``None``} | :class:`dict`
                New attributes to set in found element
            *newtail*: {``None``} | :class:`str`
                Specific final *tail* text for found dlement
            *updateattrib*: {``None``} | :class:`dict`
                Attributes to update without resetting *elem.attrib*
        :Versions:
            * 2021-10-26 ``@ddalle``: v1.0
        """
        # Prepare descriptor
        tags, v, xmlitem = self._prep_section_item(**kw)
        # Set item
        self.set_elem(tags, v, **xmlitem)

    def set_freestream(self, name: str, v):
        r"""Set the text of an *InputList.Input* element

        :Call:
            >>> xml.set_freestream(name, v)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *name*: :class:`str`
                Name of input attribute to query
            *v*: ``None`` | **any**
                Python value to save to element as text
        :Versions:
            * 2021-10-18 ``@ddalle``: v1.0 (set_input()
            * 2024-04-17 ``@ddalle``: v2.0
        """
        self.set_section_item(tag=name, value=v, section="Freestream")

    def set_kcfd(self, tag: str, v):
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
            * 2021-10-18 ``@ddalle``: v1.0
        """
        self.set_section_item(tag=tag, value=v, section="KCFD")

    def set_fvmcfd(self, tag: str, v):
        r"""Set the text of a *KCFD* element

        :Call:
            >>> xml.set_fvmcfd(tag, v)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *tag*: :class:`str`
                Element tag in *KCFD* parent
            *v*: ``None`` | **any**
                Python value to save to element as text
        :Versions:
            * 2024-04-17 ``@ddalle``: v1.0
        """
        self.set_section_item(tag=tag, value=v, section="FVMCFD")

   # --- Sections: get value ---
    def get_version(self) -> str:
        r"""Get the Kestrel feature set version identifier

        :Call:
            >>> featureset = xml.get_version()
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
        :Outputs:
            *featureset*: :class:`str`
                Value of *JobInputs* attribute *featureSet*
        :Versions:
            * 2024-04-17 ``@ddalle``: v1.0
        """
        return self.root.attrib.get("featureSet", "")

    def get_freestream(self, name: str):
        r"""Get the Python value of a Freestream condition

        :Call:
            >>> v = xml.get_freestream(name)
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
            * 2024-04-17 ``@ddalle``: v1.0
        """
        return self.get_value(f"{FS_TAG}.{name}")

    def get_fvmcfd(self, tag: str):
        r"""Get converted text from the *FVMCFD* settings

        :Call:
            >>> v = xml.get_fvmcfd(tag)
        :Inputs:
            *xml*: :class:`JobXML`
                Instance of Kestrel job XML file interface
            *tag*: :class:`str`
                Element tag in *KCFD* parent
        :Outputs:
            *v*: ``None`` | **any**
                Converted *text* from found element
        :Versions:
            * 2021-10-18 ``@ddalle``: v1.0
            * 2024-04-17 ``@ddalle``: v2.0; Kestrel moved KCFD section
        """
        # Get the value from current location
        return self.get_value(f"BodyHierarchy.ActionList.FVMCFD.{tag}")

    def get_kcfd(self, tag: str):
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
            * 2021-10-18 ``@ddalle``: v1.0
            * 2024-04-17 ``@ddalle``: v2.0; Kestrel moved KCFD section
        """
        # Get the value from current location
        return self.get_value(f"BodyHierarchy.KCFD.{tag}")

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
            * 2021-10-18 ``@ddalle``: v1.0
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
            * 2021-10-18 ``@ddalle``: v1.0
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
            *v*: :class:`object`
                Converted value
        :Versions:
            * 2021-10-18 ``@ddalle``: v1.0
        """
        # Allow None
        if txt is None:
            return
        # Check if we were given something other than a string
        if not isinstance(txt, str):
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
        if re.match("0x[0-9A-Fa-f]+$", txt):
            # Convert hex literal to int
            return ast.literal_eval(txt)
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
            * 2021-10-18 ``@ddalle``: v1.0
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

   # --- Meta ---
    @classmethod
    def _doc_keys(cls):
        # Set doc strings: InputList getters
        for key, desc in INPUTLIST_KEYS.items():
            # Try to get function
            func = cls.__dict__.get("get_%s" % key)
            # Check if found
            if not callable(func):
                continue
            # Format mapping
            fmt = {"key": key, "descr": desc}
            # Save updated doscstring
            func.__doc__ = _DOCSTRING_GET_INPUT % fmt
        # Set doc strings: InputList setters
        for key, desc in INPUTLIST_KEYS.items():
            # Try to get function
            func = cls.__dict__.get("set_%s" % key)
            # Check if found
            if not callable(func):
                continue
            # Format mapping
            fmt = {"key": key, "descr": desc}
            # Save updated doscstring
            func.__doc__ = _DOCSTRING_SET_INPUT % fmt
        # Set doc strings: *KCFD* getters
        for key, desc in KCFD_KEYS.items():
            # Try to get function
            func = cls.__dict__.get("get_kcfd_%s" % key)
            # Check if found
            if not callable(func):
                continue
            # Format mapping
            fmt = {"key": key, "descr": desc}
            # Save updated doscstring
            func.__doc__ = _DOCSTRING_GET_KCFD % fmt
        # Set doc strings: *KCFD* setters
        for key, desc in KCFD_KEYS.items():
            # Try to get function
            func = cls.__dict__.get("set_kcfd_%s" % key)
            # Check if found
            if not callable(func):
                continue
            # Format mapping
            fmt = {"key": key, "descr": desc}
            # Save updated doscstring
            func.__doc__ = _DOCSTRING_SET_KCFD % fmt


# Common *InputList.Input* keys
INPUTLIST_KEYS = {
    "job_name": "Kestrel job name",
    "alpha": "angle of attack",
    "beta": "sideslip angle",
    "mach": "Mach number",
    "pressure": "freestream pressure",
    "rey": "Reynolds number",
    "relen": "Reynolds length",
    "restart": "restart flag",
    "temperature": "freestream temperature",
    "velocity": "freestream flow velocity",
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


# Update docstrings
JobXML._doc_keys()

