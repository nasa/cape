# -*- coding: utf-8 -*-
r"""
:mod:`cape.xmlfile`: Extended interface to XML files
======================================================

This module provides the class :class:`XMLFile`, which extends slightly
the built-in class :class:`xml.etree.ElmentTree`. Compared to the
standard library class, :class:`XMLFile` has a more top-level interface.

Specifically, it is possible to find and/or edit properties of
subelements that are arbitrarily deep within the file using methods for
the top-level class. This is convenient (for example) for CFD solvers
using XML files as their input because it eases the process of changing
minor settings (for example the angle of attack) without searching
through multiple levels of elements and subelements.

"""

# Standard library
import copy
import os
import xml.etree.ElementTree as ET


# Primary class
class XMLFile(object):
    r"""Interface to XML files

    :Call:
        >>> xml = XMLFile(fxml)
        >>> xml = XMLFile(xml1)
        >>> xml = XMLFile(et)
        >>> xml = XMLFile(e)
        >>> xml = XMLFile(txt)
        >>> xml = XMLFile()
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
            Another instance
    :Versions:
        * 2021-10-06 ``@ddalle``: Version 0.0: Started
    """
   # --- __dunder__ ---
    def __init__(self, arg0=None, **kw):
        r"""Initialization method

        :Versions:
            * 2021-10-06 ``@ddalle``: Version 1.0
        """
        # Initialize elements
        self.tree = None
        self.root = None
        self.fname = None
        # Check input type
        if arg0 is None:
            # Empty element tree
            e = ET.ElementTree()
        elif isinstance(arg0, ET.ElementTree):
            # Existing element tree
            e = arg0
        elif isinstance(arg0, ET.Element):
            # Create element tree from a root element
            e = ET.ElementTree(arg0)
        elif isinstance(arg0, XMLFile):
            # Copy the element tree
            e = copy.deepcopy(arg0.tree)
            # Copy any file name
            self.fname = arg0.fname
        elif isinstance(arg0, str):
            # Check if it looks like an XML file
            if arg0.lstrip().startswith("<"):
                # Looks like an XML file
                elem = ET.fromstring(arg0)
                # Create a tree
                e = ET.ElementTree(elem)
            elif os.path.isfile(arg0):
                # Existing file
                e = ET.parse(arg0)
                # Save file name
                self.fname = arg0
            else:
                # Unreadable string
                raise ValueError(
                    "Received unreadable string starting with " +
                    ("'%s...'" % arg0[:20]))
        else:
            raise TypeError(
                ("XMLFile requires type 'str', 'XMLFile', 'ElementTree', ") +
                ("or 'Element'; got '%s'" % type(arg0).__name__))
        # Save the element tree
        self.tree = e
        self.root = e.getroot()

    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2021-10-07 ``@ddalle``: Version 1.0
        """
        # Initialize
        txt = "<%s " % self.__class__.__name__.split(".")[-1]
        # Check for file name
        if self.fname:
            txt += "fname='%s' " % os.path.basename(self.fname)
        # Add name of root element
        txt += "root='%s'>" % self.root.tag
        # Output
        return txt

    def __str__(self):
        r"""String method

        :Versions:
            * 2021-10-07 ``@ddalle``: Version 1.0
        """
        return self.__repr__()

   # --- I/O ---
    def write(self, fname=None):
        r"""Write a file

        :Call:
            >>> xml.write(fname=None)
        :Inputs:
            *xml*: :class:`XMLFile`
                XML file interface
            *fname*: {``None``} | :class:`str`
                Name of file to write
        :Versions:
            * 2021-10-07 ``@ddalle``: Version 1.0
        """
        # Default file name
        if fname is None:
            fname = self.fname
        # Check again
        if fname is None:
            raise ValueError("No file name determined")
        # Write file
        self.tree.write(fname)

   # --- Set ---
    def set_elem(self, tag, newtext=None, attrib=None, **kw):
        # Options
        attribs = kw.get("attribs")
        indent = kw.pop("indent", 2)
        insert = kw.pop("insert", True)
        tab = kw.pop("tab", " "*indent)
        # Other output properties
        newattrib = kw.pop("newattrib", None)
        newtail = kw.pop("newtail", None)
        # Get list of tags
        if isinstance(tag, (list, tuple)):
            # Already a list
            tags = list(tag)
        else:
            # Split levels using "."
            tags = tag.split(".")
        # Check for *attribs*
        if isinstance(attrib, (list, tuple)):
            attribs = list(attrib)
            attrib = None
        # Number of levels
        ntag = len(tags)
        # Find (or try to) full path to *elem*
        elems = self.find_trail(tag, attrib, **kw)
        # Number found
        nelem = len(elems)
        # Check length
        if nelem >= ntag:
            # Found element; set text
            elem = elems[ntag - 1]
        elif not insert:
            # Missing key
            raise KeyError("Couldn't find element '%s'" % ".".join(tags))
        else:
            # Get handle to last parent found
            if nelem == 0:
                # Use root element
                e = self.root
            else:
                # Use last level found
                e = elems[-1]
            # Loop through required levels
            for j in range(nelem, ntag):
                # Required and optional properties
                tagj = tags[j]
                # Set attributes from list
                if attribs and len(attribs) > j:
                    # Found attribute from list
                    attribj = attribs[j]
                else:
                    # No attributes from list
                    attribj = None
                # Check for attribute from *attrib*
                if j + 1 == ntag and attrib:
                    attribj = attrib
                # Create element
                ej = toelement(tagj, attribj)
                # Set text if needed
                if newtext:
                    ej.text = newtext
                # Use this weird code to get the first child (if any)
                # Copy tail from first element
                if len(e) == 0:
                    # Use same tail as *ej* parent
                    ej.tail = e.tail
                    # Increase indent of parent
                    if "\n" in e.text:
                        e.text = e.text + tab
                else:
                    # New element's tail same as parent's text
                    ej.tail = e.text
                # Insert element
                e.insert(0, ej)
                # Move to next level
                e = ej
            # Use last newly inserted element
            elem = e
        # Set values
        if newtext == "":
            elem.text = None
        elif newtext:
            elem.text = newtext
        # Check for other values
        if isinstance(newattrib, dict):
            elem.attrib = newattrib
        if newtail is not None:
            elem.tail = newtail

    def remove(self, tag, attrib=None, text=None, **kw):
        # Pop the element, with no error
        elem = self.pop(tag, attrib, text, **kw)
        # Check if removed
        if elem is None:
            raise ValueError("XMLFile.remove(): element not found")

    def pop(self, tag, attrib=None, text=None, **kw):
        # Get list of tags
        if isinstance(tag, (list, tuple)):
            # Already a list
            tags = list(tag)
        else:
            # Split levels using "."
            tags = tag.split(".")
        # Number of levels
        ntag = len(tags)
        # Find (or try to) full path to *elem*
        elems = self.find_trail(tag, attrib, **kw)
        # Check depth
        if len(elems) == ntag:
            # Save reference to last element
            elem = elems[-1]
            # Check if direct child of root
            if ntag == 1:
                # Remove from root
                self.root.remove(elem)
            else:
                # Remove from immediate parent
                elems[[-2].remove(elem)
            # Removed one element
            return elem

   # --- Find ---
    def find(self, tag, attrib=None, text=None, **kw):
        r"""Find an element using full path and expanded search criteria

        :Call:
            >>> elem = xml.find(tag, attrib=None, **kw)
            >>> elem = xml.find(tags, attrib=None, **kw)
        :Inputs:
            *xml*: :class:`XMLFile`
                XML file interface
            *tag*: :class:`str`
                Subelement tag, using ``'.'`` to separate levels
            *tags*: :class:`list`\ [:class:`str`]
                Path of tags to sought *elem*
            *attrib*: {``None``} | :class:`dict`
                Requirements to match for *elem.attrib*
            *attribs*: {``None``} | :class:`list`\ [*attrib*]
                Target *attrib* for each level of *tags*
            *text*: {``None``} | :class:`str`
                Target *elem.text*, ignoring head/tail white space
            *tail*: {``None``} | :class:`str`
                Target *elem.tail*, ignoring head/tail white space
            *exacttext*: {``None``} | :class:`str`
                Target *elem.text*, exact match
            *exacttail*: {``None``} | :class:`str`
                Target *elem.tail*, exact match
        :Outputs:
            *elem*: :class:`Element`
                Element matching all criteria
        :Versions:
            * 2021-10-07 ``@ddalle``: Version 1.0
        """
        # Get list of tags
        if isinstance(tag, (list, tuple)):
            # Already a list
            tags = tag
        else:
            # Split levels using "."
            tags = tag.split(".")
        # Number of levels
        ntag = len(tags)
        # Find (or try to) full path to *elem*
        elems = self.find_trail(tag, attrib, text, **kw)
        # Check length
        if len(elems) == ntag:
            return elems[-1]

    def find_trail(self, tag, attrib=None, text=None, **kw):
        r"""Find an element using full path and expanded search criteria

        :Call:
            >>> elems = xml.find_trail(tag, attrib=None, **kw)
            >>> elems = xml.find_trail(tags, attrib=None, **kw)
        :Inputs:
            *xml*: :class:`XMLFile`
                XML file interface
            *tag*: :class:`str`
                Subelement tag, using ``'.'`` to separate levels
            *tags*: :class:`list`\ [:class:`str`]
                Path of tags to sought *elem*
            *attrib*: {``None``} | :class:`dict`
                Requirements to match for *elem.attrib*
            *attribs*: {``None``} | :class:`list`\ [*attrib*]
                Target *attrib* for each level of *tags*
            *text*: {``None``} | :class:`str`
                Target *elem.text*, ignoring head/tail white space
            *tail*: {``None``} | :class:`str`
                Target *elem.tail*, ignoring head/tail white space
            *exacttext*: {``None``} | :class:`str`
                Target *elem.text*, exact match
            *exacttail*: {``None``} | :class:`str`
                Target *elem.tail*, exact match
        :Outputs:
            *elems*: :class:`list`\ [:class:`Element`]
                Path of elements leading up to requested tag
        :Versions:
            * 2021-10-07 ``@ddalle``: Version 1.0
        """
        # Turn off search-sublevels option
        kw["finditer"] = False
        # Options
        tail = kw.pop("tail", None)
        attribs = kw.pop("attribs", None)
        exacttext = kw.pop("exacttext", None)
        exacttail = kw.pop("exactfail", None)
        # Get list of tags
        if isinstance(tag, (list, tuple)):
            # Already a list
            tags = list(tag)
        else:
            # Split levels using "."
            tags = tag.split(".")
        # Check for *attribs*
        if isinstance(attrib, (list, tuple)):
            attribs = list(attrib)
            attrib = None
        # Number of levels
        ntag = len(tags)
        # Initialize output
        elems = []
        # Current parent
        e = self.root
        # Loop through levels
        for j, tagj in enumerate(tags):
            # Check for criteria on last tag
            if j + 1 == ntag:
                # Search criteria for end of trail
                kwj = {
                    "attrib": attrib,
                    "text": text,
                    "tail": tail,
                    "exacttext": exacttext,
                    "exacttail": exacttail,
                }
            else:
                # No search criteria for end of trail
                kwj = {}
            # Get search criteria
            if attribs and len(attribs) > j:
                # Use attribute for level *j*
                kwj["attrib"] = attribs[j]
            # Search
            ej = find_elem(e, tagj, **kwj)
            # Check for find
            if ej is None:
                return elems
            # Append to list
            elems.append(ej)
            # Move to next deeper level
            e = ej
        # Output
        return elems

    def find_iter(self, tag=None, attrib=None, text=None, **kw):
        r"""Find an element at any level using various search criteria
    
        :Call:
            >>> elem = xml.find_iter(tag, attrib=None, text=None, **kw)
        :Inputs:
            *xml*: :class:`XMLFile`
                XML file interface
            *tag*: {``None``} | :class:`str`
                Name of child element to match
            *attrib*: {``None``} | :class:`dict`
                Dictionary of attributes to match
            *text*: {``None``} | :class:`str`
                Element text to match, ignoring head/tail
            *tail*: {``None``} | :class:`str`
                Post-element text to match, ignoring head/tail
            *exacttext*: {``None``} | :class:`str`
                Element text to match exactly
            *exacttail*: {``None``} | :class:`str`
                Post-element text to match exactly
            *finditer*: ``True`` | {``False``}
                Option to search all levels, not just immediate children
        :Outputs:
            *elem*: :class:`xml.etree.ElementTree.Element`
                An XML element matching all criteria above
        :Versions:
            * 2021-10-07 ``@ddalle``: Version 1.0
        """
        # Set search-sublevels option
        kw["finditer"] = True
        # Search
        return find_elem(self.root, tag, attrib, text, **kw)

    def findall_iter(self, tag=None, attrib=None, text=None, **kw):
        r"""Find all elements at any level using various search criteria
    
        :Call:
            >>> elems = xml.findall_iter(tag, attrib=None, **kw)
        :Inputs:
            *xml*: :class:`XMLFile`
                XML file interface
            *tag*: {``None``} | :class:`str`
                Name of child element to match
            *attrib*: {``None``} | :class:`dict`
                Dictionary of attributes to match
            *text*: {``None``} | :class:`str`
                Element text to match, ignoring head/tail
            *tail*: {``None``} | :class:`str`
                Post-element text to match, ignoring head/tail
            *exacttext*: {``None``} | :class:`str`
                Element text to match exactly
            *exacttail*: {``None``} | :class:`str`
                Post-element text to match exactly
            *finditer*: ``True`` | {``False``}
                Option to search all levels, not just immediate children
        :Outputs:
            *elems*: :class:`list`\ [:class:`Element`]
                All XML elements matching all criteria above
        :Versions:
            * 2021-10-07 ``@ddalle``: Version 1.0
        """
        # Set search-sublevels option
        kw["finditer"] = True
        # Search
        return find_elem(self.root, tag, attrib, text, **kw)
        


# Find a subelement
def find_elem(e, tag=None, attrib=None, text=None, **kw):
    r"""Find a [direct] child of *e* using full search criteria

    :Call:
        >>> elem = find_elem(e, tag, attrib=None, text=None, **kw)
    :Inputs:
        *e*: :class:`xml.etree.ElementTree.Element`
            An XML element using standard library interface
        *tag*: {``None``} | :class:`str`
            Name of child element to match
        *attrib*: {``None``} | :class:`dict`
            Dictionary of attributes to match
        *text*: {``None``} | :class:`str`
            Element text to match, ignoring head/tail
        *tail*: {``None``} | :class:`str`
            Post-element text to match, ignoring head/tail
        *exacttext*: {``None``} | :class:`str`
            Element text to match exactly
        *exacttail*: {``None``} | :class:`str`
            Post-element text to match exactly
        *finditer*: ``True`` | {``False``}
            Option to search all levels, not just immediate children
    :Outputs:
        *elem*: :class:`xml.etree.ElementTree.Element`
            An XML element matching all criteria above
    :Versions:
        * 2021-10-07 ``@ddalle``: Version 1.0
    """
    # Other options
    tail = kw.pop("tail", None)
    exacttail = kw.pop("exacttail", None)
    exacttext = kw.pop("exacttext", None)
    finditer = kw.pop("finditer", False)
    # Get iterator for children of *e*
    if finditer:
        # Loop through children of arbitrary depth
        items = e.iter()
        # Skip first item, which is *e* itself
        next(items)
    else:
        # Loop through *e* itself, which gives direct children
        items = e
    # Loop through items from selected iterator above
    for elem in items:
        # Check for *tag*
        if tag:
            # Compare tag
            if elem.tag != tag:
                continue
        # Check for *attrib*
        if attrib:
            # Initialize pass/fail flag
            match = True
            # Loop through search criteria
            for k, v in attrib.items():
                # Get value
                v1 = elem.attrib.get(k)
                # Compare
                if v1 != v:
                    match = False
                    break
            # Check overall critera
            if not match:
                continue
        # Check for *text*
        if text is not None:
            # Compare text
            if elem.text is None:
                continue
            if elem.text.strip() != text.strip():
                continue
        # Check for *tail*
        if tail is not None:
            # Compare post-tag text
            if elem.tail is None:
                continue
            if elem.tail.strip() != tail.strip():
                continue
        # Check for exact text
        if exacttext is not None:
            # Compare text w/o stripping
            if elem.text is None:
                continue
            if elem.text != exacttext:
                continue
        # Check for exact tail
        if exacttail is not None:
            # Compare posttext w/o stripping
            if elem.tail is None:
                continue
            if elem.tail != exacttail:
                continue
        # All tests passed
        return elem
            

# Find all subelement
def findall_elem(e, tag=None, attrib=None, text=None, **kw):
    r"""Find all [direct] children of *e* using full search criteria

    :Call:
        >>> elems = findall_elem(e, tag, attrib=None, text=None, **kw)
    :Inputs:
        *e*: :class:`xml.etree.ElementTree.Element`
            An XML element using standard library interface
        *tag*: {``None``} | :class:`str`
            Name of child element to match
        *attrib*: {``None``} | :class:`dict`
            Dictionary of attributes to match
        *text*: {``None``} | :class:`str`
            Element text to match, ignoring head/tail
        *tail*: {``None``} | :class:`str`
            Post-element text to match, ignoring head/tail
        *exacttext*: {``None``} | :class:`str`
            Element text to match exactly
        *exacttail*: {``None``} | :class:`str`
            Post-element text to match exactly
        *finditer*: ``True`` | {``False``}
            Option to search all levels, not just immediate children
    :Outputs:
        *elems*: :class:`list`\ [:class:`Element`]
            All XML elements matching all criteria above
    :Versions:
        * 2021-10-07 ``@ddalle``: Version 1.0
    """
    # Other options
    tail = kw.pop("tail", None)
    exacttail = kw.pop("exacttail", None)
    exacttext = kw.pop("exacttext", None)
    finditer = kw.pop("finditer", False)
    # Initialize output
    elems = []
    # Get iterator for children of *e*
    if finditer:
        # Loop through children of arbitrary depth
        items = e.iter()
        # Skip first item, which is *e* itself
        next(items)
    else:
        # Loop through *e* itself, which gives direct children
        items = e
    # Loop through items from selected iterator above
    for elem in items:
        # Check for *tag*
        if tag:
            # Compare tag
            if elem.tag != tag:
                continue
        # Check for *attrib*
        if attrib:
            # Initialize pass/fail flag
            match = True
            # Loop through search criteria
            for k, v in attrib.items():
                # Get value
                v1 = elem.attrib.get(k)
                # Compare
                if v1 != v:
                    match = False
                    break
            # Check overall critera
            if not match:
                continue
        # Check for *text*
        if text is not None:
            # Compare text
            if elem.text is None:
                continue
            if elem.text.strip() != text.strip():
                continue
        # Check for *tail*
        if tail is not None:
            # Compare post-tag text
            if elem.tail is None:
                continue
            if elem.tail.strip() != tail.strip():
                continue
        # Check for exact text
        if exacttext is not None:
            # Compare text w/o stripping
            if elem.text is None:
                continue
            if elem.text != exacttext:
                continue
        # Check for exact tail
        if exacttail is not None:
            # Compare posttext w/o stripping
            if elem.tail is None:
                continue
            if elem.tail != exacttail:
                continue
        # All tests passed
        elems.append(elem)
    # Output
    return elems


# Convert properties to an element
def toelement(tag, attrib=None, text=None, tail=None):
    r"""Create a new element from basic properties

    :Call:
        >>> e = toelement(tag, attrib=None, text=None, tail=None)
    :Inputs:
        *tag*: :class:`str`
            Name of the XML element
        *attrib*: {``None``} | :class:`dict`
            Attributes for the XML element
        *text*: {``None``} | :class:`str`
            Text for the interior of the element
        *tail*: {``None``} | :class:`str`
            Text after the element
    :Outputs:
        *e*: :class:`Element`
            New XML element
    :Versions:
        * 2021-10-07 ``@ddalle``: Version 1.0
    """
    # Default attributes
    if attrib is None:
        attrib = {}
    # Create element
    e = ET.Element(tag, attrib=attrib)
    # Set texts
    e.text = text
    e.tail = tail
    # Output
    return e

