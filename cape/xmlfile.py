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
        txt = "<%s root='%s'" % (
            self.__class__.__name__.split(".")[-1],
            self.root.tag)
        # Check for file name
        if self.fname:
            txt += " fname='%s'>" % os.path.basename(self.fname)
        else:
            txt += ">"
        # Output
        return txt

    def __str__(self):
        r"""String method

        :Versions:
            * 2021-10-07 ``@ddalle``: Version 1.0
        """
        return self.__repr__()

   # --- Find ---
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
            
