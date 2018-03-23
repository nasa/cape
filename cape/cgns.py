#!/usr/bin/env python
"""
:mod:`cape.cgns`: CGNS File Interface 
======================================

This module provides a class for reading CGNS files of certain types that have
been implemented.  The main purpose is to read surface triangulations with
quads and convert it into a :class:`cape.tri.Tri` object.  However, the class
provided in this module, :mod:`cape.cgns.CGNS`, must be converted into a
:class:`cape.tri.Tri` or other object externally.  This module merely reads the
file, reads the data from each node, and constructs a sub-node table.

In order to create a triangulation interface from a CGNS file, see
:class:`cape.tri.Tri`, and use the *cgns* keyword argument.

"""

# Required modules
import numpy as np

# Convert a 12-byte ADF string into and address
def ADFAddress2Pos(addr):
    """Convert ADF 12-byte code into position index
    
    This is a hex code with but skipping the 8th character
    
    :Call:
        >>> pos = ADFAddress2Pos(addr)
    :Inputs:
        *addr*: :class:`str`
            12-character string hex code
    :Outputs:
        *pos*: :class:`int`
            Position in file in bytes
    :Versions:
        * 2018-03-02 ``@ddalle``: First version
    """
    # Use '0x' to signify hex and convert to integer
    return eval('0x' + addr[:8] + addr[9:])
            

# CGNS class
class CGNS(object):
    """
    Interface to CGNS files
    
    :Call:
        >>> cgns = cape.cgns.CGNS(fname)
    :Inputs:
        *fname*: :class:`str` | :class:`unicode`
            Name of CGNS file
    :Outputs:
        *cgns*: :class:`cape.cgns.CGNS`
            CGNS file interface
        *cgns.nNode*: :class:`int`
            Number of nodes read
        *cgns.NodeNames*: :class:`np.ndarray` (:class:`str`)
            Names of each node
        *cgns.NodeLabels*: :class:`np.ndarray` (:class:`str`)
            Labels for each node (often a data type holder)
        *cgns.NodeAddresses*: :class:`np.ndarray` (:class:`int`)
            File position of the beginning of each node
        *cgns.DataTypes*: :class:`list` (:class:`str`)
            Data type for each node
        *cgns.Data*: :class:`list` (:class:`np.ndarray` | :class:`str`)
            Data set for each node
        *cgns.SubNodeTables*: :class:`list` (:class:`list` | ``None``)
            List of any child nodes for each node
    :Versions:
        * 2018-03-02 ``@ddalle``: First version
    """
  # ========
  # Config
  # ========
  # <
    # Initialization method
    def __init__(self, fname):
        """Initialization method
        
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        # Open the file for binary reading
        try:
            f = open(fname, 'rb')
        except Exception:
            raise IOError("Unable to open CGNS file '%s' for binary reading"
                % fname)
        # Save the file name
        self.fname = fname
        # Initialize fields
        self.NodeNames = [] 
        self.NodeLabels = []
        self.NodeAddresses = []
        self.SubNodeTables = []
        self.DataTypes = []
        self.Data = []
        # Node count
        self.nNode = 0
        
        # Call readers
        try:
            # ADF format
            self.ReadADF(f)
            # Cleanup
            f.close()
        except Exception as e:
            # Close the file
            f.close()
            # Not implemented or failed
            print("Unimplemented file format or " +
                "failed to read ADF file")
            # Return original error
            raise e
    
    # String command
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        return "<CGNS '%s', nNode=%i>" % (self.fname, self.nNode)
        
    # String command
    def __str__(self):
        """String method
        
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        return "<CGNS '%s', nNode=%i>" % (self.fname, self.nNode)
  # >
  
  # =========
  # Readers
  # =========
  # <
   # -------
   # ADF
   # -------
   # [  
    # Read ADF file
    def ReadADF(self, f):
        """Read open CGNS/ADF file, currently at position 0
        
        :Call:
            >>> cgns.ReadADF(f)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *f*: :class:`file`
                Open file currently at the beginning of *NoDe* field
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
       # --- Header ---
        # Go to beginning of file, but skip first four characters
        f.seek(4)
        # Read next 28 chars, should be database version
        s = f.read(28)
        # Check for error
        if not s.startswith("ADF"):
            raise ValueError("File is not in ADF format")
        # Try to read through the header...
        ih = 0
        while (ih < 10):
            # Header read counter
            ih += 1
            # Read the next four characters
            s = f.read(4).lower()
            # Check against known list
            if s == "fcte":
                # End; start data
                break
            elif s == "adf0":
                # Read date
                self.date = f.read(28).rstrip()
            elif s == "adf1":
                # This is also date
                self.date = f.read(28).rstrip()
            elif s == "adf2":
                # This is the characters "LB"
                f.seek(2, 1)
            elif s == "adf3":
                # This is some sort of 24-byte code
                f.seek(24, 1)
            elif s == "adf4":
                # This is some sort of 48-byte code
                f.seek(48, 1)
            elif s == "adf5":
                # This is some sort of 76-byte code (unusual...)
                f.seek(76, 1)
            else:
                # Uh oh
                raise ValueError("Unreadable header field '%s'" % s)
       # --- Nodes ---
        # Loop through nodes
        iNode = 0
        while (iNode < 5000):
            # Increase read attempt count
            iNode += 1
            # Read the node
            q = self.ReadADFNode(f)
            # Exit if no node
            if not q:
                break
       # --- Cleanup ---
        # Convert to arrays
        self.NodeNames     = np.array(self.NodeNames)
        self.NodeLabels    = np.array(self.NodeLabels)
        self.NodeAddresses = np.array(self.NodeAddresses)
       # --- Error Handling ---
        # Fix obvious errors with sub-node-tables...
        # Find *Elements_t* and *IndexRange_t* (if needed) nodes
        KE = self.GetNodeIndex(label="Elements_t")
        KI = self.GetNodeIndex(label="IndexRange_t")
        # Loop through them
        for k in KE:
            # Check for sub-node-table
            if self.SubNodeTables[k] is not None:
                continue
            # Find the *IndexRange_t* nodes **after** *k*
            JI = KI[np.where(KI > k)]
            # Check for match
            if len(JI) == 0:
                # No such node
                continue
            else:
                # Take the first node
                j = JI[0]
            # Assemble the table
            self.SubNodeTables[k] = [
                [self.NodeNames[k], self.NodeAddresses[k]],
                [self.NodeNames[j], self.NodeAddresses[j]]
            ]
        
    # Read node
    def ReadADFNode(self, f):
        """Read a (new) node from an open CGNS/ADF file
        
        :Call:
            >>> cgns.ReadADFNode(f)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *f*: :class:`file`
                Open file currently at the beginning of *NoDe* field
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        # Save current location to store if valid node
        ia = f.tell()
        # Read the next four bytes
        s = f.read(4)
        # Check for error
        if s.lower() != "node":
            return 0
        # Read the node name
        NodeName  = f.read(32).rstrip()
        NodeLabel = f.read(32).rstrip()
        # Ignore next 16 bytes (no idea... mostly zeros)
        f.seek(16, 1)
        # Read the address of the end except for data (I guess?)
        # It seems to get you to 'DaTa' or be an invalid address...
        ib = ADFAddress2Pos(f.read(12))
        # Read data format
        DataType = f.read(2)
        # Skip the next 108 bytes of nonsense
        f.seek(132, 1)
        # Here's another address, which seems to get you to 'TaiL'
        ic = ADFAddress2Pos(f.read(12))
        # Read the next four characters, which should hopefully be 'TaiL'
        s = f.read(4)
        # Check
        if s.lower() != "tail":
            return
        # Skip FreE field (if any)
        self.ReadADFFree(f)
        # Read SubNodeTable (if any)
        sntb = self.ReadADFSubNodeTable(f)
        # Read Data (if any)
        data = self.ReadADFData(f, DataType)
        # Any "zzzzzzzzz" nonsense?
        self.ReadADFZs(f)
        # Save node information
        self.NodeNames.append(NodeName)
        self.NodeLabels.append(NodeLabel)
        self.NodeAddresses.append(ia)
        self.DataTypes.append(DataType)
        self.SubNodeTables.append(sntb)
        self.Data.append(data)
        # Node count
        self.nNode += 1
        # Output successful read
        return 1
        
    # Read annoying "z"s
    def ReadADFZs(self, f):
        """Read strings like "zzzzzzzzzzzz" until reaching a relevant char
        
        :Call:
            >>> cgns.ReadADFZs(f)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *f*: :class:`file`
                Open file currently at the beginning of *DaTa* field
        :Versions:
            * 2018-03-05 ``@ddalle``: First version
        """
        # Read the next four bytes
        s = f.read(4)
        # Deal with "z"s (I have no idea WTF this is about)
        if s == "zzzz":
            # Read until the next character is NOT a "z"
            while f.read(1) == "z":
                continue
            # Go back one character
            f.seek(-1, 1)
        else:
            # Go back to original position by seeking backward 4 chars
            f.seek(-4, 1)
        
    # Read annoying "FreE" block
    def ReadADFFree(self, f):
        """Read one *FreE* entry from an open CGNS/ADF file
        
        The next 4 bytes must be the string ``"FreE"``, and the following 12
        bytes must give the address of the end of the field as a hex code
        string.
        
        :Call:
            >>> cgns.ReadADFFreE(f)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *f*: :class:`file`
                Open file currently at the beginning of *DaTa* field
        :Versions:
            * 2018-03-05 ``@ddalle``: First version
        """
        # Save current location
        ja = f.tell()
        # Read the next four bytes
        s = f.read(4)
        # Deal with "z"s (I have no idea WTF this is about)
        if s == "zzzz":
            # Read until the next character is NOT a "z"
            while f.read(1) == "z":
                continue
            # Go back one character
            f.seek(-1, 1)
            # Reread next four characters
            s = f.read(4)
        # Check for error
        if len(s) < 4:
            # EOF
            return
        elif s.lower() != "free":
            # Some other field...
            f.seek(-4, 1)
            return
        # Read the next 12 bytes to get the address of the end of the field
        jb = ADFAddress2Pos(f.read(12))
        # Go to that position
        f.seek(jb)
        # Read the tail
        s = f.read(4)
        # Check the correct end-of-subnodetable
        if s.lower() != "endc":
            raise ValueError("FreE field must end with string 'EndC'; " +
                ("file contains '%s'" % s))
    
    # Read data
    def ReadADFSubNodeTable(self, f):
        """Read one *SNTb* entry from an open CGNS/ADF file
        
        The next four bytes must be the string ``"SNTb"``, and the following 12
        bytes must give the address of the end of the field as a hex code
        string.
        
        :Call:
            >>> sntb = cgns.ReadADFSubNodeTable(f)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *f*: :class:`file`
                Open file currently at the beginning of *DaTa* field
        :Outputs:
            *sntb*: :class:`np.array` ([:class:`str`, :class:`int`])
                List of name and begin addresses of any subnodes
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        # Save current location
        ja = f.tell()
        # Read the next four bytes
        s = f.read(4)
        # Deal with "z"s (I have no idea WTF this is about)
        if s == "zzzz":
            # Read until the next character is NOT a "z"
            while f.read(1) == "z":
                continue
            # Go back one character
            f.seek(-1, 1)
            # Reread next four characters
            s = f.read(4)
        # Check for error
        if len(s) < 4:
            # EOF
            return
        elif s.lower() != "sntb":
            # Some other field...
            f.seek(-4, 1)
            return
        # Read the next 12 bytes to get the address of the end of the field
        jb = ADFAddress2Pos(f.read(12))
        # Initialize table
        sntb = []
        # Loop until reaching *jb*
        while f.tell() < jb:
            # Read node name and address
            SubZone = f.read(32).rstrip()
            SubAddr = ADFAddress2Pos(f.read(12))
            # Check for "unused" subnode
            if SubZone.startswith("unused entry"):
                continue
            # Save
            sntb.append([SubZone, SubAddr])
        # Read the tail
        s = f.read(4)
        # Check the correct end-of-subnodetable
        if s.lower() != "snte":
            raise ValueError("Data field must end with string 'snTE'; " +
                ("file contains '%s'" % s))
        # Output
        return sntb
    
    # Read data
    def ReadADFData(self, f, dt):
        """Read one *DaTa* entry from an open CGNS/ADF file
        
        The next four bytes must be the string ``"DaTa"``, and the following 12
        bytes must give the address of the end of the field as a hex code
        string.
        
        :Call:
            >>> data = cgns.ReadADFData(f, dt)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *f*: :class:`file`
                Open file currently at the beginning of *DaTa* field
            *dt*: ``"MT"`` | ``"C1"`` | ``"I4"`` | ``"R4"`` | ``"R8"``
                Data type, two-digit code
        :Outputs:
            *data*: :class:`np.ndarray` | :class:`str`
                Data read from file
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        # Read the next four bytes
        s = f.read(4)
        # Deal with "z"s (I have no idea WTF this is about)
        if s == "zzzz":
            # Read until the next character is NOT a "z"
            while f.read(1) == "z":
                continue
            # Go back one character
            f.seek(-1, 1)
            # Reread next four characters
            s = f.read(4)
        # Check for error
        if len(s) < 4:
            # EOF
            return
        elif s.lower() != "data":
            # Some other field...
            f.seek(-4, 1)
            return
        # Read the next 12 bytes to get the address of the end of the field
        addr = f.read(12)
        # Get current address
        ja = f.tell()
        # Convert end address to a hex
        jb = ADFAddress2Pos(addr)
        # Number of bytes
        nb = jb - ja
        # Read the data
        if dt == "MT":
            # No data
            data = None
            f.seek(jb)
        elif dt == "C1":
            # String
            data = f.read(nb)
        elif dt == "I4":
            # Regular integer
            data = np.fromfile(f, dtype="i4", count=nb/4)
        elif dt == "I8":
            # Long integer
            data = np.fromfile(f, dtype="i8", count=nb/8)
        elif dt == "U4":
            # Unsigned integer
            data = np.fromfile(f, dtype="u4", count=nb/4)
        elif dt == "U8":
            # Signed integer
            data = np.fromfile(f, dtype="u8", count=nb/8)
        elif dt == "R4":
            # Single-precision real
            data = np.fromfile(f, dtype="f4", count=nb/4)
        elif dt == "R8":
            # Double-precision real
            data = np.fromfile(f, dtype="f8", count=nb/8)
        elif dt == "X4":
            # Single-precision complex
            data = np.fromfile(f, dtype="f4", count=nb/2)
            # Convert floats to complex
            data = data[::2] + 1j*data[1::2]
        elif dt == "X8":
            # Double-precision complex
            data = np.fromfile(f, dtype="f8", count=nb/4)
            # Convert floats to complex
            data = data[::2] + 1j*data[1::2]
        else:
            raise ValueError("Unrecognized data type '%s'" % dt)
        # Read the tail
        s = f.read(4)
        # Check the correct end-of-data
        if s.lower() != "dend":
            raise ValueError("Data field must end with string 'dEnD'; " +
                ("file contains '%s'" % s))
        # Output
        return data
   # ]
  # >
  
  # ===============
  # Node Interface
  # ===============
  # <
    # Get node index
    def GetNodeIndex(self, name=None, label=None, addr=None):
        """Get index of a node by name and label or address
        
        :Call:
            >>> K = cgns.GetNodeIndex(name=None, label=None, addr=None)
            >>> K = cgns.GetNodeIndex(name)
            >>> K = cgns.GetNodeIndex(name, label)
            >>> K = cgns.GetNodeIndex(addr)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *name*: {``None``} | :class:`str`
                Node name
            *label*: { ``None``} | :class:`str`
                Node label
            *addr*: {``None``} | :class:`int`
                Address of the beginning of the node
        :Outputs:
            *k*: :class:`np.ndarray` (:class:`int`)
                Node index list
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        # Process inputs
        if name is not None:
            # Check type
            tn = name.__class__.__name__
            # Check string
            if tn not in ["str", "unicode"]:
                # Pass it on to address
                addr = name
                name = None
        # Initialize
        M = np.ones(self.nNode, dtype="bool")
        # Check for name constraint
        if name is not None:
            # Filter *name*
            M = np.logical_and(M, self.NodeNames==name)
        # Check for label constraint
        if label is not None:
            # Filter *label*
            M = np.logical_and(M, self.NodeLabels==label)
        # Check for address constraint
        if addr is not None:
            M = np.logical_and(M, self.NodeAddresses==addr)
        # Find matches
        K = np.where(M)[0]
        # Output
        return K
            
    # Get node by address only
    def GetNodeByAddress(self, addr):
        """Get index of node by explicit address (only unique identifier)
        
        :Call:
            >>> k = cgns.GetNodeByAddress(addr)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *addr*: {``None``} | :class:`int`
                Address of the beginning of the node
        :Outputs:
            *k*: :class:`int`
                Node index
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        # Find matching address
        k = np.where(self.NodeAddresses == addr)[0]
        # Check for match
        if len(k) != 1:
            raise ValueError("No match for address '%s'" % addr)
        # Output
        return k[0]
        
    # Get *IndexRange_t* node index for a named zone
    def GetCompIDInfo(self, comp):
        """Get the info for a named zone with type *Elements_t*
        
        :Call:
            >>> ka, kb, Elems = cgns.GetCompIDInfo(comp)
            >>> ka, kb, Elems = cgns.GetCompIDInfo(addr)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *comp*: :class:`str`
                Name of component to read
            *addr*: :class:`int`
                Address of the node within the file
        :Outputs:
            *ka*: :class:`int`
                Element index of the first element in *comp* (1-based)
            *kb*: :class:`int`
                Element index of the last element in *comp* (1-based)
            *Elems*: :class:`np.ndarray` shape=(*nElem*, *nVert*)
                Array of vertex indices for each element
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
       # --- Find *Elements_t* node ---
        # If given an address, use that
        if comp.__class__.__name__ not in ["str", "unicode"]:
            # Get the single node by address
            k = self.GetNodeByAddress(comp)
        else:
            # Get the node matching this *comp*
            K = self.GetNodeIndex(comp, label="Elements_t")
            # Check for exactly one match
            if len(K) > 1:
                # Multiple matches
                raise ValueError("Multiple *Elements_t* nodes '%s'" % comp)
            elif len(K) == 0:
                # No matches
                raise ValueError("No *Elements_t* nodes '%s'" % comp)
            # Extract one match
            k = K[0]
        # Get data
        elem_t = self.Data[k][0]
        # Process it
        if elem_t == 5:
            # Triangles
            nVert = 3
        elif elem_t == 7:
            # Quads
            nVert = 4
        else:
            # Not implemented!
            raise NotImplementedError("CGNS element type '%s' is not supported"
                % elem_t)
       # --- Process 'ElementRange' *IndexRange_t* node ---
        # Get *IndexRange_t* node
        KER = self.GetSubNodeByLabel(k, "IndexRange_t")
        # Check for nonsense
        if len(KER) == 0:
            # Check if the *next* node is the one that was really meant
            if self.NodeLabels[k+1] == "IndexRange_t":
                # Use the next node; it's probably correct
                KER = [k+1]
        # Must be exactly one
        if len(KER) != 1:
            raise ValueError("Component '%s' has %s *IndexRange_t* children"
                % (comp, len(KER)))
        # One node
        kER = KER[0]
        # Get data, which is the range of elements in this component
        ka, kb = self.Data[kER]
        # Save the number of elements
        nElem = kb - ka + 1
       # --- Process 'ElementConnectivity' *DataArray_t* node ---
        # Get *IndexRange_t* node
        KEC = self.GetSubNodeByName(kER, "ElementConnectivity")
        # Must be exactly one
        if len(KEC) != 1:
            raise ValueError(
                "Component '%s' has %s 'ElementConnectivity' children"
                % (comp, len(KEC)))
        # One node
        kEC = KEC[0]
        # Get the vertex info
        Elems = np.reshape(self.Data[kEC], (nElem, nVert))
        # Output
        return ka, kb, Elems
        
    # Get a node's subnode given a desired type
    def GetSubNodeByName(self, k, name):
        """Get subnode by desired name
        
        :Call:
            >>> J = cgns.GetSubNodeByName(k, name)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *k*: :class:`int`
                Node index (0-based)
            *name*: :class:`str`
                Desired name
        :Outputs:
            *J*: :class:`np.ndarray` (:class:`int`)
                List of such subnodes
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        # Get SubNodeTable
        sntb = self.SubNodeTables[k]
        # Check if empty
        if sntb is None:
            return np.array([], dtype="int")
        # Get the SubNode indices
        I = np.array([self.GetNodeByAddress(sn[1]) for sn in sntb])
        # Get the labels of these nodes
        Names = self.NodeNames[I]
        # Check for a match
        K = np.where(Names == name)[0]
        # Check for no matches
        if len(K) == 0:
            # No post-processing needed
            return K
        else:
            # Back-index to *I*
            return I[K]
        
    # Get a node's subnode given a desired type
    def GetSubNodeByLabel(self, k, label):
        """Get subnode by desired type
        
        :Call:
            >>> J = cgns.GetSubNodeByLabel(k, label)
        :Inputs:
            *cgns*: :class:`cape.cgns.CGNS`
                CGNS file interface
            *k*: :class:`int`
                Node index (0-based)
            *label*: :class:`str`
                Desired label
        :Outputs:
            *J*: :class:`np.ndarray` (:class:`int`)
                List of such subnodes
        :Versions:
            * 2018-03-02 ``@ddalle``: First version
        """
        # Get SubNodeTable
        sntb = self.SubNodeTables[k]
        # Check if empty
        if sntb is None:
            return np.array([], dtype="int")
        # Get the SubNode indices
        I = np.array([self.GetNodeByAddress(sn[1]) for sn in sntb])
        # Get the labels of these nodes
        Labels = self.NodeLabels[I]
        # Check for a match
        K = np.where(Labels == label)[0]
        # Check for no matches
        if len(K) == 0:
            # No post-processing needed
            return K
        else:
            # Back-index to *I*
            return I[K]
  # >
# class CGNS

