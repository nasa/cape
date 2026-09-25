r"""
:mod:`cape.pyvul.inpfile`: Interface to VULCAN-CFD input files
==============================================================

This module provides the class :class:`VulcanInpFile`, which is used to
parse and modify the main VULCAN-CFD input file (usually ``*.inp``).

The file format (see Chapter 8 of the VULCAN-CFD user manual,
NASA/TM-20220008781) consists of lines of the form

    ``KEYWORD  value (optional descriptor)``

where the keyword may contain single blank spaces, but at least two
blank spaces must separate the keyword from its value.  Lines that
begin with ``$`` are comments.  Flag keywords have no value.  Several
keywords expect data on the following (indented) line(s), such as file
names after ``UNS GRID`` or the list of names after ``PLOT FUNCTION``.

The recognized keyword names are hard-coded in :data:`OPTSPEC`.
Values are stored as :class:`float` (or ``None`` for flags), and
continuation lines are accessed with :meth:`VulcanInpFile.get_cont`
and :meth:`VulcanInpFile.set_cont`.

Reading preserves every original line; :meth:`VulcanInpFile.write`
rewrites only the lines whose values were changed, so comments,
descriptors, and unrecognized blocks survive a read/write round trip
byte-for-byte.

Complex blocks are exposed through dedicated interfaces:

    * :class:`BlockConfig` for the ``BLOCK CONFIG.`` table
    * :class:`RegionControls` for the region control specification
      (solver methodology) blocks
    * :class:`BCGroups` for the ``BC GROUPS:`` block
    * :class:`BCObjects` for the ``BC OBJECTS:`` block

:See Also:
    * :mod:`cape.pyvul.cntl`
    * :mod:`cape.optdict`
"""

# Standard library
import os
import re
from typing import Any, List, Optional

# Local imports
from ..optdict import OptionsDict


# Regular expressions
RE_FLOAT = re.compile(r"[+-]?[0-9]+\.?[0-9]*([DEdde][+-]?[0-9]+)?")
RE_INT = re.compile(r"[+-]?[0-9]+")

# Option kinds for OPTSPEC
KIND_NUM = 'num'        # keyword + real value
KIND_FLAG = 'flag'      # keyword only
KIND_NUM_FILE = 'num_file'   # value; name/file on next line(s)
KIND_FLAG_NAME = 'flag_name'  # name/value on next line(s)
KIND_NUM_LIST = 'num_list'    # count of tokens on following line(s)
KIND_NUM_NAMES = 'num_names'  # count of names, one per line

# Mapping of input file keyword -> kind
#
# Names and semantics per Chapter 8 of NASA/TM-20220008781
OPTSPEC = {
    # 8.1 Parallel processing control data
    "PROCESSORS": KIND_NUM,
    "MESSAGE MODE": KIND_NUM,
    # 8.2 Computational domain dimension specification
    "THREED": KIND_FLAG,
    "AXISYM": KIND_FLAG,
    "TWOD": KIND_FLAG,
    "ONED": KIND_FLAG,
    "SUPPRESS U-MOMENTUM": KIND_FLAG,
    "SUPPRESS V-MOMENTUM": KIND_FLAG,
    "SUPPRESS W-MOMENTUM": KIND_FLAG,
    # 8.3 Computational grid input data
    "STR GRID": KIND_NUM_FILE,
    "GRID": KIND_NUM_FILE,
    "STR GRID FORMAT": KIND_NUM,
    "GRID FORMAT": KIND_NUM,
    "UNS GRID": KIND_NUM_FILE,
    "USE TINF GRAPH PARTITIONER": KIND_FLAG,
    "USE TINF CACHE REORDER": KIND_FLAG,
    "USE TINF RANDOM REORDER": KIND_FLAG,
    "USE TINF Q REORDER": KIND_NUM,
    "GRID SCALING FACTOR": KIND_NUM,
    "C(0) TOLERANCE": KIND_NUM,
    "SYMMETRY TOLERANCE": KIND_NUM,
    "PROFILE XYZ TOLERANCE": KIND_NUM,
    "AXISYM ANGLE": KIND_NUM,
    "K-AXIS ORIENTATION": KIND_NUM,
    # 8.4 Simulation initialization and restart control data
    "RESTART IN": KIND_FLAG_NAME,
    "RESTART OUT": KIND_NUM_FILE,
    "USE TINF RESTART FILE": KIND_NUM,
    "RESTART ON GRID LEVEL": KIND_NUM,
    "IGNORE 1ST-ORDER RESTART CYCLES": KIND_FLAG,
    "RE-INITIALIZE BLOCKS": KIND_NUM_LIST,
    "INITIAL VELOCITY SCALE FACTOR": KIND_NUM,
    "BLOCKS TO SCALE VELOCITY": KIND_NUM_LIST,
    "INIT. BLENDING FACTOR": KIND_NUM,
    "INIT. BOUNDARY LAYER THICKNESS": KIND_NUM,
    # 8.5 Output control data
    "CGNS OUTPUT": KIND_NUM,
    "TECPLOT OUTPUT": KIND_NUM,
    "VTK OUTPUT": KIND_NUM,
    "PLOT3D OUTPUT": KIND_NUM,
    "PLOT PLANAR": KIND_NUM,
    "PLOT FUNCTION": KIND_NUM_NAMES,
    "32 BIT BINARY": KIND_FLAG,
    "64 BIT BINARY": KIND_FLAG,
    "EXCLUDE SLIP WALLS": KIND_FLAG,
    "OUTPUT FLUXES": KIND_FLAG,
    "UPDATE TIME AVERAGE": KIND_NUM,
    "OUTPUT H.O.T.": KIND_NUM,
    "OUTPUT S.G.T.": KIND_NUM,
    "OUTPUT TIME HISTORY": KIND_NUM,
    "COMPUTE TOTAL PRESSURE LOSS": KIND_FLAG,
    "COMPUTE COMBUSTION EFFICIENCY": KIND_FLAG,
    "COMPUTE MIXING EFFICIENCY": KIND_FLAG,
    "OXIDIZER SPECIES INDEX": KIND_NUM,
    "PURE OXIDIZER FRACTION": KIND_NUM,
    "FUEL SPECIES INDEX": KIND_NUM,
    "PURE FUEL FRACTION": KIND_NUM,
    "STOICHIOMETRIC F/O MASS RATIO": KIND_NUM,
    "COMPUTE FUEL PENETRATION": KIND_NUM,
    "WARNING MESSAGES": KIND_NUM,
    "QUIET OUTPUT": KIND_FLAG,
    # 8.6 Global solver input data
    "GLOBAL EULER": KIND_FLAG,
    "GLOBAL VISCOUS": KIND_FLAG,
    "HEXAHEDRAL GREEN-GAUSS GRADIENT": KIND_FLAG,
    "OCTAHEDRAL GREEN-GAUSS GRADIENT": KIND_FLAG,
    "ENABLE TOPOLOGY ANALYSIS": KIND_FLAG,
    "FORCE TOPOLOGY ANALYSIS": KIND_FLAG,
    "PERIODIC BODY FORCE": KIND_NUM,
    "AVERAGE SHEAR STRESS": KIND_NUM,
    "UPWIND ADVECTION VARIABLES": KIND_NUM,
    "LDFSS SHOCK CONSTANT": KIND_NUM,
    "APPROX ROE INVISCID JACOBIAN": KIND_NUM,
    "DDT ROE INVISCID JACOBIAN": KIND_NUM,
    "DDT INVISCID JACOBIAN": KIND_NUM,
    "ENABLE TIME STEP REDUCTION LOGIC": KIND_NUM,
    "RESET TIME STEP REDUCTION LOGIC": KIND_NUM,
    "UNS ADAPTIVE CFL METHOD": KIND_NUM,
    "CELL SKEW CFL LIMITER": KIND_NUM,
    "UNS VOLUME CENTER METHOD": KIND_NUM,
    "UNS CELL AVG GRADIENT METHOD": KIND_NUM,
    "UNS NODE CENTERED GRADIENTS": KIND_NUM,
    "UNS LEAST-SQUARES WEIGHTS": KIND_NUM,
    "UNS LSQ COEF METHOD": KIND_NUM,
    "UNS VISCOUS GRADIENT METHOD": KIND_NUM,
    "UNS ALPHA DAMPING FACTOR": KIND_NUM,
    "UNS SOURCE GRADIENT METHOD": KIND_NUM,
    "UNS SPECTRAL RADIUS VISCOUS JACOBIAN": KIND_FLAG,
    "UNS TLNS VISCOUS JACOBIAN": KIND_NUM,
    "UNS DDT VISCOUS JACOBIAN": KIND_NUM,
    "UNS SMOOTH LIMITER COEFFICIENT": KIND_NUM,
    "UMUSCL LIMITER CONVERGENCE FREEZING CRITERIA": KIND_NUM,
    "UMUSCL LIMITER ITER NUMBER FREEZING CRITERIA": KIND_NUM,
    "UNSTRUCTURED MAX TMP MEMORY": KIND_NUM,
    "NSTAGE": KIND_NUM_LIST,
    "TVD 3-STAGE RK": KIND_FLAG,
    "HEUN 3-STAGE RK": KIND_FLAG,
    "CAR-KEN 5-STAGE RK": KIND_FLAG,
    "ENABLE HYBRID WALL MATCHING": KIND_NUM,
    "DISABLE OUTFLOW BC RULE": KIND_FLAG,
    "RIEMANN STAGNATION BC": KIND_FLAG,
    "STAGNATION BC RELAX ITERATIONS": KIND_NUM,
    # 8.7 Hybrid advection input specification
    "HYBRID ADVECTION SCHEME": KIND_NUM,
    "HYBRID ADVECTION CD ORDER": KIND_NUM,
    "HYBRID ADVECTION SENSOR": KIND_NUM,
    "HYBRID ADVECTION SENSOR LOWER LIMIT": KIND_NUM,
    "HYBRID ADVECTION SENSOR UPPER LIMIT": KIND_NUM,
    "HYBRID ADVECTION CONTINUITY LOWER LIMIT": KIND_NUM,
    "HYBRID ADVECTION MOMENTUM LOWER LIMIT": KIND_NUM,
    "HYBRID ADVECTION ENERGY LOWER LIMIT": KIND_NUM,
    "HYBRID ADVECTION TURBULENCE LOWER LIMIT": KIND_NUM,
    "HYBRID ADVECTION VORTICITY SCALE COEF": KIND_NUM,
    "HYBRID ADVECTION BACKGROUND SCALE COEF": KIND_NUM,
    "HYBRID ADVECTION SENSOR FREEZING CRITERIA": KIND_NUM,
    # 8.8 Thermodynamic model specification
    "GAS/THERMO MODEL": KIND_NUM,
    "UNIVERSAL GAS CONSTANT": KIND_NUM,
    "MIN. STATIC TEMPERATURE": KIND_NUM,
    "MAX. STATIC TEMPERATURE": KIND_NUM,
    "MIN. STATIC TEMP": KIND_NUM,
    "MAX. STATIC TEMP": KIND_NUM,
    "T-V EXCHANGE MODEL": KIND_NUM,
    # 8.9 Transport model specification
    "SPECIES VISCOSITY MODEL": KIND_NUM,
    "MIXTURE VISCOSITY MODEL": KIND_NUM,
    "VISCOSITY MODEL": KIND_NUM,
    "SPECIES CONDUCTIVITY MODEL": KIND_NUM,
    "MIXTURE CONDUCTIVITY MODEL": KIND_NUM,
    "CONDUCTIVITY MODEL": KIND_NUM,
    "SPECIES DIFFUSION MODEL": KIND_NUM,
    # 8.10 Chemical constituent data
    "NO. OF CHEMICAL SPECIES": KIND_NUM_LIST,
    "SPECIES THERMODYNAMIC DATA FORMAT": KIND_NUM_FILE,
    "SPECIES TRANSPORT DATA FORMAT": KIND_NUM_FILE,
    "SPECIES NONEQUILIBRIUM THERMODYNAMIC DATA": KIND_FLAG_NAME,
    # 8.11 Chemistry model specification
    "CHEMISTRY MODEL": KIND_NUM,
    "NO. OF CHEMICAL REACTIONS": KIND_NUM_FILE,
    "ISAT PARAMETERS": KIND_NUM_LIST,
    "MAX. TREE NODES": KIND_NUM,
    "ERROR TOLERANCE": KIND_NUM,
    "CHEMICAL TIME SCALE": KIND_NUM,
    "PARK RATE EXPONENT": KIND_NUM,
    "PREFERENTIAL DISSOCIATION CONSTANT": KIND_NUM,
    "IMPLICIT CHEMISTRY": KIND_NUM,
    "EXPLICIT CHEMISTRY": KIND_FLAG,
    # 8.12 Reference frame specification
    "ANGLE REF. FRAME": KIND_NUM,
    "ALPHA": KIND_NUM,
    "BETA": KIND_NUM,
    "DIRECTION COSINE METHOD": KIND_FLAG,
    "DIR COS(U)": KIND_NUM,
    "DIR COS(V)": KIND_NUM,
    "DIR COS(W)": KIND_NUM,
    "INTEGRATION REF. PRESSURE": KIND_NUM,
    "MOMENT REF. X": KIND_NUM,
    "MOMENT REF. Y": KIND_NUM,
    "MOMENT REF. Z": KIND_NUM,
    "INTEGRATION REF. LENGTH": KIND_NUM,
    "INTEGRATION REF. AREA": KIND_NUM,
    "INTEGRATION X SCALE FACTOR": KIND_NUM,
    "INTEGRATION Y SCALE FACTOR": KIND_NUM,
    "INTEGRATION Z SCALE FACTOR": KIND_NUM,
    # 8.13 Reference condition specification
    "NONDIM": KIND_NUM,
    "SPECIFIC HEAT RATIO": KIND_NUM,
    "GAMMA": KIND_NUM,
    "MACH NO.": KIND_NUM,
    "GAS CONSTANT": KIND_NUM,
    "STATIC PRESSURE": KIND_NUM,
    "STATIC TEMPERATURE": KIND_NUM,
    "STATIC DENSITY": KIND_NUM,
    "TOTAL PRESSURE": KIND_NUM,
    "TOTAL TEMPERATURE": KIND_NUM,
    "TOTAL DENSITY": KIND_NUM,
    "UNIT REYNOLDS NO.": KIND_NUM,
    "POWER LAW MU0": KIND_NUM,
    "POWER LAW T0": KIND_NUM,
    "POWER LAW EXP": KIND_NUM,
    "SUTHERLANDS LAW MU0": KIND_NUM,
    "SUTHERLANDS LAW T0": KIND_NUM,
    "SUTHERLANDS LAW S0": KIND_NUM,
    "PRANDTL NO.": KIND_NUM,
    "SCHMIDT NO.": KIND_NUM,
    "VIBRATIONAL TEMPERATURE": KIND_NUM,
    "MCBRIDE POLYNOMIAL": KIND_NUM_LIST,
    # 8.14 RAS turbulence model specification
    "TURBULENCE MODEL": KIND_FLAG_NAME,
    "STRAIN-BASED PRODUCTION": KIND_FLAG,
    "VORTICITY-BASED PRODUCTION": KIND_FLAG,
    "STRAIN/VORTICITY-BASED PRODUCTION": KIND_FLAG,
    "QCR-2000": KIND_FLAG,
    "QCR-2013": KIND_FLAG,
    "QCR-2013-V": KIND_FLAG,
    "THIVET REALIZABILITY": KIND_NUM,
    "DURBIN REALIZABILITY": KIND_NUM,
    "TURB. INTENSITY": KIND_NUM,
    "TURB. VISC. RATIO": KIND_NUM,
    "TURB. PRANDTL NO.": KIND_NUM,
    "TURB. SCHMIDT NO.": KIND_NUM,
    "INIT. MIN. DIST.": KIND_FLAG,
    "EXCLUDE SPALART FT2 TERM": KIND_FLAG,
    "SPALART SHAT LIMITER 1B": KIND_FLAG,
    "SPALART SHAT LIMITER 1C": KIND_FLAG,
    "NO 2/3 RHOK": KIND_FLAG,
    "THIN LAYER SOURCE TERMS": KIND_FLAG,
    "ENABLE WILCOX LOW RE TERMS": KIND_FLAG,
    "DISABLE WILCOX POPE TERM": KIND_FLAG,
    "MAX. ALLOWABLE MUT/MU": KIND_NUM,
    "KARMAN CONSTANT": KIND_NUM,
    "EDDY VISCOSITY CONSTANT": KIND_NUM,
    "TKE DIFFUSION CONSTANT": KIND_NUM,
    "TKE DIFFUSION CONSTANT (OMEGA)": KIND_NUM,
    "TKE DIFFUSION CONSTANT (EPSILON)": KIND_NUM,
    "OMEGA DIFFUSION CONSTANT": KIND_NUM,
    "EPSILON DIFFUSION CONSTANT": KIND_NUM,
    "OMEGA DESTRUCTION CONSTANT": KIND_NUM,
    "EPSILON DESTRUCTION CONSTANT": KIND_NUM,
    "MENTER-SST C LIM": KIND_NUM,
    "WILCOX-2006 C LIM": KIND_NUM,
    "SPALART CV1 CONSTANT": KIND_NUM,
    "SPALART CB1 CONSTANT": KIND_NUM,
    "SPALART CB2 CONSTANT": KIND_NUM,
    "SPALART CW2 CONSTANT": KIND_NUM,
    "SPALART CW3 CONSTANT": KIND_NUM,
    "SPALART DIFFUSION CONSTANT": KIND_NUM,
    # 8.15 RAS turbulence chemistry model specification
    "TURBULENCE CHEMISTRY MODEL": KIND_FLAG_NAME,
    "1ST EDC CONSTANT": KIND_NUM,
    "2ND EDC CONSTANT": KIND_NUM,
    "EDC LIMIT": KIND_NUM,
    # 8.16 LES turbulence model specification
    "SGS VISCOSITY CONSTANT": KIND_NUM,
    "VREMAN COMPRESSIBILITY CONSTANT": KIND_NUM,
    "GERMANO DYNAMIC SGS": KIND_FLAG,
    "HEINZ DYNAMIC SGS": KIND_FLAG,
    "SMOOTH DYNAMIC SGS": KIND_NUM,
    "DO NOT CLIP DYNAMIC SGS": KIND_FLAG,
    "SPATIALLY AVERAGE DYNAMIC SGS": KIND_FLAG,
    "HOMOGENEOUS-I": KIND_FLAG,
    "HOMOGENEOUS-J": KIND_FLAG,
    "HOMOGENEOUS-K": KIND_FLAG,
    "ISOTROPIC FILTER WIDTH": KIND_FLAG,
    "IDDES FILTER WIDTH": KIND_FLAG,
    "TIME SCALE SGS": KIND_FLAG,
    # 8.17 Hybrid RAS/LES specification
    "HYBRID-LES": KIND_NUM,
    "HYBRID-LES SGS MODEL": KIND_FLAG_NAME,
    "HYBRID-LES ALPHA": KIND_NUM,
    "HYBRID-LES CDES": KIND_NUM,
    # 8.18 Inflow recycling/rescaling control data
    "RECYCLE INFLOW": KIND_NUM,
    "BOUNDARY LAYER CELLS": KIND_NUM,
    "INFLOW BLOCK CORNER": KIND_FLAG,
    "RECYCLE BLOCK CORNER": KIND_FLAG,
    "INFLOW BL THICKNESS": KIND_NUM,
    "BL THICKNESS RATIO": KIND_NUM,
    "SKIN FRICTION RATIO": KIND_NUM,
    "FRICTION VELOCITY RATIO": KIND_NUM,
    "FREESTREAM VELOCITY": KIND_NUM,
    "MAX. INTEGRATION TIME": KIND_NUM,
    # 8.19 Structured grid adaptation control data
    "USE STR GRID ADAPTION": KIND_NUM,
    "ADAPTATION ON ITERATION": KIND_NUM,
    "ADAPTATION NUMBER": KIND_NUM,
    "ADAPTATION OFF ITERATION": KIND_NUM,
    "ADAPTATION FREQUENCY": KIND_NUM,
    "ADAPTATION L2 STOP": KIND_NUM,
    "SHOCK ADAPTATION": KIND_NUM,
    "SHOCK DETECTION COEF": KIND_NUM,
    "SPACING AT SHOCK": KIND_NUM,
    "FREESTREAM BOUNDARY": KIND_NUM,
    "SHOCK SMOOTHING COEF": KIND_NUM,
    "SHOCK TO FS BUFFER": KIND_NUM,
    "WALL ADAPTATION": KIND_NUM,
    "WALL PARAMETER": KIND_NUM,
    "WALL SMOOTHING COEF": KIND_NUM,
    # 8.20 Block, subblock, and block boundary specifications
    "CURV BLOCKS": KIND_NUM,
    "BLOCKS": KIND_NUM,
    "UNST BLOCKS": KIND_NUM,
    "BCGROUPS": KIND_NUM,
    "BCOBJECTS": KIND_NUM,
    "INIT. SUB-DOMAINS": KIND_NUM,
    "LAMINAR SUB-DOMAINS": KIND_NUM,
    "IGNITION SUB-DOMAINS": KIND_NUM,
    "TIME SUB-DOMAINS": KIND_NUM,
    "FLOWBCS": KIND_NUM,
    "CUTBCS": KIND_NUM,
    "PATCHBCS": KIND_NUM,
    "PATCH FILE": KIND_FLAG_NAME,
    "LAMINAR SUB-BLOCKS": KIND_NUM,
    "IGNITION SUB-BLOCKS": KIND_NUM,
    "TIME HISTORY I/O": KIND_NUM,
    "BLOCK CONFIG": KIND_NUM,
    "BLOCK CONFIG.": KIND_NUM,
}

# Alternate names for user convenience (alias -> keyword as stored)
OPTMAP = {
    "GRID": "STR GRID",
    "GRID FORMAT": "STR GRID FORMAT",
    "BLOCKS": "CURV BLOCKS",
}

# Known boundary condition types (Chapters 9+) used to identify BC group
# lines. Values are the "TYPE" column of a BC GROUPS: entry.
BC_TYPES = [
    "P0 T0 IN",
    "MF T0 IN",
    "FIX IN",
    "REF IN",
    "CHAR REF",
    "CHAR FIX",
    "PRES OUT",
    "MIXED OUT",
    "MDOT OUT",
    "REG OUT2",
    "REG OUT",
    "EXTRAP2",
    "EXTRAP",
    "AWALL",
    "AWALLM",
    "IWALL",
    "IWALLM",
    "TCWALL",
    "TCWALLM",
    "REWALL",
    "REWALLM",
    "LWBLOW",
    "EWALL",
    "SWALL",
    "SYMM AXI",
    "SYMM X",
    "SYMM Y",
    "SYMM Z",
    "SYMM",
    "SINGULAR",
    "NOSLPSNG",
    "C-PERIODIC",
    "P-PERIODIC",
]

# Standard column names for region control rows, keyed by first header
# word where possible. See Section 8.21 of NASA/TM-20220008781.
COLS_KAPPA = [
    'KAPPA', 'LIMITER', 'LIM COEFF', 'FLUX SCHEME',
    'ENT FIX (U)', 'ENT FIX (U+a)',
]
COLS_FMG6 = [
    'FMG-LVLS', 'NITSF', '1ST-ORDER LEVELS',
    'REL RES', 'ABS RES', 'DQ COEF',
]
COLS_FMG7 = [
    'FMG-LVLS', 'NITSC', 'NITSF', '1ST-ORDER LEVELS',
    'REL RES', 'ABS RES', 'DQ COEF',
]
COLS_TURB = [
    'TURB CONV', 'DT RATIO', 'NON EQUIL',
    'POINT IMP', 'COMP MODEL', 'CG WALL BC',
]
COLS_SCHEME = [
    'SCHEME', 'T-STEP', 'IT-STATS', 'MIN-CFL', 'VAR-CFL',
    'CFL-VALS', 'VIS-DT', 'IMP-BC', 'REG-RES',
]
COLS_SM = [
    'SM-ORD', 'BEG:END', 'MG BEG:END', 'VIG COEF',
    'MEAN-LIM', 'TURB-LIM', 'SUB-STEP',
]
COLS_MG = [
    'MG CYCLE', 'CRS LVLS', 'DQ SMTH',
    'DQ COEF DAMP', 'MEAN DAMP', 'TURB DAMP',
]


# --- Utility functions ---
def to_number(txt: str):
    r"""Convert one VULCAN input token to a number if possible

    Handles Fortran-style ``D`` exponents and leading ``+``.

    :Call:
        >>> v = to_number(txt)
    :Inputs:
        *txt*: :class:`str`
            Single whitespace-delineated token
    :Outputs:
        *v*: :class:`float` | :class:`str`
            Numeric value if *txt* is a number, else *txt*
    """
    if RE_FLOAT.fullmatch(txt):
        return float(txt.replace("D", "E").replace("d", "e"))
    return txt


def is_number(txt: str) -> bool:
    r"""Check if one token parses as a VULCAN real value

    :Call:
        >>> q = is_number(txt)
    :Inputs:
        *txt*: :class:`str`
            Single whitespace-delineated token
    :Outputs:
        *q*: :class:`bool`
            ``True`` if *txt* is numeric
    """
    return RE_FLOAT.fullmatch(txt) is not None


def fmt_number(v: Any) -> str:
    r"""Format a value for writing back to a VULCAN input file

    :Call:
        >>> txt = fmt_number(v)
    :Inputs:
        *v*: :class:`object`
            Value to format
    :Outputs:
        *txt*: :class:`str`
            Text representation
    """
    if v is None:
        return ''
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        txt = "%g" % v
        if '.' not in txt and 'e' not in txt and 'n' not in txt:
            txt += '.0'
        return txt
    return str(v)


def match_bc_type(tokens: List[str], start: int = 1):
    r"""Match a known BC TYPE from token list starting at *start*

    :Call:
        >>> typ, ntok = match_bc_type(tokens, start=1)
    :Inputs:
        *tokens*: :class:`list`\ [:class:`str`]
            Whitespace-split BC group line
        *start*: {``1``} | :class:`int`
            Index at which the TYPE column begins
    :Outputs:
        *typ*: :class:`str` | ``None``
            Matched type name, if any
        *ntok*: :class:`int`
            Number of tokens consumed by *typ*
    """
    for tt in BC_TYPES:
        nw = len(tt.split())
        if len(tokens) >= start + nw:
            joined = ' '.join(tokens[start:start+nw])
            if joined == tt or joined.replace('_', ' ') == tt:
                return joined, nw
        if nw > 1 and len(tokens) > start:
            single = tokens[start]
            if single.replace('_', ' ') == tt:
                return single, 1
    return None, 0


def region_columns(header: str) -> List[str]:
    r"""Get standard column names for a region control header line

    :Call:
        >>> cols = region_columns(header)
    :Inputs:
        *header*: :class:`str`
            Header line of a region control row
    :Outputs:
        *cols*: :class:`list`\ [:class:`str`]
            Column names to pair with data line tokens
    """
    w = header.split()
    if not w:
        return []
    f0 = w[0]
    if f0.startswith('SM'):
        return list(COLS_SM)
    if f0 == 'MG':
        return list(COLS_MG)
    if f0.startswith('FMG'):
        return list(COLS_FMG7) if 'NITSC' in w else list(COLS_FMG6)
    if f0 == 'KAPPA':
        cols = list(COLS_KAPPA)
        if any(ww.startswith('HYB') for ww in w):
            cols.append('HYB-ADV')
        return cols
    if f0 == 'TURB':
        return list(COLS_TURB)
    if f0 == 'SCHEME':
        return list(COLS_SCHEME)
    if f0 == 'TIME':
        cols = ['TIME STEP', 'SUB-ITS', 'RES-RED', 'METHOD']
        if any(ww in ('C-N', 'RELAX') for ww in w):
            cols.append('RELAX')
        return cols
    if f0 in ('SGS', 'ILU', 'SOR', 'SSOR', 'DAF'):
        # First token repeats scheme name from the SCHEME row
        return w[1:]
    return w


# --- Region control specification ---
class RegionRow(OptionsDict):
    r"""One header/data row pair of a region control specification

    If the number of data tokens matches the (standardized) column
    names from the header line, values are stored by column name.
    Otherwise, values are stored as a list under the ``'_values'``
    key.  Additional unlabeled data lines (e.g. a CFL schedule) are
    stored as a list of lists in :py:attr:`extras`.

    :Call:
        >>> row = RegionRow()
    :Slots:
        *row.header*: :class:`str`
            Raw header line
        *row.columns*: :class:`list`\ [:class:`str`]
            Standardized column names
        *row.extras*: :class:`list`\ [:class:`list`]
            Unlabeled data lines as lists of values
    """
    __slots__ = (
        "header",
        "columns",
        "extras",
        "_rawlines",
        "_dirty",
    )

    def __init__(self, *args, **kw):
        super().__init__()
        self.header = ''
        self.columns = []
        self.extras = []
        self._rawlines = []
        self._dirty = False
        if args or kw:
            self.update(*args, **kw)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        self._dirty = True

    def add_extra(self, values: List[Any]):
        r"""Append one unlabeled data line

        :Call:
            >>> row.add_extra(values)
        :Inputs:
            *row*: :class:`RegionRow`
                Region control row
            *values*: :class:`list`
                Values for the line
        """
        self.extras.append(list(values))
        self._dirty = True

    def data_values(self) -> List[Any]:
        r"""Get values of the main data line in column order

        :Call:
            >>> vals = row.data_values()
        :Outputs:
            *vals*: :class:`list`
                Values in column order (or ``'_values'``)
        """
        if '_values' in self:
            return list(self['_values'])
        return [self.get(col) for col in self.columns]

    def to_lines(self) -> List[str]:
        r"""Convert row back to file lines

        :Call:
            >>> lines = row.to_lines()
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                Header, data, and extras lines
        """
        if not self._dirty and self._rawlines:
            return list(self._rawlines)
        out = [self.header]
        vals = [fmt_number(v) for v in self.data_values()]
        out.append('  ' + '  '.join(vals))
        for xi in self.extras:
            out.append(' ' + '  '.join(fmt_number(v) for v in xi))
        return out


class RegionControl(OptionsDict):
    r"""Region control specification for one region (Section 8.21)

    Rows are stored in file order keyed by the first token of each
    header line, e.g. ``'SOLVER/STATUS'``, ``'KAPPA'``,
    ``'FMG-LVLS'``, ``'TURB'``, ``'SCHEME'``, or the implicit-scheme
    name (``'SGS'``, ``'ILU'``, ...).

    :Call:
        >>> reg = RegionControl()
    """
    __slots__ = ()

    def to_lines(self) -> List[str]:
        r"""Convert region back to file lines

        :Call:
            >>> lines = reg.to_lines()
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                All header/data line pairs
        """
        out = []
        for row in self.values():
            out.extend(row.to_lines())
        return out


class RegionControls(OptionsDict):
    r"""Collection of region control specifications by region number

    :Call:
        >>> regs = RegionControls()
    """
    __slots__ = ()

    def _read(self, blocks: List[List[str]]):
        r"""Parse raw line blocks, one per region

        :Call:
            >>> regs._read(blocks)
        :Inputs:
            *blocks*: :class:`list`\ [:class:`list`\ [:class:`str`]]
                Lines for each region, starting with SOLVER/STATUS
        """
        for k, blk in enumerate(blocks, start=1):
            self[k] = self._read_one(blk)

    def _read_one(self, blk: List[str]) -> RegionControl:
        r"""Parse lines for one region

        :Call:
            >>> reg = regs._read_one(blk)
        """
        reg = RegionControl()
        header = None
        rawlines = []
        row = None
        for line in blk:
            if not line.strip():
                if row is not None:
                    rawlines.append(line)
                continue
            tokens = line.split()
            # A line with multiple non-numeric tokens is a header line
            # even when indented, e.g. an implicit-scheme control line
            isheader = (
                (not line[:1].isspace()) or
                (len(tokens) > 1 and
                 not any(is_number(t) for t in tokens))
            )
            if isheader:
                # New header line
                if row is not None:
                    row._rawlines = list(rawlines)
                    row._dirty = False
                header = line
                cols = region_columns(header.strip())
                row = RegionRow()
                row.header = header
                row.columns = cols
                reg[tokens[0]] = row
                rawlines = [line]
            else:
                # Data line for current row
                rawlines.append(line)
                if row is None:
                    continue
                vals = [to_number(ww) for ww in line.split()]
                if len(row):
                    # Already have main data line; this is an extra
                    row.extras.append(vals)
                elif len(vals) == len(row.columns):
                    for ci, col in enumerate(row.columns):
                        dict.__setitem__(row, col, vals[ci])
                else:
                    dict.__setitem__(row, '_values', vals)
        if row is not None:
            row._rawlines = list(rawlines)
            row._dirty = False
        return reg

    def to_lines(self, i: Optional[int] = None) -> List[str]:
        r"""Convert to file lines for one region or all

        :Call:
            >>> lines = regs.to_lines(i=None)
        :Inputs:
            *i*: {``None``} | :class:`int`
                Region number; all regions if ``None``
        """
        if i is None:
            out = []
            for reg in self.values():
                out.extend(reg.to_lines())
            return out
        return self[i].to_lines()


# --- Block configuration table ---
class BlockConfigRow(OptionsDict):
    r"""One line of the ``BLOCK CONFIG.`` table

    :Call:
        >>> row = BlockConfigRow()
    """
    __slots__ = (
        "_rawline",
        "_dirty",
    )

    _optlist = {"VISC", "TURB", "REAC", "REGION"}
    _opttypes = {
        "VISC": (str, type(None)),
        "TURB": (str, type(None)),
        "REAC": (str, type(None)),
        "REGION": (str, type(None)),
    }

    def __init__(self, *args, **kw):
        super().__init__()
        self._rawline = None
        self._dirty = False
        if args or kw:
            self.update(*args, **kw)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        self._dirty = True

    def to_text(self, positions: dict, blk: int = None) -> str:
        r"""Convert row back to a line using header column positions

        :Call:
            >>> line = row.to_text(positions, blk=None)
        :Inputs:
            *positions*: :class:`dict`
                Mapping of column name -> start index in header line
            *blk*: {``None``} | :class:`int`
                Block number for the ``BLK`` column
        """
        if not self._dirty and self._rawline is not None:
            return self._rawline
        cols = sorted(positions, key=lambda cc: positions[cc])
        line = ''
        for col in cols:
            if col == 'BLK':
                if blk is None:
                    continue
                v = str(blk)
            else:
                v = self.get(col)
            if v is None or v == '':
                continue
            target = positions[col]
            if len(line) < target:
                line = line[:target].ljust(target) + str(v)
            else:
                line = f"{line}  {v}"
        return line


class BlockConfig(OptionsDict):
    r"""Interface to the ``BLOCK CONFIG.`` table (Section 8.20)

    Rows are keyed by integer block number (``0`` applies to all
    blocks).

    :Call:
        >>> bc = BlockConfig()
    :Slots:
        *bc.header*: :class:`str`
            Raw header line
    """
    __slots__ = (
        "header",
        "_positions",
    )

    def __init__(self, *args, **kw):
        super().__init__()
        self.header = None
        self._positions = {}

    def _read(self, lines: List[str]):
        r"""Parse header plus data lines

        :Call:
            >>> bc._read(lines)
        """
        if not lines:
            return
        self.header = lines[0]
        colnames = ('BLK', 'VISC', 'TURB', 'REAC', 'REGION')
        self._positions = {}
        for col in colnames:
            ip = self.header.find(col)
            if ip >= 0:
                self._positions[col] = ip
        starts = sorted(self._positions.values())
        for line in lines[1:]:
            if not line.strip():
                continue
            vals = {}
            for col, ip in self._positions.items():
                # Find end of this column slice
                nxt = [ss for ss in starts if ss > ip]
                ep = min(nxt) if nxt else len(line)
                vals[col] = line[ip:ep].strip() or None
            blkstr = vals.pop('BLK', None)
            if blkstr is None or not RE_INT.fullmatch(blkstr):
                continue
            row = BlockConfigRow(**vals)
            row._rawline = line
            dict.__setitem__(self, int(blkstr), row)

    def to_lines(self) -> List[str]:
        r"""Convert table back to file lines

        :Call:
            >>> lines = bc.to_lines()
        """
        if self.header is None:
            return []
        out = [self.header]
        for blk in sorted(self):
            out.append(self[blk].to_text(self._positions, blk))
        return out


# --- BC groups block ---
class BCGroupOptions(OptionsDict):
    r"""Options for one ``BC GROUPS:`` entry

    :Call:
        >>> grp = BCGroupOptions(name, TYPE, OPTIONS, BL_delta)
    :Inputs:
        *name*: :class:`str`
            Group name (12 characters max in VULCAN)
    :Slots:
        *grp.name*: :class:`str`
            Name of the group
    """
    __slots__ = (
        "name",
        "_rawlines",
        "_dirty",
    )

    _optlist = {"TYPE", "OPTIONS", "BL_delta"}
    _opttypes = {
        "TYPE": str,
        "OPTIONS": list,
        "BL_delta": (float, int, type(None)),
    }
    _optlistdepth = {"OPTIONS": 1}

    def __init__(self, name: str = "", *args, **kw):
        super().__init__()
        self.name = name
        self._rawlines = []
        self._dirty = False
        dict.__setitem__(self, "TYPE", None)
        dict.__setitem__(self, "OPTIONS", [])
        dict.__setitem__(self, "BL_delta", None)
        if args or kw:
            self.update(*args, **kw)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        self._dirty = True

    def to_lines(self) -> List[str]:
        r"""Convert group back to file lines

        :Call:
            >>> lines = grp.to_lines()
        """
        if not self._dirty and self._rawlines:
            return list(self._rawlines)
        if self._rawlines:
            raw0 = self._rawlines[0]
            indent = raw0[:len(raw0) - len(raw0.lstrip())]
        else:
            indent = ' '*13
        parts = [self.name, self.get("TYPE") or '']
        parts.extend(self.get("OPTIONS") or [])
        bl = self.get("BL_delta")
        if bl is not None:
            parts.append(fmt_number(bl))
        out = [indent + '  '.join(parts)]
        # Auxiliary lines (if any) are preserved verbatim
        if self._rawlines:
            out.extend(self._rawlines[1:])
        return out


class BCGroups(OptionsDict):
    r"""Interface to the ``BC GROUPS:`` block (Chapter 9)

    Entries are :class:`BCGroupOptions` keyed by group name. The
    ``BL_delta`` column can be set by name, and groups can be looked
    up by TYPE or option.

    :Call:
        >>> bcg = BCGroups()
    """
    __slots__ = (
        "_header_lines",
    )

    def __init__(self, *args, **kw):
        super().__init__()
        self._header_lines = []

    def _read(self, lines: List[str]):
        r"""Parse the BC GROUPS block

        :Call:
            >>> bcg._read(lines)
        """
        if not lines:
            return
        # Header line(s): up to the first group
        ig = 0
        for ig, line in enumerate(lines[1:], start=1):
            if line.strip():
                break
        else:
            ig = len(lines)
        self._header_lines = lines[:ig]
        cur = None
        for line in lines[ig:]:
            tokens = line.split()
            typ, nw = (None, 0)
            if len(tokens) >= 2 and not is_number(tokens[0]):
                typ, nw = match_bc_type(tokens, 1)
            if typ is not None:
                opts = list(tokens[1+nw:])
                bl = None
                if opts and is_number(opts[-1]):
                    bl = to_number(opts[-1])
                    opts = opts[:-1]
                cur = BCGroupOptions(tokens[0])
                dict.__setitem__(cur, "TYPE", typ)
                dict.__setitem__(cur, "OPTIONS", opts)
                dict.__setitem__(cur, "BL_delta", bl)
                cur._rawlines = [line]
                dict.__setitem__(self, tokens[0], cur)
            elif cur is not None:
                # Auxiliary line (profile name, state data, blank, ...)
                cur._rawlines.append(line)

    def add_group(self, name: str, typ: str, options=None, bl=None):
        r"""Add or replace a BC group

        :Call:
            >>> bcg.add_group(name, typ, options=None, bl=None)
        :Inputs:
            *name*: :class:`str`
                Name of group
            *typ*: :class:`str`
                BC TYPE column
            *options*: {``None``} | :class:`list`\ [:class:`str`]
                OPTION column entries
            *bl*: {``None``} | :class:`float`
                Boundary layer thickness ``BL_delta`` column
        """
        grp = BCGroupOptions(name, TYPE=typ, OPTIONS=list(options or []),
                             BL_delta=bl)
        dict.__setitem__(self, name, grp)
        return grp

    def find_type(self, typ: str) -> List[str]:
        r"""Get names of all groups with a given TYPE

        :Call:
            >>> names = bcg.find_type(typ)
        """
        return [nn for nn, gg in self.items() if gg.get("TYPE") == typ]

    def find_option(self, opt: str) -> List[str]:
        r"""Get names of all groups with a given OPTION

        :Call:
            >>> names = bcg.find_option(opt)
        """
        return [
            nn for nn, gg in self.items() if opt in (gg.get("OPTIONS") or [])
        ]

    def set_bl_delta(self, name: str, bl: float):
        r"""Set the ``BL_delta`` column of one group

        :Call:
            >>> bcg.set_bl_delta(name, bl)
        """
        self[name]["BL_delta"] = bl

    def to_lines(self) -> List[str]:
        r"""Convert block back to file lines

        :Call:
            >>> lines = bcg.to_lines()
        """
        out = list(self._header_lines)
        for grp in self.values():
            out.extend(grp.to_lines())
        return out


# --- BC objects block ---
class BCObjects(OptionsDict):
    r"""Interface to the ``BC OBJECTS:`` block (Section 8.20)

    Members are stored as a :class:`list` of names keyed by object
    name; the ``NO_OF_MEMBERS`` count is generated on write.

    :Call:
        >>> bco = BCObjects()
    """
    __slots__ = (
        "_header_lines",
        "_rawlines",
        "_dirty",
    )

    def __init__(self, *args, **kw):
        super().__init__()
        self._header_lines = []
        self._rawlines = {}
        self._dirty = set()

    def _read(self, lines: List[str]):
        r"""Parse the BC OBJECTS block

        :Call:
            >>> bco._read(lines)
        """
        if not lines:
            return
        self._header_lines = [lines[0]]
        i = 1
        n = len(lines)
        while i < n:
            line = lines[i]
            tokens = line.split()
            if len(tokens) >= 2 and RE_INT.fullmatch(tokens[-1]):
                name = tokens[0]
                count = int(tokens[-1])
                raw = [line]
                members = []
                j = i + 1
                while j < n and len(members) < count:
                    raw.append(lines[j])
                    members.extend(lines[j].split())
                    j += 1
                dict.__setitem__(self, name, members[:count])
                self._rawlines[name] = raw
                i = j
            elif not line.strip():
                i += 1
            else:
                # Not a count line; should not happen
                i += 1

    def set_members(self, name: str, members: List[str]):
        r"""Set the members of one BC object

        :Call:
            >>> bco.set_members(name, members)
        """
        dict.__setitem__(self, name, list(members))
        self._dirty.add(name)

    def to_lines(self) -> List[str]:
        r"""Convert block back to file lines

        :Call:
            >>> lines = bco.to_lines()
        """
        out = list(self._header_lines)
        for name, members in self.items():
            if name not in self._dirty and name in self._rawlines:
                out.extend(self._rawlines[name])
                continue
            out.append(f"{' '*13}{name}  {len(members)}")
            out.append(f"{' '*13}{'  '.join(members)}")
        return out


# --- Main file interface ---
class VulcanInpFile(OptionsDict):
    r"""Interface to VULCAN-CFD input files

    :Call:
        >>> inp = VulcanInpFile(fname=None)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of input file
    :Outputs:
        *inp*: :class:`VulcanInpFile`
            Interface to one VULCAN-CFD input file
        *inp.bcgroups*: :class:`BCGroups`
            Interface to ``BC GROUPS:`` block
        *inp.bcobjects*: :class:`BCObjects`
            Interface to ``BC OBJECTS:`` block
        *inp.blockconfig*: :class:`BlockConfig`
            Interface to ``BLOCK CONFIG.`` table
        *inp.regions*: :class:`RegionControls`
            Interface to region control specifications
    """
    __slots__ = (
        "fdir",
        "fname",
        "_entries",
        "_optlines",
        "_contlines",
        "_contdirty",
        "_tails",
        "_dirty",
        "_bcgroups",
        "_bcobjects",
        "_blockconfig",
        "_regions",
    )

    _name = "options for VULCAN-CFD ``.inp`` input files"

    _optlist = set(OPTSPEC)
    _opttypes = {
        kk: (float, int, type(None))
        for kk, kind in OPTSPEC.items()
        if kind in (KIND_NUM, KIND_NUM_FILE, KIND_NUM_LIST, KIND_NUM_NAMES)
    }
    _optmap = dict(OPTMAP)

   # --- __dunder__ ---
    def __init__(self, fname: Optional[str] = None):
        super().__init__()
        self.fdir = None
        self.fname = None
        self._entries = []
        self._optlines = {}
        self._contlines = {}
        self._contdirty = set()
        self._tails = {}
        self._dirty = set()
        self._bcgroups = BCGroups()
        self._bcobjects = BCObjects()
        self._blockconfig = BlockConfig()
        self._regions = RegionControls()
        if isinstance(fname, str):
            self.read_inpfile(fname)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        self._dirty.add(key)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._dirty.discard(key)

    def __str__(self):
        clsname = self.__class__.__name__
        fname = getattr(self, "fname", None)
        if fname is None:
            return f"<{clsname}>"
        return f"<{clsname}('{fname}')>"

    def __repr__(self):
        return self.__str__()

   # --- Blocks ---
    @property
    def bcgroups(self) -> BCGroups:
        """Interface to ``BC GROUPS:`` block"""
        return self._bcgroups

    @property
    def bcobjects(self) -> BCObjects:
        """Interface to ``BC OBJECTS:`` block"""
        return self._bcobjects

    @property
    def blockconfig(self) -> BlockConfig:
        """Interface to ``BLOCK CONFIG.`` table"""
        return self._blockconfig

    @property
    def regions(self) -> RegionControls:
        """Interface to region control specifications"""
        return self._regions

   # --- Readers ---
    def read_inpfile(self, fname: str):
        r"""Read one VULCAN-CFD input file

        :Call:
            >>> inp.read_inpfile(fname)
        """
        # Absolutize file name
        if not os.path.isabs(fname):
            fname = os.path.realpath(fname)
        self.fdir, self.fname = os.path.split(fname)
        # Read text
        with open(fname, 'r') as fp:
            lines = fp.read().splitlines()
        # Clear previous contents
        dict.clear(self)
        self._entries = []
        self._optlines = {}
        self._contlines = {}
        self._contdirty = set()
        self._tails = {}
        self._dirty = set()
        self._bcgroups = BCGroups()
        self._bcobjects = BCObjects()
        self._blockconfig = BlockConfig()
        self._regions = RegionControls()
        # Parse
        self._parse(lines)

    # Keywords sorted longest-first for prefix matching
    _keys_sorted = sorted(OPTSPEC, key=len, reverse=True)

    def _match_keyword(self, line: str):
        r"""Find the keyword that matches the start of *line*

        :Call:
            >>> key = inp._match_keyword(line)
        :Outputs:
            *key*: :class:`str` | ``None``
                Keyword name, or ``None`` if no match
        """
        for key in self._keys_sorted:
            if line.startswith(key):
                rest = line[len(key):]
                if rest == '' or rest[:1] in ' \t':
                    return key
        return None

    def _parse(self, lines: List[str]):
        r"""Parse all lines of the input file

        :Call:
            >>> inp._parse(lines)
        """
        entries = self._entries
        n = len(lines)
        i = 0
        while i < n:
            line = lines[i]
            s = line.strip()
            # Comments & blanks
            if s == '' or s.startswith('$'):
                entries.append(('line', line))
                i += 1
                continue
            # End of solver control data
            if s.startswith('!'):
                entries.append(('line', line))
                i = self._parse_post(lines, i+1)
                continue
            if not line[:1].isspace():
                key = self._match_keyword(line)
                if key is not None:
                    nq = self._read_option(key, lines, i)
                    entries.append(('opt', key))
                    i += nq
                    if key in ("BLOCK CONFIG", "BLOCK CONFIG."):
                        ncfg = int(round(self.get(key) or 0.0))
                        jend = min(i + ncfg + 1, n)
                        self._blockconfig._read(lines[i:jend])
                        entries.append(('blockconfig', 0))
                        i = jend
                    continue
                if s.startswith('SOLVER/STATUS'):
                    jend = i
                    blocks = []
                    cur = None
                    while jend < n:
                        sl = lines[jend].strip()
                        if sl.startswith('!'):
                            break
                        if not lines[jend][:1].isspace():
                            if sl.startswith('SOLVER/STATUS'):
                                cur = []
                                blocks.append(cur)
                            if sl and cur is not None:
                                cur.append(lines[jend])
                        elif cur is not None:
                            cur.append(lines[jend])
                        jend += 1
                    self._regions._read(blocks)
                    for kreg in sorted(self._regions):
                        entries.append(('region', kreg))
                    i = jend
                    continue
            # Unknown line
            entries.append(('line', line))
            i += 1

    def _read_option(self, key: str, lines: List[str], i: int) -> int:
        r"""Read one keyword option and its continuation lines

        :Call:
            >>> nq = inp._read_option(key, lines, i)
        :Outputs:
            *nq*: :class:`int`
                Number of lines consumed
        """
        kind = OPTSPEC[key]
        raw = lines[i]
        idx = raw.find(key) + len(key)
        rest = raw[idx:]
        mval = re.match(r"\s+(\S+)(.*)", rest)
        value = None
        if mval and not mval.group(1).startswith('('):
            value = to_number(mval.group(1))
            if not isinstance(value, float):
                # Trailing text such as memory units
                self._tails[key] = mval.group(1) + mval.group(2)
                value = None
            elif mval.group(2).strip():
                g2 = mval.group(2).strip()
                if not g2.startswith('('):
                    self._tails[key] = g2
        dict.__setitem__(self, key, value)
        self._optlines[key] = [raw]
        nq = 1
        # Continuation lines
        if kind in (KIND_NUM_FILE, KIND_FLAG_NAME):
            cont = []
            while i+nq < len(lines) and lines[i+nq][:1] in ' \t':
                if not lines[i+nq].strip():
                    break
                cont.append(lines[i+nq].strip())
                self._optlines[key].append(lines[i+nq])
                nq += 1
            self._contlines[key] = cont
        elif kind == KIND_NUM_NAMES:
            nnames = int(round(value)) if value else 0
            cont = []
            while i+nq < len(lines) and len(cont) < nnames:
                li = lines[i+nq]
                if li[:1] not in ' \t' or not li.strip():
                    break
                cont.append(li.strip())
                self._optlines[key].append(li)
                nq += 1
            self._contlines[key] = cont
        elif kind == KIND_NUM_LIST:
            ntok = int(round(value)) if value else 0
            cont = []
            nfound = 0
            while i+nq < len(lines) and nfound < ntok:
                li = lines[i+nq]
                if li[:1] not in ' \t' or not li.strip():
                    break
                cont.append(li.strip())
                self._optlines[key].append(li)
                nfound += len(li.split())
                nq += 1
            self._contlines[key] = cont
        return nq

    def _parse_post(self, lines: List[str], i: int) -> int:
        r"""Parse the block after the ``!`` end-of-data line

        :Call:
            >>> i = inp._parse_post(lines, i)
        :Outputs:
            *i*: :class:`int`
                Index of next unconsumed line
        """
        entries = self._entries
        n = len(lines)
        while i < n:
            line = lines[i]
            su = line.strip().upper()
            if (not line[:1].isspace()) and su.startswith('BC GROUPS'):
                j = i + 1
                while j < n and (not lines[j].strip() or
                                 lines[j][:1].isspace()):
                    j += 1
                self._bcgroups._read(lines[i:j])
                entries.append(('bcgroups', 0))
                i = j
                continue
            if (not line[:1].isspace()) and su.startswith('BC OBJECTS'):
                j = i + 1
                while j < n and (not lines[j].strip() or
                                 lines[j][:1].isspace()):
                    j += 1
                self._bcobjects._read(lines[i:j])
                entries.append(('bcobjects', 0))
                i = j
                continue
            entries.append(('line', line))
            i += 1
        return i

   # --- Continuation-line access ---
    def get_cont(self, key: str) -> List[str]:
        r"""Get continuation lines for an option

        For example, the grid file name after ``UNS GRID`` or the
        list of names after ``PLOT FUNCTION``.

        :Call:
            >>> cc = inp.get_cont(key)
        :Inputs:
            *inp*: :class:`VulcanInpFile`
                VULCAN input file interface
            *key*: :class:`str`
                Keyword name
        :Outputs:
            *cc*: :class:`list`\ [:class:`str`]
                Continuation lines, stripped
        """
        return list(self._contlines.get(key, []))

    def set_cont(self, key: str, values: List[str], per_line: bool = True):
        r"""Set continuation lines for an option

        :Call:
            >>> inp.set_cont(key, values, per_line=True)
        :Inputs:
            *inp*: :class:`VulcanInpFile`
                VULCAN input file interface
            *key*: :class:`str`
                Keyword name
            *values*: :class:`list`\ [:class:`str`]
                One entry per continuation line if *per_line*,
                otherwise all on a single line
            *per_line*: {``True``} | ``False``
                Write each entry on its own line
        """
        vals = [str(vv) for vv in values]
        if per_line:
            self._contlines[key] = vals
        else:
            self._contlines[key] = ['  '.join(vals)]
        self._contdirty.add(key)

   # --- Write ---
    def to_lines(self) -> List[str]:
        r"""Convert interface back to a list of file lines

        :Call:
            >>> lines = inp.to_lines()
        :Outputs:
            *lines*: :class:`list`\ [:class:`str`]
                Lines of the output file
        """
        out = []
        added = set()
        # Insertion point for new options: before end/region blocks
        ins = None
        for kind, ref in self._entries:
            if ins is None and kind in (
                'region', 'bcgroups', 'bcobjects'
            ):
                ins = len(out)
            if kind == 'line':
                out.append(ref)
            elif kind == 'opt':
                if ref not in self:
                    # Option was deleted
                    continue
                out.extend(self._opt_to_lines(ref))
                added.add(ref)
            elif kind == 'blockconfig':
                out.extend(self._blockconfig.to_lines())
            elif kind == 'region':
                out.extend(self._regions.to_lines(ref))
            elif kind == 'bcgroups':
                out.extend(self._bcgroups.to_lines())
            elif kind == 'bcobjects':
                out.extend(self._bcobjects.to_lines())
        # Options added since reading
        newlines = []
        for key, val in self.items():
            if key in added or key not in OPTSPEC:
                continue
            text = key
            sval = fmt_number(val)
            if sval:
                tail = self._tails.get(key)
                if tail:
                    sval = f"{sval} {tail}"
                text = f"{key}  {sval}"
            newlines.append(text)
            newlines.extend(self._contlines.get(key, []))
        if newlines:
            if ins is None:
                out.extend(newlines)
            else:
                out[ins:ins] = newlines
        return out

    def _opt_to_lines(self, key: str) -> List[str]:
        r"""Convert one option to its file lines

        :Call:
            >>> lines = inp._opt_to_lines(key)
        """
        rawlines = self._optlines.get(key, [])
        out = []
        if key not in self._dirty and rawlines:
            out.append(rawlines[0])
        else:
            valstr = fmt_number(self.get(key))
            tail = self._tails.get(key)
            if tail:
                valstr = f"{valstr} {tail}" if valstr else str(tail)
            if rawlines:
                out.append(_splice_value(rawlines[0], key, valstr))
            elif valstr:
                out.append(f"{key}  {valstr}")
            else:
                out.append(key)
        if key in self._contdirty:
            out.extend('  ' + cc for cc in self._contlines.get(key, []))
        else:
            out.extend(rawlines[1:])
        return out

    def write(self, fname: Optional[str] = None):
        r"""Write contents back to a VULCAN input file

        :Call:
            >>> inp.write(fname=None)
        :Inputs:
            *inp*: :class:`VulcanInpFile`
                VULCAN input file interface
            *fname*: {``None``} | :class:`str`
                File name to write; defaults to file that was read
        """
        if fname is None:
            if self.fname is None:
                raise ValueError("No file name given and no file was read")
            fname = os.path.join(self.fdir or '', self.fname)
        lines = self.to_lines()
        with open(fname, 'w') as fp:
            fp.write('\n'.join(lines) + '\n')

   # --- Flight conditions ---
    def get_mach(self):
        """Get freestream Mach number"""
        return self.get("MACH NO.")

    def set_mach(self, mach: float):
        """Set freestream Mach number"""
        self["MACH NO."] = float(mach)

    def get_alpha(self):
        """Get angle of attack [deg]"""
        return self.get("ALPHA")

    def set_alpha(self, alpha: float):
        """Set angle of attack [deg]"""
        self["ALPHA"] = float(alpha)

    def get_beta(self):
        """Get angle of yaw [deg]"""
        return self.get("BETA")

    def set_beta(self, beta: float):
        """Set angle of yaw [deg]"""
        self["BETA"] = float(beta)

    def get_pressure(self):
        """Get freestream static pressure [Pa]"""
        return self.get("STATIC PRESSURE")

    def set_pressure(self, p: float):
        """Set freestream static pressure [Pa]"""
        self["STATIC PRESSURE"] = float(p)

    def get_temperature(self):
        """Get freestream static temperature [K]"""
        return self.get("STATIC TEMPERATURE")

    def set_temperature(self, t: float):
        """Set freestream static temperature [K]"""
        self["STATIC TEMPERATURE"] = float(t)

    def get_processors(self):
        """Get number of processors"""
        return self.get("PROCESSORS")

    def set_processors(self, n: int):
        """Set number of processors"""
        self["PROCESSORS"] = float(n)

   # --- Named data lines ---
    def get_gridfile(self) -> Optional[str]:
        r"""Get grid file name from ``STR GRID`` or ``UNS GRID``

        :Call:
            >>> fgrid = inp.get_gridfile()
        """
        for key in ("UNS GRID", "STR GRID"):
            cc = self._contlines.get(key)
            if cc:
                return cc[0]
        return None

    def set_gridfile(self, fgrid: str):
        r"""Set grid file name for whichever grid type is present

        :Call:
            >>> inp.set_gridfile(fgrid)
        """
        key = "UNS GRID" if "UNS GRID" in self else "STR GRID"
        self.set_cont(key, [fgrid])


# Splice a new value into an original line, preserving columns
def _splice_value(raw: str, key: str, valstr: str) -> str:
    r"""Replace the value in an original keyword line

    :Call:
        >>> line = _splice_value(raw, key, valstr)
    :Inputs:
        *raw*: :class:`str`
            Original line as read from the file
        *key*: :class:`str`
            Keyword at the start of the line
        *valstr*: :class:`str`
            New value text, empty to remove the value
    :Outputs:
        *line*: :class:`str`
            Updated line
    """
    idx = raw.find(key) + len(key)
    head = raw[:idx]
    rest = raw[idx:]
    mval = re.match(r"(\s+)(\S+)(.*)", rest)
    if mval:
        sp, tok, trail = mval.groups()
        if valstr == '':
            # Remove value; keep spacing before descriptor
            if trail.strip():
                return f"{head}  {trail.strip()}"
            return head.rstrip()
        return f"{head}{sp}{valstr}{trail}"
    if valstr == '':
        return raw.rstrip()
    if not rest.strip():
        return f"{head}  {valstr}"
    # Only a descriptor is present
    return f"{head}  {valstr}  {rest.strip()}"
