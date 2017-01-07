#!/usr/bin/env python
"""
Color Conversion Tools
======================

This module contains tools to convert color strings to RGB values.  The primary
tool is to get a variety of colors via name (instead of just ``'r'``, ``'g'``,
``'b'``, ``'c'``, ``'m'``, ``'y'``, ``'k'``, and ``'w'``).  However, there
additional tools to convert a hex code to RGB and vice-versa.

:Versions:
    * 2017-01-05 ``@ddalle``: First version
"""

# Create a dictionary of named colors copied from matplotlib.colors.cnames
cnames = {
    'aliceblue': '#F0F8FF',
    'antiquewhite': '#FAEBD7',
    'aqua': '#00FFFF',
    'aquamarine': '#7FFFD4',
    'azure': '#F0FFFF',
    'b': '#0000FF',
    'beige': '#F5F5DC',
    'bisque': '#FFE4C4',
    'black': '#000000',
    'blanchedalmond': '#FFEBCD',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'burlywood': '#DEB887',
    'c': '#00FFFF',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',
    'cornsilk': '#FFF8DC',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkblue': '#00008B',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkgrey': '#A9A9A9',
    'darkkhaki': '#BDB76B',
    'darkmagenta': '#8B008B',
    'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkpurple': '#400040',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F',
    'darkslategrey': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dimgrey': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'fuchsia': '#FF00FF',
    'g': '#008000',
    'gainsboro': '#DCDCDC',
    'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'grey': '#808080',
    'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'ivory': '#FFFFF0',
    'k': '#000000',
    'khaki': '#F0E68C',
    'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5',
    'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD',
    'lightblue': '#ADD8E6',
    'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen': '#90EE90',
    'lightgrey': '#D3D3D3',
    'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'lightskyblue': '#87CEFA',
    'lightslategray': '#778899',
    'lightslategrey': '#778899',
    'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'linen': '#FAF0E6',
    'm': '#FF00FF',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumaquamarine': '#66CDAA',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'mintcream': '#F5FFFA',
    'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD',
    'navy': '#000080',
    'oldlace': '#FDF5E6',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'palevioletred': '#AFEEEE',
    'papayawhip': '#FFEFD5',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'r': '#FF0000',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'slategrey': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'w': '#FFFFFF',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'y': '$FFFF00',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'
}

# Convert Hex to RGB
def Hex2RGB(col):
    """Convert hex-code color to RGB values
    
    :Call:
        >>> rgb = Hex2RGB(col)
        >>> r, g, b = Hex2RGB(col)
    :Inputs:
        *col*: :class:`str`
            Six-digit hex code including ``'#'`` character
    :Outputs:
        *rgb*: :class:`list` (:class:`int`, size=3)
            Vector of RGB values
        *r*: 0 <= :class:`int` <= 255
            Red value
        *g*: 0 <= :class:`int` <= 255
            Green value
        *b*: 0 <= :class:`int` <= 255
            Blue value
    :Versions:
        * 2017-01-05 ``@ddalle``: First version
    """
    # Check first character
    if col[0] != '#':
        raise ValueError(
            ("Color '%s' is not a valid hex code\n" % col) +
            "Valid hex codes start with '#'")
    # Conversions
    r = eval('0x' + col[1:3])
    g = eval('0x' + col[3:5])
    b = eval('0x' + col[5:7])
    # Output
    return [r, g, b]
    
# Convert RGB to Hex code
def RGB2Hex(col):
    """Convert RGB integer values to a hex code
    
    :Call:
        >>> hx = RGB2Hex(col)
        >>> hx = RGB2Hex((r,g,b))
    :Inputs:
        *col*: :class:`tuple` | :class:`list`
            Indexable RGB specification
        *r*: 0 <= :class:`int` <= 255
            Red value
        *g*: 0 <= :class:`int` <= 255
            Green value
        *b*: 0 <= :class:`int` <= 255
            Blue value
    :Outputs:
        *hx*: :class:`str`
            Six-digit hex code including ``'#'`` character
    :Versions:
        * 2017-01-05 ``@ddalle``: First version
    """
    # Create hex code
    hx = "#%02x%02x%02x" % (col[0], col[1], col[2])
    # Output
    return hx
    
# Convert generic type to RGB
def ToRGB(col):
    """Convert generic value to RGB
    
    :Call:
        >>> rgb = ToRGB(col)
        >>> rgb = ToRGB(hx)
        >>> rgb = ToRGB(cname)
        >>> rgb = ToRGB(rgb)
        >>> rgb = ToRGB(RGB)
    :Inputs:
        *col*: :class:`str` | :class:`list` (:class:`int` | :class:`float`)
            Generic color specification
        *hx*: :class:`str`
            Six-digit hex code starting with ``'#'`` character
        *rgb*: :class:`list` (:class:`int`, size=3)
            Vector of RGB values
        *cname*: :class:`str`
            Color by name, such as ``"orange"``
        *V*: 0 <= :class:`float` <= 1
            Grayscale value
        *RGB*: :class:`list` (:class:`float`, size=3)
            Vector of fractional RGB values, 0 to 1 (inclusive)
    :Outputs:
        *rgb*: :class:`list` (:class:`int`, size=3)
            Vector of RGB values
    :Versions:
        * 2017-01-05 ``@ddalle``: First version
    """
    # Get input type
    t = type(col).__name__
    # Check input type
    if t in ['unicode', 'str']:
        # Convert to lower case and remove spaces
        col = col.lower()
        col = col.replace(' ', '')
        # Check for hex code
        if col.startswith('#'):
            # Convert from hex
            return Hex2RGB(col)
        elif col not in cnames:
            # Could not find
            raise ValueError(
                "Could not find a definition for color named '%s'" % col)
        else:
            # Get hex code from the dictionary
            return Hex2RGB(cnames[col])
    elif t == ["float"]:
        # Grayscale
        return max(0, min(255, [int(255*col)] * 3))
    elif t not in ['list', 'ndarray', 'tuple']:
        # Not an array
        raise TypeError(
            "Could not interpret a color that's not a string or list")
    elif len(col) != 3:
        # Not long enough
        raise ValueError("Color does not have three components")
    # Already RGB if reached this point; check for int
    r = col[0]; tr = type(r).__name__
    g = col[0]; tg = type(g).__name__
    b = col[0]; tb = type(b).__name__
    # Check type
    if tr.startswith('float'): r = int(255*r)
    if tg.startswith('float'): g = int(255*g)
    if tb.startswith('float'): b = int(255*b)
    # Limit
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    # Output
    return [r, g, b]
    

