"""
Command-line argument processor: :mod:`pyCart.argread`
======================================================

Parse command-line inputs based on one of two methods.  The first method counts
both "-" and "--" as prefixes for keyword names; this is common among many
advanced programs.  For example, the two following examples would be treated as
equivalent (assuming it is called by some script "myScript.py".

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
    
"""

# Import from CAPE
from cape.argread import *

