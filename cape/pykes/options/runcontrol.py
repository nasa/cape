r"""
:mod:`cape.pykes.options.runcontrol`: Run control options
==============================================================

Options interface for aspects of running a case of Kestrel. It is only
moderately modified from the template :mod:`cape.cfdx.options` template.

"""

# Local imports
from ...cfdx.options import runControl


# Class for Report settings
class RunControl(runControl.RunControl):
    r"""Dictionary-based interface for *RunControl* section
    
    :Call:
        >>> opts = RunControl(**kw)
    :Versions:
        * 2021-10-20 ``@ddalle``: Version 1.0
    """
   
   # ============== 
   # Local settings
   # ==============
   # <
    # Function to get prefix
    def get_ProjectName(self, i=None):
        r"""Get the project rootname, or file prefix
        
        :Call:
            >>> fproj = opts.get_ProjectName(i=None)
        :Inputs:
            *opts*: :class:`pyOver.options.Options`
                Options interface
        :Outputs:
            *fproj*: :class:`str`
                Name of Kestrel job prefix
            *i*: {``None``} | :class:`int`
                Phase number
        :Versions:
            * 2021-10-20 ``@ddalle``: Version 1.0
        """
        return self.get_key("ProjectName", i)
   # >

