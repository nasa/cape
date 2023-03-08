r"""
This is a module built off of the :mod:`cape.filecntl.namelist` module
customized for manipulating FUN3D's namelist files.  Such files are
split into sections which are called "name lists."  Each name list has
syntax similar to the following.

    .. code-block:: none

        &project
            project_rootname = "pyfun"
            case_title = "Test case"
        /

and this module is designed to recognize such sections.  The main
feature of this module is methods to set specific properties of a
namelist file, for example the Mach number or CFL number.

Namelists are the primary FUN3D input file, and one is written for each
phase of a FUN3D case.  The namelist files prepared using this module
are written to ``fun3d.00.nml``, ``fun3d.01.nml``, etc.  These must be
linked to a hard-coded file name ``fun3d.nml`` as appropriate for the
currently running phase.

This function provides a class :class:`cape.filecntl.namelist.Namelist`
that can both read and set values in the namelist.  The key functions
are

    * :func:`Namelist.SetVar`
    * :func:`Namelist.GetVar`

The conversion from namelist text to Python is handled by
:func:`Namelist.ConvertToText`, and the reverse is handled by
:func:`Namelist.ConvertToVal`.  Conversions cannot quite be performed
just by the Python functions :func:`print` and :func:`eval` because
delimiters are not used in the same fashion.  Some of the conversions
are tabulated below.

    +----------------------+------------------------+
    | Namelist             | Python                 |
    +======================+========================+
    | ``val = "text"``     | ``val = "text"``       |
    +----------------------+------------------------+
    | ``val = 'text'``     | ``val = 'text'``       |
    +----------------------+------------------------+
    | ``val = 3``          | ``val = 3``            |
    +----------------------+------------------------+
    | ``val = 3.1``        | ``val = 3.1``          |
    +----------------------+------------------------+
    | ``val = .false.``    | ``val = False``        |
    +----------------------+------------------------+
    | ``val = .true.``     | ``val = True``         |
    +----------------------+------------------------+
    | ``val = .f.``        | ``val = False``        |
    +----------------------+------------------------+
    | ``val = .t.``        | ``val = True``         |
    +----------------------+------------------------+
    | ``val = 10.0 20.0``  | ``val = [10.0, 20.0]`` |
    +----------------------+------------------------+
    | ``val = 1, 100``     | ``val = [1, 100]``     |
    +----------------------+------------------------+
    | ``val(1) = 1.2``     | ``val = [1.2, 1.5]``   |
    +----------------------+------------------------+
    | ``val(2) = 1.5``     |                        |
    +----------------------+------------------------+
    | ``val = _mach_``     | ``val = "_mach_"``     |
    +----------------------+------------------------+

In most cases, the :class:`Namelist` will try to interpret invalid
values for any namelist entry as a string with missing quotes.  The
reason for this is that users often create template namelist with
entries like ``_mach_`` that can be safely replaced with appropriate
values using ``sed`` commands or something similar.

There is also a function :func:`Namelist.ReturnDict` to access the
entire namelist as a :class:`dict`.  Similarly,
:func:`Namelist.ApplyDict` can be used to apply multiple settings using
a :class:`dict` as input.

See also:

    * :mod:`cape.filecntl.namelist`
    * :func:`pyFun.case.GetNamelist`
    * :func:`cape.pyfun.cntl.Cntl.ReadNamelist`
    * :func:`cape.pyfun.cntl.Cntl.PrepareNamelist`

"""

# Standard library
import sys

# Local imports
from ..filecntl import namelist


# Base this class off of the main file control class.
class Namelist(namelist.Namelist):
    r"""File control class for :file:`fun3d.nml`

    This class is derived from the :class:`pyCart.fileCntl.FileCntl`
    class, so all methods applicable to that class can also be used for
    instances of this class.

    :Call:
        >>> nml = pyFun.Namelist()
        >>> nml = pyfun.Namelist(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of namelist file to read, defaults to ``'fun3d.nml'``
    :Version:
        * 2015-10-15 ``@ddalle``: Started
    """

    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="fun3d.nml"):
        r"""Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.SplitToSections(reg=r"\&([\w_]+)")

    # Find component by name
    def find_comp_index(self, comp):
        r"""Find index of a `component_name` in `component_parameters`

        :Call:
            >>> icomp = nml.find_comp_index(comp)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *icomp*: :class:`int` > 0 | ``None``
                Index of component w/ matching name, if any
        :Versions:
            * 2023-02-03 ``@ddalle``: v1.0
        """
        # Loop through indices until we get an empty result
        icomp = 1
        while True:
            # Get name
            name = self.GetVar("component_parameters", "component_name", icomp)
            # Check for positive result
            if comp == name:
                return icomp
            # Exit if no name
            if name is None:
                return
            # Increase counter
            icomp += 1

    # Set restart on
    def SetRestart(self, q=True, nohist=False):
        r"""Set the FUN3D restart flag on or off

        :Call:
            >>> nml.SetRestart(q=True, nohist=False)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *q*: {``True``} | ``False`` | ``None``
                Restart option, ``None`` turns flag to ``"on"``
            *nohist*: ``True`` | {``False``}
                If true, use 'on_nohistorykept' for 'restart_read'
        :Versions:
            * 2015-11-03 ``@ddalle``: First version
        """
        # Check status
        if (q is None) or (q and (q != "off")):
            # Turn restart on.
            if nohist:
                # Changing time solver
                self.SetVar(
                    'code_run_control', 'restart_read',
                    'on_nohistorykept')
            else:
                # Consistent phases
                self.SetVar('code_run_control', 'restart_read', 'on')
        else:
            # Turn restart off.
            self.SetVar('code_run_control', 'restart_read', 'off')

    # Function set the Mach number.
    def SetMach(self, mach):
        r"""Set the freestream Mach number

        :Call:
            >>> nml.SetMach(mach)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *mach*: :class:`float`
                Mach number
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.SetVar('reference_physical_properties', 'mach_number', mach)

    # Function to get the current Mach number.
    def GetMach(self):
        r"""Find the current Mach number

        :Call:
            >>> mach = nml.GetMach()
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
        :Outputs:
            *M*: :class:`float` (or :class:`str`)
                Mach number specified in :file:`input.cntl`
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Get the value.
        return self.GetVar('reference_physical_properties', 'mach_number')

    # Function to set the angle of attack
    def SetAlpha(self, alpha):
        r"""Set the angle of attack

        :Call:
            >>> nml.SetAlpha(alpha)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *alpha*: :class:`float`
                Angle of attack
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.SetVar(
            'reference_physical_properties',
            'angle_of_attack', alpha)

    # Function to set the sideslip angle
    def SetBeta(self, beta):
        r"""Set the sideslip angle

        :Call:
            >>> nml.SetBeta(beta)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *beta*: :class:`float`
                Sideslip angle
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.SetVar(
            'reference_physical_properties',
            'angle_of_yaw', beta)

    # Set temperature unites
    def SetTemperatureUnits(self, units=None):
        r"""Set the temperature units

        :Call:
            >>> nml.SetTemperatureUnits(units)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *units*: :class:`str`
                Units, defaults to ``"Rankine"``
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Check for defaults.
        if units is None: units = "Rankine"
        # Replace the line or add it if necessary.
        self.SetVar(
            'reference_physical_properties',
            'temperature_units', units)

    # Set the density
    def SetDensity(self, rho):
        r"""Set the freestream density

        :Call:
            >>> nml.SetDensity(rho)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *rho*: :class:`float`
                Freestream density [kg/m^3]
        :Versions:
            * 2018-04-19 ``@ddalle``: First version
        """
        self.SetVar('reference_physical_properties', 'density', rho)

    # Set the velocity
    def SetVelocity(self, V):
        r"""Set the freestream velocity magnitude

        :Call:
            >>> nml.SetTemperature(T)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *V*: :class:`float`
                Magnitude of freestream velocity [m/s]
        :Versions:
            * 2018-04-19 ``@ddalle``: First version
        """
        self.SetVar('reference_physical_properties', 'velocity', V)

    # Set the temperature
    def SetTemperature(self, T):
        r"""Set the freestream temperature

        :Call:
            >>> nml.SetTemperature(T)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *T*: :class:`float`
                Freestream temperature
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        self.SetVar('reference_physical_properties', 'temperature', T)

    # Set the Reynolds number
    def SetReynoldsNumber(self, Re):
        r"""Set the Reynolds number per unit length

        :Call:
            >>> nml.SetReynoldsNumber(Re)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *Re*: :class:`float`
                Reynolds number per unit length
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        self.SetVar('reference_physical_properties', 'reynolds_number', Re)

    # Set the number of iterations
    def SetnIter(self, nIter):
        r"""Set the number of iterations

        :Call:
            >>> nml.SetnIter(nIter)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *nIter*: :class:`int`
                Number of iterations to run
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        self.SetVar('code_run_control', 'steps', nIter)

    # Get the project root name
    def GetRootname(self):
        r"""Get the project root name

        :Call:
            >>> name = nml.GetRootname()
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
        :Outputs:
            *name*: :class:`str`
                Name of project
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        return self.GetVar('project', 'project_rootname')

    # Set the project root name
    def SetRootname(self, name):
        r"""Set the project root name

        :Call:
            >>> nml.SetRootname(name)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *name*: :class:`str`
                Name of project
        :Versions:
            * 2015-12-31 ``@ddalle``: First version
        """
        self.SetVar('project', 'project_rootname', name)

    # Get the grid format
    def GetGridFormat(self):
        r"""Get the mesh file extention

        :Call:
            >>> fext = nml.GetGridFormat()
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
        :Outputs:
            *fext*: {``"b8.ugrid"``} | :class:`str`
                Mesh file extension
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        # Format
        fmt = self.GetVar('raw_grid', 'grid_format')
        typ = self.GetVar('raw_grid', 'data_format')
        # Defaults
        if fmt is None: fmt = 'aflr3'
        if typ is None: typ = 'stream'
        # Create the extension
        if fmt == 'aflr3':
            # Check the type
            if typ == 'ascii':
                # ASCII AFLR3 mesh
                return 'ugrid'
            elif typ == 'unformatted':
                # Unformatted Fortran file
                if sys.byteorder == "big":
                    # Big-endian
                    return 'r8.ugrid'
                else:
                    # Little-endian
                    return 'lr8.ugrid'
            else:
                # C-type AFLR3 mesh
                if sys.byteorder == "big":
                    # Big-endian
                    return 'b8.ugrid'
                else:
                    # Little-endian
                    return 'lb8.ugrid'
        elif fmt == 'fast':
            # FAST
            return 'fgrid'
        else:
            # Use the raw format
            return fmt

    # Get the adapt project root name
    def GetAdaptRootname(self):
        r"""Get the post-adaptation project root name

        :Call:
            >>> name = nml.GetAdaptRootname()
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
        :Outputs:
            *name*: :class:`str`
                Name of adapted project
        :Versions:
            * 2015-12-31 ``@ddalle``: First version
        """
        return self.GetVar('adapt_mechanics', 'adapt_project')

    # Set the adapt project root name
    def SetAdaptRootname(self, name):
        r"""Set the post-adaptation project root name

        :Call:
            >>> nml.SetAdaptRootname(name)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *name*: :class:`str`
                Name of adapted project
        :Versions:
            * 2015-12-31 ``@ddalle``: First version
        """
        self.SetVar('adapt_mechanics', 'adapt_project', name)

    # Get the number of flow initialization volumes
    def GetNFlowInitVolumes(self):
        r"""Get the current number of flow initialization volumes

        :Call:
            >>> n = nml.GetNFlowInitVolumes()
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
        :Outputs:
            *n*: :class:`int`
                Number of flow initialization volumes
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        # Get the nominal value
        n = self.GetVar('flow_initialization', 'number_of_volumes')
        # Check for None
        if n is None:
            # Default is zero
            return 0
        else:
            # Use the number
            return n

    # Set the number of flow initialization volumes
    def SetNFlowInitVolumes(self, n):
        r"""Set the number of flow initialization volumes

        :Call:
            >>> nml.SetNFlowInitVolumes(n)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *n*: :class:`int`
                Number of flow initialization volumes
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        # Set value
        self.SetVar('flow_initialization', 'number_of_volumes', n)

