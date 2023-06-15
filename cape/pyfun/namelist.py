r"""

This is a module built off of the :mod:`cape.nmlfile.namelist` module
customized for manipulating FUN3D's namelist files. Such files are
split into sections which are called "namelists." Each namelist has
syntax similar to the following.

    .. code-block:: none

        &project
            project_rootname = "pyfun"
            case_title = "Test case"
        /

and this module is designed to recognize such sections. The main
feature of this module is methods to set specific properties of a
namelist file, for example the Mach number or CFL number.

Namelists are the primary FUN3D input file, and one is written for each
phase of a FUN3D case.  The namelist files prepared using this module
are written to ``fun3d.00.nml``, ``fun3d.01.nml``, etc.  These must be
linked to a hard-coded file name ``fun3d.nml`` as appropriate for the
currently running phase.

See also:

    * :mod:`cape.nmlfile`
    * :func:`cape.pyfun.GetNamelist`
    * :func:`cape.pyfun.cntl.Cntl.ReadNamelist`
    * :func:`cape.pyfun.cntl.Cntl.PrepareNamelist`

"""

# Standard library
import sys

# Local imports
from ..nmlfile import NmlFile


# FUN3D namelist class
class Namelist(NmlFile):
    r"""File control class for ``fun3d.nml``

    :Call:
        >>> nml = Namelist()
        >>> nml = Namelist(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of namelist file to read, defaults to ``'fun3d.nml'``
    :Version:
        * 2015-10-15 ``@ddalle``: v0.1; started
        * 2015-12-31 ``@ddalle``: v1.0; using ``filecntl.namelist``
        * 2023-06-15 ``@ddalle``: v2.0; use ``nmlfile``
    """

    # Set restart on
    def SetRestart(self, q=True, nohist=False):
        r"""Set the FUN3D restart flag on or off

        :Call:
            >>> nml.SetRestart(q=True, nohist=False)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *q*: {``True``} | ``False`` | ``None``
                Restart option, ``None`` turns flag to ``"on"``
            *nohist*: ``True`` | {``False``}
                If true, use 'on_nohistorykept' for 'restart_read'
        :Versions:
            * 2015-11-03 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        # Common values
        sec = "code_run_control"
        opt = "restart_read"
        # Check status
        if (q is None) or (q and (q != "off")):
            # Turn restart on.
            if nohist:
                # Changing time solver
                self.set_opt(sec, opt, 'on_nohistorykept')
            else:
                # Consistent phases
                self.set_opt(sec, opt, 'on')
        else:
            # Turn restart off.
            self.set_opt(sec, opt, 'off')

    # Function set the Mach number.
    def SetMach(self, mach):
        r"""Set the freestream Mach number

        :Call:
            >>> nml.SetMach(mach)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *mach*: :class:`float`
                Mach number
        :Versions:
            * 2015-10-15 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        # Replace the line or add it if necessary.
        self.set_opt('reference_physical_properties', 'mach_number', mach)

    # Function to get the current Mach number.
    def GetMach(self):
        r"""Find the current Mach number

        :Call:
            >>> mach = nml.GetMach()
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
        :Outputs:
            *mach*: :class:`float`
                Mach number specified in :file:`input.cntl`
        :Versions:
            * 2014-06-10 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        # Get the value.
        return self.get_opt('reference_physical_properties', 'mach_number')

    # Function to set the angle of attack
    def SetAlpha(self, alpha):
        r"""Set the angle of attack

        :Call:
            >>> nml.SetAlpha(alpha)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *alpha*: :class:`float`
                Angle of attack
        :Versions:
            * 2015-10-15 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        # Replace the line or add it if necessary.
        self.set_opt(
            'reference_physical_properties',
            'angle_of_attack', alpha)

    # Function to set the sideslip angle
    def SetBeta(self, beta):
        r"""Set the sideslip angle

        :Call:
            >>> nml.SetBeta(beta)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *beta*: :class:`float`
                Sideslip angle
        :Versions:
            * 2014-06-04 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        # Replace the line or add it if necessary.
        self.set_opt(
            'reference_physical_properties',
            'angle_of_yaw', beta)

    # Set temperature unites
    def SetTemperatureUnits(self, units=None):
        r"""Set the temperature units

        :Call:
            >>> nml.SetTemperatureUnits(units)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *units*: :class:`str`
                Units, defaults to ``"Rankine"``
        :Versions:
            * 2015-10-15 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        # Check for defaults.
        if units is None:
            units = "Rankine"
        # Replace the line or add it if necessary.
        self.set_opt(
            'reference_physical_properties',
            'temperature_units', units)

    # Set the density
    def SetDensity(self, rho):
        r"""Set the freestream density

        :Call:
            >>> nml.SetDensity(rho)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *rho*: :class:`float`
                Freestream density [kg/m^3]
        :Versions:
            * 2018-04-19 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        self.set_opt('reference_physical_properties', 'density', rho)

    # Set the velocity
    def SetVelocity(self, V):
        r"""Set the freestream velocity magnitude

        :Call:
            >>> nml.SetTemperature(T)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *V*: :class:`float`
                Magnitude of freestream velocity [m/s]
        :Versions:
            * 2018-04-19 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        self.set_opt('reference_physical_properties', 'velocity', V)

    # Set the temperature
    def SetTemperature(self, T):
        r"""Set the freestream temperature

        :Call:
            >>> nml.SetTemperature(T)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *T*: :class:`float`
                Freestream temperature
        :Versions:
            * 2015-10-15 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        self.set_opt('reference_physical_properties', 'temperature', T)

    # Set the Reynolds number
    def SetReynoldsNumber(self, Re):
        r"""Set the Reynolds number per unit length

        :Call:
            >>> nml.SetReynoldsNumber(Re)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *Re*: :class:`float`
                Reynolds number per unit length
        :Versions:
            * 2015-10-15 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        self.set_opt('reference_physical_properties', 'reynolds_number', Re)

    # Set the number of iterations
    def SetnIter(self, nIter):
        r"""Set the number of iterations

        :Call:
            >>> nml.SetnIter(nIter)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *nIter*: :class:`int`
                Number of iterations to run
        :Versions:
            * 2015-10-20 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        self.set_opt('code_run_control', 'steps', nIter)

    # Get the project root name
    def GetRootname(self):
        r"""Get the project root name

        :Call:
            >>> name = nml.GetRootname()
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
        :Outputs:
            *name*: :class:`str`
                Name of project
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        return self.get_opt('project', 'project_rootname')

    # Set the project root name
    def SetRootname(self, name):
        r"""Set the project root name

        :Call:
            >>> nml.SetRootname(name)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *name*: :class:`str`
                Name of project
        :Versions:
            * 2015-12-31 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        self.set_opt('project', 'project_rootname', name)

    # Get the grid format
    def GetGridFormat(self):
        r"""Get the mesh file extention

        :Call:
            >>> fext = nml.GetGridFormat()
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
        :Outputs:
            *fext*: {``"b8.ugrid"``} | :class:`str`
                Mesh file extension
        :Versions:
            * 2016-04-05 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        # Format
        fmt = self.get_opt('raw_grid', 'grid_format')
        typ = self.get_opt('raw_grid', 'data_format')
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
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
        :Outputs:
            *name*: :class:`str`
                Name of adapted project
        :Versions:
            * 2015-12-31 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        return self.get_opt('adapt_mechanics', 'adapt_project')

    # Set the adapt project root name
    def SetAdaptRootname(self, name):
        r"""Set the post-adaptation project root name

        :Call:
            >>> nml.SetAdaptRootname(name)
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *name*: :class:`str`
                Name of adapted project
        :Versions:
            * 2015-12-31 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        self.set_opt('adapt_mechanics', 'adapt_project', name)

    # Get the number of flow initialization volumes
    def GetNFlowInitVolumes(self):
        r"""Get the current number of flow initialization volumes

        :Call:
            >>> n = nml.GetNFlowInitVolumes()
        :Inputs:
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
        :Outputs:
            *n*: :class:`int`
                Number of flow initialization volumes
        :Versions:
            * 2016-03-29 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; switch to ``nmlfile``
        """
        # Get the nominal value
        n = self.get_opt('flow_initialization', 'number_of_volumes')
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
            *nml*: :class:`Namelist`
                Interface to ``fun3d.nml`` file
            *n*: :class:`int`
                Number of flow initialization volumes
        :Versions:
            * 2016-03-29 ``@ddalle``: v1.0
        """
        # Set value
        self.set_opt('flow_initialization', 'number_of_volumes', n)


