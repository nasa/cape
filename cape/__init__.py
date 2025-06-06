# -*- coding: utf-8 -*-
r"""

The :mod:`cape` module contains the top-level interface setup of various
solvers. It loads the most important methods from the various submodules
so that they are easier to access. Most tasks using the Cape API can be
accessed by loading this module.

    .. code-block:: python

        import cape

For example the following will read in a global settings instance
assuming that the present working directory contains the correct files.
(If not, various defaults will be used, but it is unlikely that the
resulting setup will be what you intended.)

    .. code-block:: python

        from cape.cfdx.cntl import Cntl
        cntl = Cntl()

Most of the pyCart submodules essentially contain a single class
definition, and many of these classes are accessible directly from the
:mod:`cape` module. The list of classes loaded directly in :mod:`cape`.

    * :class:`cape.cfdx.cntl.Cntl`
    * :class:`cape.trifile.Tri`
    * :class:`cape.trifile.Triq`

A categorized list of modules available to the API are listed below.

    * Core modules
       - :mod:`cape.cntl`
       - :mod:`cape.cfdx.options`
       - :mod:`cape.cfdx.case`
       - :mod:`cape.runmatrix`
       - :mod:`cape.cfdx.dataBook`
       - :mod:`cape.cfdx.pointSensor`
       - :mod:`cape.cfdx.lineload`
       - :mod:`cape.cfdx.report`

    * Primary supporting modules
       - :mod:`cape.cfdx.cmdgen`
       - :mod:`cape.cfdx.cmdrun`
       - :mod:`cape.cfdx.queue`
       - :mod:`cape.argread`
       - :mod:`cape.capeio`
       - :mod:`cape.manage`

    * File interfaces
       - :mod:`cape.filecntl`
       - :mod:`cape.filecntl.namelist`
       - :mod:`cape.filecntl.namelist2`
       - :mod:`cape.filecntl.tecplot`
       - :mod:`cape.filecntl.tex`
       - :mod:`cape.tri`
       - :mod:`cape.cgns`
       - :mod:`cape.msh`
       - :mod:`cape.plot3d`
       - :mod:`cape.plt`
       - :mod:`cape.step`

    * Utilities
       - :mod:`cape.util`
       - :mod:`cape.atm`
       - :mod:`cape.config`
       - :mod:`cape.convert`
       - :mod:`cape.geom`
       - :mod:`cape.tar`
       - :mod:`cape.text`
       - :mod:`cape.cfdx.volcomp`


**Notices**

Copyright © 2022 United States Government as represented by the
Administrator of the National Aeronautics and Space Administration.  All
Rights Reserved.

**Disclaimers**

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY
WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY,
INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE
WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM
INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE,
OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE,
IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS
AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND
SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF
THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES,
EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM
PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT
SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES
GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR
RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
"""

# Standard library modules
import os


# Save version number
version = "1.2.0"
__version__ = version

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
CapeFolder = os.path.split(_fname)[0]
TemplateFolder = os.path.join(CapeFolder, "templates")

