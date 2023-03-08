
Description
--------------

CAPE is software to improve Computational Aerosciences Productivity &
Execution.

CAPE has two main purposes:

1. To execute and post-process three different Computational Fluid Dynamics
(CFD) solvers:

    A. Cart3D
    B. FUN3D
    C. OVERFLOW

2. Create and use "datakits" a combination of database and toolkit. These are
   data structures particularly well-suited to databases created from
   aerosciences data.

This package has been used by NASA teams to create large databases (in some
instances including over 10,000 CFD solutions) for flight programs such as
Space Launch System.

The main benefit of using CAPE for CFD run matrices is having a single tool to
do many of the steps:

    * create separate folders for each CFD case,
    * copy files and make any modifications such as setting flight conditions
      in input files,
    * submit jobs to high-performance computing scheduling system such as PBS,
    * monitor the status of those jobs,
    * create PDF reports of results from one or more solutions,
    * extract data and conduct other post-processing,
    * archive (and unarchive, if necessary) solution files.

This software is released under the NASA Open Source Agreement Version 1.3.
While the software is freely available to everyone, user registration is
*requested* by emailing the author(s).

A collection of tutorials and examples can be found at

    https://github.com/nasa-ddalle/

Full documentation can be found at

    https://nasa.github.io/cape-doc


Installation
--------------

The simplest method to install CAPE is to use ``pip``:

:Linux/MacOS:

    .. code-block:: console

        $ python3 -m pip install git+https://github.com/nasa/cape.git#egg=cape

:Windows:

    .. code-block:: console

        $ py -m pip install git+https://github.com/nasa/cape.git#egg=cape

**Notes**:

    1.  On Windows (where CAPE might be of limited use), replace ``python3``
        with ``py``
    2.  It's also possible to install from one of the provided ``.whl`` files.
    3.  In many cases, adding ``--user`` to the above command is appropriate;
        on systems where you are not a root/admin user, such is likely
        required.
    4.  If installing a new version of CAPE where one might already be present,
        add the option ``--upgrade``.


**Notices**

Copyright © 2022 United States Government as represented by the Administrator
of the National Aeronautics and Space Administration.  All Rights Reserved.

**Disclaimers**

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
