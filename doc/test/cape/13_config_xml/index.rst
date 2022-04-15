
.. This documentation written by TestDriver()
   on 2022-04-15 at 01:40 PDT

Test ``13_config_xml``: **FAIL** (command 5)
==============================================

This test **FAILED** (command 5) on 2022-04-15 at 01:40 PDT

This test is run in the folder:

    ``test/cape/13_config_xml/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_config_xml.py
        $ python3 test01_config_xml.py
        $ python2 test02_config_xml.py
        $ python3 test02_config_xml.py
        $ python2 test03_config_xml.py
        $ python3 test03_config_xml.py

**Included file:** ``test01_config_xml.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.config
        
        # Read JSON config
        cfgj = cape.config.ConfigJSON("arrow.json")

**Included file:** ``test02_config_xml.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.config
        
        # Read JSON config
        cfgx = cape.config.ConfigXML("arrow.xml")

**Included file:** ``test03_config_xml.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape.config
        
        # Read JSON config
        cfgj = cape.config.ConfigJSON("arrow.json")
        
        # Write arrow2 XML config from JSON
        cfgj.WriteXML("arrow2.xml", Name="bullet sample", Source="bullet.tri")
        
        # Open arrow2 XML
        with open("arrow2.xml", "r") as f:
            # Print lines of f
            for line in f:
                # Skip comments
                if line.startswith(" <!--"):
                    continue
                else:
                    print(line)

Command 1: Read JSON Configuration: Python 2 (PASS)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.54 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 2: Read JSON Configuration: Python 3 (PASS)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.50 seconds
    * Cumulative time: 1.04 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 3: Read XML Configuration: Python 2 (PASS)
---------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.36 seconds
    * Cumulative time: 1.40 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 4: Read XML Configuration: Python 3 (PASS)
---------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.48 seconds
    * Cumulative time: 1.88 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 5: Compare XML Configurations: Python 2 (**FAIL**)
-----------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test03_config_xml.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.60 seconds
    * Cumulative time: 2.48 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
    * Target:

      .. code-block:: none

          No parent for component 'bullet_total'
        <?xml version="1.0" encoding="utf-8"?>
        
        
        
        <Configuration Name="bullet sample" Source="bullet.tri">
        
        
        
          <Component Name="cap" Parent="bullet_no_base" Type="tri">
        
            <Data>Face Label=1</Data>
        
          </Component>
        
        
        
          <Component Name="body" Parent="bullet_no_base" Type="tri">
        
            <Data>Face Label=2</Data>
        
          </Component>
        
        
        
          <Component Name="base" Parent="bullet_total" Type="tri">
        
            <Data>Face Label=3</Data>
        
          </Component>
        
        
        
          <Component Name="fin1" Parent="fins" Type="tri">
        
            <Data>Face Label=11</Data>
        
          </Component>
        
        
        
          <Component Name="fin2" Parent="fins" Type="tri">
        
            <Data>Face Label=12</Data>
        
          </Component>
        
        
        
          <Component Name="fin3" Parent="fins" Type="tri">
        
            <Data>Face Label=13</Data>
        
          </Component>
        
        
        
          <Component Name="fin4" Parent="fins" Type="tri">
        
            <Data>Face Label=14</Data>
        
          </Component>
        
        
        
          <Component Name="bullet_no_base" Parent="bullet_total" Type="container">
        
          </Component>
        
        
        
          <Component Name="bullet_total" Type="container">
        
          </Component>
        
        
        
          <Component Name="fins" Parent="bullet_no_base" Type="container">
        
          </Component>
        
        
        
        </Configuration>
        
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test03_config_xml.py", line 11, in <module>
            cfgj.WriteXML("arrow2.xml", Name="bullet sample", Source="bullet.tri")
        TypeError: WriteXML() got an unexpected keyword argument 'Name'
        


