
.. This documentation written by TestDriver()
   on 2021-03-19 at 09:42 PDT

Test ``13_config_xml``
========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/cape/13_config_xml/``

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
        

Command 1: Read JSON Configuration: Python 2
---------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.350254 seconds
    * Cumulative time: 0.350254 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 2: Read JSON Configuration: Python 3
---------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.556431 seconds
    * Cumulative time: 0.906685 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 3: Read XML Configuration: Python 2
--------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.193984 seconds
    * Cumulative time: 1.10067 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 4: Read XML Configuration: Python 3
--------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.472582 seconds
    * Cumulative time: 1.57325 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 5: Compare XML Configurations: Python 2
------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test03_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.336923 seconds
    * Cumulative time: 1.91017 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:

        .. code-block:: none


              .. code-block:: none

            <?xml version="1.0" encoding="ISO-8859-1"?>
            
            <Configuration Name="bullet sample" Source="bullet.tri">
            
             <!-- Containers -->
              <Component Name="bullet_no_base" Type="container" Parent="bullet_total">
              </Component>
              <Component Name="fins" Type="container" Parent="bullet_no_base">
              </Component>
             
              <Component Name="bullet_total" Type="container">
              </Component>
             <!-- Containers -->
            
             <!-- body -->
              <Component Name="cap" Type="tri" Parent="bullet_no_base">
               <Data> Face Label=1 </Data>
              </Component>
             
              <Component Name="body" Type="tri" Parent="bullet_no_base">
               <Data> Face Label=2 </Data>
              </Component>
             
              <Component Name="base" Parent="bullet_total" Type="tri">
               <Data> Face Label=3 </Data>
              </Component>
             <!-- body -->
             
             <!-- fins -->
              <Component Name="fin1" Parent="fins" Type="tri">
               <Data> Face Label=11 </Data>
              </Component>
              
              <Component Name="fin2" Parent="fins" Type="tri">
               <Data> Face Label=12 </Data>
              </Component>
              
              <Component Name="fin3" Parent="fins" Type="tri">
               <Data> Face Label=13 </Data>
              </Component>
              
              <Component Name="fin4" Parent="fins" Type="tri">
               <Data> Face Label=14 </Data>
              </Component>
             <!-- fins -->
            
            </Configuration>


Command 6: Compare XML Configurations: Python 3
------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test03_config_xml.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.542945 seconds
    * Cumulative time: 2.45312 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:

        .. code-block:: none


              .. code-block:: none

            <?xml version="1.0" encoding="ISO-8859-1"?>
            
            <Configuration Name="bullet sample" Source="bullet.tri">
            
             <!-- Containers -->
              <Component Name="bullet_no_base" Type="container" Parent="bullet_total">
              </Component>
              <Component Name="fins" Type="container" Parent="bullet_no_base">
              </Component>
             
              <Component Name="bullet_total" Type="container">
              </Component>
             <!-- Containers -->
            
             <!-- body -->
              <Component Name="cap" Type="tri" Parent="bullet_no_base">
               <Data> Face Label=1 </Data>
              </Component>
             
              <Component Name="body" Type="tri" Parent="bullet_no_base">
               <Data> Face Label=2 </Data>
              </Component>
             
              <Component Name="base" Parent="bullet_total" Type="tri">
               <Data> Face Label=3 </Data>
              </Component>
             <!-- body -->
             
             <!-- fins -->
              <Component Name="fin1" Parent="fins" Type="tri">
               <Data> Face Label=11 </Data>
              </Component>
              
              <Component Name="fin2" Parent="fins" Type="tri">
               <Data> Face Label=12 </Data>
              </Component>
              
              <Component Name="fin3" Parent="fins" Type="tri">
               <Data> Face Label=13 </Data>
              </Component>
              
              <Component Name="fin4" Parent="fins" Type="tri">
               <Data> Face Label=14 </Data>
              </Component>
             <!-- fins -->
            
            </Configuration>


