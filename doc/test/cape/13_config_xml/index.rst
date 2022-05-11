
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:40 PDT

Test ``13_config_xml``: **FAIL** (command 1)
==============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:40 PDT

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
        cfgj.WriteXML("arrow2.xml", name="bullet sample", source="bullet.tri")
        
        # Open arrow2 XML
        with open("arrow2.xml", "r") as f:
            # Print lines of f
            for line in f:
                # Skip comments
                if line.startswith(" <!--"):
                    continue
                else:
                    print(line)

Command 1: Read JSON Configuration: Python 2 (**FAIL**)
--------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_config_xml.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.13 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_config_xml.py", line 5, in <module>
            import cape.config
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


