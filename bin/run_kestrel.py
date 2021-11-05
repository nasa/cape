#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
from cape.pykes.case import run_kestrel
if __name__ == "__main__":
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(run_kestrel())

