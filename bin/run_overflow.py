#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import sys
from cape.pyover.casecntl import run_overflow
if __name__ == "__main__":
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(run_overflow())

