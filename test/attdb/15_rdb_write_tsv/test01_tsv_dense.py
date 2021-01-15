#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CAPE modules
import cape.attdb.rdb as rdb

# Read DataKit from MAT file
db = rdb.DataKit("CN-alpha-beta.mat", DefaultWriteFormat="%.3f")

# Write simple dense TSV file
db.write_tsv_dense("CN-dense.tsv")

