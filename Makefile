SHELL = /bin/sh

# Top-level directory
TOPDIR = .
# Source directory
MODULEDIR = pyCart
CARTDIR = pyCart
CAPEDIR = cape
FUNDIR = pyFun
all: build

.PHONY: build
build:
	(cd $(MODULEDIR); ./build.py)

.PHONY: pycart
pycart:
	(cd $(CARTDIR); ./build.py)

.PHONY: cape
cape:
	(cd $(CAPEDIR); ./build.py)

.PHONY: clean
clean:
	(cd $(MODULEDIR); rm _pycart.so)
