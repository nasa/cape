SHELL = /bin/sh

# Top-level directory
TOPDIR = .
# Source directory
MODULEDIR = pyCart
all: build

.PHONY: build
build:
	(cd $(MODULEDIR); ./build.py)

.PHONY: clean
clean:
	(cd $(MODULEDIR); rm _pycart.so)
