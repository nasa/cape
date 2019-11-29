/*!
  \file capec_BaseFile.h
  \brief Key file-reading utilities for CAPE C extension
  
  This file contains functions that perform basic tasks of reading text files
  for the extension module for CAPE.
*/
#ifndef _CAPEC_BASEFILE_H
#define _CAPEC_BASEFILE_H


//! Acceptable text data types
enum capeDTYPE_ENUM {
    capeDTYPE_float16,            //!< short (16-bit) float
    capeDTYPE_float32,            //!< single-precision (32-bit) float
    capeDTYPE_float64,            //!< double-precision (64-bit) real
    capeDTYPE_float128,           //!< long (128-bit) real
    capeDTYPE_int8,               //!< extra short (8-bit) int
    capeDTYPE_int16,              //!< short (16-bit) int
    capeDTYPE_int32,              //!< single (32-bit) int
    capeDTYPE_int64,              //!< long (64-bit) int
    capeDTYPE_uint8,              //!< extra short (8-bit) int
    capeDTYPE_uint16,             //!< short (16-bit) int
    capeDTYPE_uint32,             //!< single (32-bit) int
    capeDTYPE_uint64,             //!< long (64-bit) int
    capeDTYPE_str,                //!< string
    capeDTYPE_Last                //!< number of data types
};

//! Names for data types
static char *capeDTYPE_NAMES[capeDTYPE_Last] = {
    "float16",
    "float32",
    "float64",
    "float128",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "str"
};


//! \brief Advance file to end of current line
void
capec_FileAdvanceEOL(
    FILE *fp //!< File handle
    );

//! \brief Advance file to end of current white space
void
capcec_FileAdvanceWhiteSpace(
    FILE *fp //!< File handle
    );

//! \brief Find integer code for data type by name
//!
//! \return Data type enumeration code
int
capeDTYPE_FromString(
    const char *dtype //!< Name of data type
    );

//! \brief Add data type integers to module
//!
//! \return Error flag (0 for ok)
int capec_AddDTypes(
    PyObject *m      //!< Handle to module to which DTYPEs are added
    );

#endif