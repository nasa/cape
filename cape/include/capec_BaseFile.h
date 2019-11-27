/*!
  \file capec_BaseFile.h
  \brief Key file-reading utilities for CAPE C extension
  
  This file contains functions that perform basic tasks of reading text files
  for the extension module for CAPE.
*/
#ifndef _CAPEC_BASEFILE_H
#define _CAPEC_BASEFILE_H




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

#endif