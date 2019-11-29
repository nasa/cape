/*!
  \file capec_CSVFile.h
  \brief Key file-reading utilities for CAPE C extension
  
  This file contains functions that perform basic tasks of reading text files
  for the extension module for CAPE.
*/
#ifndef _CAPEC_CSVFILE_H
#define _CAPEC_CSVFILE_H




//! \brief Count data lines remaining in CSV file
//!
//! \return Number of data lines
int
capec_CSVFileCountLines(
    FILE *fp //!< File handle
    );

#endif