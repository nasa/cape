/*!
  \file capec_CSVFile.h
  \brief Key file-reading utilities for CAPE C extension
  
  This file contains functions that perform basic tasks of reading text files
  for the extension module for CAPE.
*/
#ifndef _CAPEC_CSVFILE_H
#define _CAPEC_CSVFILE_H



//! Default type for CSV column
#define capeCSV_DEFAULT_TYPE "float64"


//! \brief Count data lines remaining in CSV file
//!
//! \return Number of data lines
size_t
capec_CSVFileCountLines(
    FILE *fp //!< File handle
    );

//! \brief Read next entry of file
//!
//! \return Error indicator
int capeCSV_ReadNext(
    FILE *fp,           //!< File handle
    void *coldata,      //!< Pointer to data (list or C array)
    int dtype,          //!< Column data type index
    size_t irow         //!< Row index to read data to
    );

#endif