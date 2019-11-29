#ifndef _CAPE_CSVFILE_H
#define _CAPE_CSVFILE_H

//! \brief Read number of data lines from current position to end of file
//!
//! \return Number of lines in file
PyObject *
cape_CSVFileCountLines(PyObject *self, PyObject *args);
char doc_CSVFileCountLines[] = 
"Read CSV file to count valid data lines\n"
"\n"
":Call:\n"
"    >>> n = CSVFileCountLines(f)\n"
":Inputs:\n"
"    *f*: :class:`file`\n"
"        Open file interface\n"
":Outputs:\n"
"    *n*: :class:`int`\n"
"        Number of data lines from current position\n"
":Versions:\n"
"    * 2019-11-27 ``@ddalle``: First version\n"
"\n";

//! \brief Read data portion of CSV file in C
PyObject *
cape_CSVFileReadData(PyObject *self, PyObject *args);
char doc_CSVFileReadData[] = 
"Read data from CSV file\n"
"\n"
":Call:\n"
"    >>> CSVFileReadData(db, f)\n"
":Inputs:\n"
"    *db*: :class:`cape.attdb.ftypes.csv.CSVFile`\n"
"        CSV data file interface\n"
"    *f*: :class:`file`\n"
"        Open file interface\n"
":Versions:\n"
"    * 2019-11-29 ``@ddalle``: First version\n"
"\n";

#endif  // _CAPE_CSVFILE_H