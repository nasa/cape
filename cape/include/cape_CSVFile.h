#ifndef _CAPE_CSVFILE_H
#define _CAPE_CSVFILE_H

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

#endif  // _CAPE_CSVFILE_H