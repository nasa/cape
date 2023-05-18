#include <Python.h>
#include <stdio.h>
#include <math.h>

// Local includes
#include "capec_Error.h"
#include "capec_PyTypes.h"
#include "capec_BaseFile.h"
#include "capec_CSVFile.h"


// Count data lines in open file
size_t capec_CSVFileCountLines(FILE *fp)
{
    // Line counts
    size_t nline = 0;
    // File position
    long pos;
    // Strings
    char c;
    
    // Remember current location
    pos = ftell(fp);
    
    // Read lines
    while (!feof(fp)) {
        // Read spaces
        capcec_FileAdvanceWhiteSpace(fp);
        // Get next character
        c = getc(fp);
        // Check it
        if (c == '\n') {
            // Empty line
            continue;
        } else if (feof(fp)) {
            // Last line
            printf("\nLabel 0080: EOF\n");
            break;
        }
        // Read to end of line (comment or not)
        capec_FileAdvanceEOL(fp);
        // Check if it was a comment
        if (c != '#') {
            // Increase line count
            nline += 1;
        }
        printf("Label 0072: nline=%i, pos=%i\n", nline, ftell(fp));
    }
    
    // Return to original location
    fseek(fp, pos, SEEK_SET);
    
    // Output
    return nline;
}


// Read next entry as a string
int capeCSV_ReadSTR(FILE *fp, PyObject *coldata, size_t irow)
{
    int ierr;
    int nscan;
    char buff[80];
    PyObject *v;
    
    // Attempt to read next entry as a string
    nscan = fscanf(fp, "%80[^,\n\r\t]", buff);
    // Check for empty string (otherwise we'll get "  " in Python)
    if (nscan == 0) {
        buff[0] = '\0';
    }
    // Convert to Python string
    v = capePyString_FromString((const char *) buff);
    // Check for errors
    if (v == NULL) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read string in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    ierr = PyList_SetItem(coldata, (Py_ssize_t) irow, v);
    // Pass along error indicator
    return ierr;
    
}


// Read next entry as a double
int capeCSV_ReadFLOAT32(FILE *fp, float *coldata, size_t irow)
{
    int nscan;
    float v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%f", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read float32 in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}

// Read next entry as a double
int capeCSV_ReadFLOAT64(FILE *fp, double *coldata, size_t irow)
{
    int nscan;
    double v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%lf", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read float64 in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}

// Read next entry as a long double
int capeCSV_ReadFLOAT128(FILE *fp, long double *coldata, size_t irow)
{
    int nscan;
    long double v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%Lf", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read float128 in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}


// Read next entry as an int8
int capeCSV_ReadINT8(FILE *fp, signed char *coldata, size_t irow)
{
    int nscan;
    signed char v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%hhi", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read int8 in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}

// Read next entry as an int16
int capeCSV_ReadINT16(FILE *fp, short *coldata, size_t irow)
{
    int nscan;
    short v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%hi", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read int16 in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}

// Read next entry as an int32
int capeCSV_ReadINT32(FILE *fp, int *coldata, size_t irow)
{
    int nscan;
    int v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%i", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read int in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}

// Read next entry as an int64
int capeCSV_ReadINT64(FILE *fp, long *coldata, size_t irow)
{
    int nscan;
    long v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%li", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read long int in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}


// Read next entry as an int8
int capeCSV_ReadUINT8(FILE *fp, unsigned char *coldata, size_t irow)
{
    int nscan;
    unsigned char v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%hhu", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read int8 in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}

// Read next entry as a uint16
int capeCSV_ReadUINT16(FILE *fp, short unsigned *coldata, size_t irow)
{
    int nscan;
    short unsigned v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%hu", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read int16 in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}

// Read next entry as a uint32
int capeCSV_ReadUINT32(FILE *fp, unsigned *coldata, size_t irow)
{
    int nscan;
    unsigned v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%u", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read unsigned int in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}

// Read next entry as a uint64
int capeCSV_ReadUINT64(FILE *fp, long unsigned *coldata, size_t irow)
{
    int nscan;
    long v;
    
    // Attempt to read next entry as a double
    nscan = fscanf(fp, "%lu", &v);
    // Error checks
    if (nscan != 1) {
        PyErr_Format(PyExc_ValueError,
            "Failed to read long unsigned in data row %li", (long) irow);
        return capeERROR_VALUE;
    }
    
    // Save it
    coldata[irow] = v;
    // Nominal exit
    return 0;
}


// Read next data column
int capeCSV_ReadNext(FILE *fp, void *coldata, int dtype, size_t irow)
{
    // Error flag
    int ierr;
    
    // Check dtype for what to read next
    if (dtype == capeDTYPE_float64) {
        ierr = capeCSV_ReadFLOAT64(fp, (double *) coldata, irow);
    } else if (dtype == capeDTYPE_int32) {
        ierr = capeCSV_ReadINT32(fp, (int *) coldata, irow);
    } else if (dtype == capeDTYPE_str) {
        ierr = capeCSV_ReadSTR(fp, (PyObject *) coldata, irow);
    } else if (dtype == capeDTYPE_float32) {
        ierr = capeCSV_ReadFLOAT32(fp, (float *) coldata, irow);
    } else if (dtype == capeDTYPE_float128) {
        ierr = capeCSV_ReadFLOAT128(fp, (long double *) coldata, irow);
    } else if (dtype == capeDTYPE_int8) {
        ierr = capeCSV_ReadINT8(fp, (signed char *) coldata, irow);
    } else if (dtype == capeDTYPE_int16) {
        ierr = capeCSV_ReadINT16(fp, (short *) coldata, irow);
    } else if (dtype == capeDTYPE_int64) {
        ierr = capeCSV_ReadINT64(fp, (long *) coldata, irow);
    } else if (dtype == capeDTYPE_uint8) {
        ierr = capeCSV_ReadUINT8(fp, (unsigned char *) coldata, irow);
    } else if (dtype == capeDTYPE_uint16) {
        ierr = capeCSV_ReadUINT16(fp, (short unsigned *) coldata, irow);
    } else if (dtype == capeDTYPE_uint32) {
        ierr = capeCSV_ReadUINT32(fp, (unsigned *) coldata, irow);
    } else if (dtype == capeDTYPE_uint64) {
        ierr = capeCSV_ReadUINT64(fp, (long unsigned *) coldata, irow);
    } else {
        PyErr_Format(PyExc_NotImplementedError,
            "Reading DTYPE '%s' not implemented", capeDTYPE_NAMES[dtype]);
        return capeERROR_NOT_IMPLEMENTED;
    }
    
    // Pass long error indicator
    return ierr;
}
