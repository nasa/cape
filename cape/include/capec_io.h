#ifndef _CAPEC_IO_H
#define _CAPEC_IO_H

#include <byteswap.h>

// Byteswap macros
#define bs32(x) (*(unsigned *)&(x) = __bswap_32(*(unsigned *)&(x)))
#define bs64(x) (*(unsigned long *)&(x) = __bswap_64(*(unsigned long *)&(x)))

// Check if little-endian
int is_le(void);

// Get total size of NumPy array
int np_size(PyArrayObject *P);

// Byteswap functions for floats/doubles
float  swap_single(const float f);
double swap_double(const double f);      

// Individual integers
int capec_Write_b4_i(FILE *fid, int    v);
int capec_Write_b4_f(FILE *fid, float  v);
int capec_Write_b4_d(FILE *fid, double v);
int capec_Write_lb4_i(FILE *fid, int    v);
int capec_Write_lb4_f(FILE *fid, float  v);
int capec_Write_lb4_d(FILE *fid, double v);

// Big-endian single-precision writers
int capec_WriteRecord_b4_f1(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_b4_f2(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_b4_f3(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_b4_i1(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_b4_i2(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_b4_i3(FILE *fid, PyArrayObject *P);
// Big-endian double-precision writers
int capec_WriteRecord_b8_f1(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_b8_f2(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_b8_f3(FILE *fid, PyArrayObject *P);

// Little-endian single-precision writers
int capec_WriteRecord_lb4_f1(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_lb4_f2(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_lb4_f3(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_lb4_i1(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_lb4_i2(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_lb4_i3(FILE *fid, PyArrayObject *P);
// Little-endian double-precision writers
int capec_WriteRecord_lb8_f1(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_lb8_f2(FILE *fid, PyArrayObject *P);
int capec_WriteRecord_lb8_f3(FILE *fid, PyArrayObject *P);



# endif
                                      