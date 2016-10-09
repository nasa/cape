#ifndef _PC_IO_H
#define _PC_IO_H

// Check if little-endian
int is_le(void);

// Get total size of NumPy array
int np_size(PyArrayObject *P);

// Byteswap functions for floats/doubles
float  swap_single(const float f);
double swap_double(const double f);

// Big-endian single-precision writers
int write_record_b4_1f(FILE *fid, PyArrayObject *P);
int write_record_b4_2f(FILE *fid, PyArrayObject *P);
int write_record_b4_1i(FILE *fid, PyArrayObject *P);
int write_record_b4_2i(FILE *fid, PyArrayObject *P);
// Big-endian double-precision writers
int write_record_b8_1f(FILE *fid, PyArrayObject *P);
int write_record_b8_2f(FILE *fid, PyArrayObject *P);
int write_record_b8_1i(FILE *fid, PyArrayObject *P);
int write_record_b8_2i(FILE *fid, PyArrayObject *P);

// Little-endian single-precision writers
int write_record_lb4_1f(FILE *fid, PyArrayObject *P);
int write_record_lb4_2f(FILE *fid, PyArrayObject *P);
int write_record_lb4_1i(FILE *fid, PyArrayObject *P);
int write_record_lb4_2i(FILE *fid, PyArrayObject *P);
// Little-endian double-precision writers
int write_record_lb8_1f(FILE *fid, PyArrayObject *P);
int write_record_lb8_2f(FILE *fid, PyArrayObject *P);
int write_record_lb8_1i(FILE *fid, PyArrayObject *P);
int write_record_lb8_2i(FILE *fid, PyArrayObject *P);



# endif
