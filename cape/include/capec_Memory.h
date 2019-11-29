#ifndef _CAPEC_MEMORY_H
#define _CAPEC_MEMORY_H

//! Allocate 1D array
int
capec_New1D(
    void **A,             //!< Pointer to pointer to allocated array
    size_t n,             //!< Length to allocate
    size_t size           //!< Width (bytes) for each entry
    );
// Allocate 2D array
int
capec_New2D(
    void ***A,            //!< Pointer to pointer to allocated array
    size_t n1,            //!< Number of rows to allocate
    size_t n2,            //!< Number of columns to allocate
    size_t size           //!< Width (bytes) for each entry
    );

// Deallocate 1D array
void
capec_Del1D(void *A);

// Deallocate 2D array
void
capec_Del2D(void **A);


// Specific types
#define capec_New2DDouble(A, n1, n2) \
    capec_New2D((void ***) A, n1, n2, sizeof(double));

#endif
