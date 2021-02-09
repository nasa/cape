// Standard library
#include <Python.h>
#include <stdio.h>

// Local includes
#include "capec_Error.h"
#include "capec_Memory.h"

// Allocate 1D array (type agnostic)
int
capec_New1D(void **A, size_t n, size_t size)
{
    // Actual number of requested bytes
    long long totalsize;
    // Typecast
    size_t Asize;
    
    // Initialize in case something already pointed to. 
    (*A) = NULL;
    
    // Get size of the entire array
    totalsize = ((long long) n) * ((long long) size);
    // Convert to size_t
    Asize = (size_t) totalsize;
    
    // Check for null allocation
    if (totalsize == 0)
        return 0;
    
    // Someone asked for negative memory
    if ((long long) Asize != totalsize)
        return capeERROR_MEM_OVERFLOW;
    
    // Initialize array (pointers to rows)
    (*A) = (void *) malloc(Asize);
    // Check for errors
    if (*A == NULL)
        return capeERROR_MEM_ALLOC;
    
    // Normal output
    return 0;
}

// Allocate 2D array (type agnostic)
int
capec_New2D(void ***A, size_t n1, size_t n2, size_t size)
{
    char *temp;
    size_t i;
    size_t Asize;
    long long totalsize;
    
    // Initialize in case something already pointed to. 
    (*A) = NULL;
    
    // Get size of the entire array
    totalsize = ((long long) n1*n2) * ((long long) size);
    // Convert to size_t
    Asize = (size_t) totalsize;
    
    // Check for null allocation
    if (totalsize == 0)
        return 0;
    
    // Someone asked for negative memory
    if ((long long) Asize != totalsize)
        return capeERROR_MEM_OVERFLOW;
    
    // Create temporary array
    temp = (char *) malloc(Asize);
    // Check for errors
    if (temp == NULL)
        return capeERROR_MEM_ALLOC;
    
    // Initialize array (pointers to rows)
    (*A) = (void **) malloc(n1*sizeof(char *));
    // Check for errors
    if (*A == NULL)
        return capeERROR_MEM_ALLOC;
    
    // Allocate each row
    for(i = 0; i<n1; i++)
    {
        (*A)[i] = temp + i*n2*size;
    }
    
    // Normal output
    return 0;
}

// Deallocate 1D array
void
capec_Del1D(void *A)
{
    // Check already freed
    if (A == NULL)
        return;
    // Free
    free((void *) A);
}

// Deallocate 2D array
void
vgc_Del2D(void **A)
{
    // Check already freed
    if (A == NULL)
        return;
    // Free top-level and rows
    free((void *)  A[0]);
    free((void **) A);
}

