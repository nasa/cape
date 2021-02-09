/*!
  \file capec_Error.h
  \brief Error codes and tools for CAPE C extension(s)
  
  Usage of these error codes is minimal since most error handling is done using
  the Python libraries error message functions.
*/

#ifndef _CAPEC_ERROR_H
#define _CAPEC_ERROR_H

// Error types
enum vge_ERROR_TYPES {
    capeOK,                         //!< Error code 0: success
    capeERROR_ATTR,                 //!< Missing attribute
    capeERROR_ATTR_TYPE,            //!< Attribute present with wrong type
    capeERROR_TYPE,                 //!< Object has wrong type
    capeERROR_VALUE,                //!< Other value error
    capeERROR_MEM_NEG,              //!< Negative memory allocation
    capeERROR_MEM_ALLOC,            //!< Other failed memory allocation
    capeERROR_MEM_OVERFLOW,         //!< Overflow of maximum malloc() size
    capeERROR_NOT_IMPLEMENTED       //!< Option is logical but not implemented
};

#endif // _VGC_ERROR_H
