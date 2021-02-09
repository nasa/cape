/*!
  \file capec_CSVFile.h
  \brief Key CAPE C extension functions for read/write Cart3D tri files
  
  This file contains functions that perform basic tasks of reading and writing
  unstructured surface triangulation files
*/
#ifndef _CAPEC_TRI_H
#define _CAPEC_TRI_H


//! \brief Write node coordinates to TRI file
//!
//! \return Status code
int
capec_WriteTriNodes(
    FILE *fid,              //!< File handle
    PyArrayObject *P        //!< Array of node coordinates (nNode x 3)
    );


//! \brief Write node coordinates and BCs to SURF file
//!
//! \return Status code
int
capec_WriteSurfNodes(
    FILE *fid,              //!< File handle
    PyArrayObject *P,       //!< Array of node coordinates (nNode x 3)
    PyArrayObject *blds,    //!< Surface spacing (nNode)
    PyArrayObject *bldel    //!< Number of prism layers (nNode)
    );


//! \brief Write tri node numbers to TRI file
//!
//! \return Status code
int
capec_WriteTriTris(
    FILE *fid,              //!< File handle
    PyArrayObject *T        //!< Array of tri node indices (nTri x 3)
    );


//! \brief Write tris, compIDs, and BCs to SURF file
//!
//! \return Status code
int
capec_WriteSurfTris(
    FILE *fid,              //!< File handle
    PyArrayObject *T,       //!< Array of tri node indices (nTri x 3)
    PyArrayObject *C,       //!< Comp IDs for each tri (nTri)
    PyArrayObject *BC       //!< BC for each tri (nTri)
    );


//! \brief Write quads, compIDs, and BCs to SURF file
//!
//! \return Status code
int
capec_WriteSurfQuads(
    FILE *fid,              //!< File handle
    PyArrayObject *Q,       //!< Array of quad node indices (nQuad x 4)
    PyArrayObject *C,       //!< Comp IDs for each quad (nQuad)
    PyArrayObject *BC       //!< BC for each quad (nQuad)
    );


//! \brief Write compIDs to TRI file
//!
//! \return Status code
int
capec_WriteTriCompID(
    FILE *fid,              //!< File handle
    PyArrayObject *C        //!< Comp IDs for each quad (nTri)
    );


// Function to write states
//! \brief Write state variables to TRIQ file
//!
//! \return Status code
int
capec_WriteTriState(
    FILE *fid,              //!< File handle
    PyArrayObject *Q        //!< Array of conditions (nTri x nq)
    );

#endif