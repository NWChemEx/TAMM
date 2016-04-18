/* -*- c-file-offsets: ((inextern-lang . nil)); -*- */

#ifndef GTD_H_INCLUDED
#define GTD_H_INCLUDED

#ifdef __cplusplus 
extern "C" {
#endif

/* Implementation-specific details*/

#if GT_GA_PORT

#include <ga.h>

typedef ga_nbhdl_t GTD_handle;
typedef Integer    GT_fint;
typedef int        GTD_procgroup;

#define GT_HANDLE_NULL (0x0)

#elif GT_MPI_PORT

#include <mpi.h>

typedef MPI_Request GTD_handle;
typedef long long    GTD_tensor_mapel;
typedef MPI_Comm    GTD_procgroup;
  typedef MPI_Fint    GT_fint;

#define GT_HANDLE_NULL MPI_REQUEST_NULL

#define GT_PROCGROUP_ISNULL(pg)   \
  ((pg) == MPI_COMM_NULL)

#else
#  error "Choose a communication port to compile"
#endif

#ifdef __cplusplus 
}
#endif

#endif /* GTD_H_INCLUDED */


