#ifndef PARSEINPUT_H
#define PARSEINPUT_H

extern "C" {
#include <stdio.h>
#include <stdlib.h>

typedef struct param {
  char   hmatfname[200];
  char   smatfname[200];
  char   logprefix[200];
  int    n;
  int    hnnz;
  int    snnz;
  int    *hcolptr;
  int    *hrowind;
  double *hnzvals;
  int    *scolptr;
  int    *srowind;
  double *snzvals;
  int    nev;
  int    nshifts;
  int    maxiter;
  int    maxscf;
  int    maxlan;
  double tol;
} sislice_param;

typedef struct scfparam {
  char   hmatfname[100];
  char   smatfname[100];
  char   logprefix[200];
  int    nev;
  int    nshifts;
  int    maxiter;
  int    maxscf;
  int    maxlan;
  double tol;
} scf_param;

int parseinput(char *fname, sislice_param *params);
int parse_scf_input(char *fname, scf_param *params);
}
#endif
