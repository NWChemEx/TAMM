extern "C" {
#include <string.h>
}
#include "parseinput.hpp"

int parseinput(char *filename, sislice_param *params)
{
   int info = 0;
   int hmatrix_ready = 0, smatrix_ready = 0, shift_ready = 0;
   int n, hnnz, snnz;
   FILE *fp=NULL, *fhmat=NULL, *fsmat=NULL;
   char key[100], value[100];

   fp = fopen(filename,"r");
   if (!fp) {
      fprintf(stderr,"failed to open %s\n", filename);
      return -5;
   }

   params->hmatfname[0]='\0';
   params->smatfname[0]='\0';
   strcpy(params->logprefix,"logfile");
   //params->shift = 0.0;
   params->nev     = 100;
   params->nshifts = 10;
   params->maxiter = 3;
   params->maxscf = 3;
   params->maxlan = 20;
   params->tol  = 1.0e-6;
   while (fscanf(fp,"%s %s", key, value) != EOF) {
      if (strncmp(key,"hmatrix",4) == 0) {
         strcpy(params->hmatfname,value);
         /* go ahead and read the matrix */
 
         fhmat = fopen(params->hmatfname,"r");
         if (!fhmat) {
            fprintf(stderr,"parseinput: cannot open %s\n", params->hmatfname);
            info = -1;
            break;
         } 
         fread(&n, sizeof(int), 1, fhmat);
         fread(&hnnz, sizeof(int), 1, fhmat);
         printf("n = %d,  hnnz = %d\n", n, hnnz);
         params->n = n;
         params->hnnz = hnnz;

         params->hcolptr = (int*)malloc((n+1)*sizeof(int));
         params->hrowind = (int*)malloc(hnnz*sizeof(int));
         params->hnzvals = (double*)malloc(hnnz*sizeof(double));

         fread((int*)(params->hcolptr), sizeof(int), n+1, fhmat);
         fread((int*)(params->hrowind), sizeof(int), hnnz, fhmat);
         fread((double*)(params->hnzvals), sizeof(double), hnnz, fhmat);
         fclose(fhmat);

         hmatrix_ready = 1;
      }
      else if (strncmp(key,"smatrix",4) == 0) {
         strcpy(params->smatfname,value);
         /* go ahead and read the matrix */
 
         fsmat = fopen(params->smatfname,"r");
         if (!fsmat) {
            fprintf(stderr,"parseinput: cannot open %s\n", params->smatfname);
            info = -1;
            break;
         } 
         fread(&n, sizeof(int), 1, fsmat);
         fread(&snnz, sizeof(int), 1, fsmat);
         printf("n = %d,  snnz = %d\n", n, snnz);
         params->n = n;
         params->snnz = snnz;

         params->scolptr = (int*)malloc((n+1)*sizeof(int));
         params->srowind = (int*)malloc(snnz*sizeof(int));
         params->snzvals = (double*)malloc(snnz*sizeof(double));

         fread((int*)(params->scolptr), sizeof(int), n+1, fhmat);
         fread((int*)(params->srowind), sizeof(int), snnz, fhmat);
         fread((double*)(params->snzvals), sizeof(double), snnz, fhmat);
         fclose(fsmat);

         smatrix_ready = 1;
      }
      else if (strncmp(key,"logfile",3) == 0) {
         strcpy(params->logprefix,value);
      }
      /*else if (strncmp(key,"shift",5) == 0) {
         params->shift = atof(value);
         shift_ready = 1; 
      }*/
      else if (strncmp(key,"nev",3) == 0) {
         params->nev = atoi(value);
      }
      else if (strncmp(key,"nshifts",3) == 0) {
         params->nshifts = atoi(value);
      }
      else if (strncmp(key,"maxiter",5) == 0) {
         params->maxiter = atoi(value);
      }
      else if (strncmp(key,"maxscf",5) == 0) {
         params->maxscf = atoi(value);
      }
      else if (strncmp(key,"maxlan",5) == 0) {
         params->maxlan = atoi(value);
      }
      else if (strncmp(key,"tol",3) == 0) {
         params->tol = atof(value);
      }
   } /* endwhile */

   fclose(fp);

   if (!hmatrix_ready || !smatrix_ready) {
      fprintf(stderr,"parseinput: no matrix specified\n");
      info = -1;
      goto EXIT;
   }

   /*if (!shift_ready) { 
      fprintf(stderr,"parseinput: no shift or invalid shift\n");
      fprintf(stderr, " shift = %11.3e\n", params->shift);
      info = -2;
   }*/

EXIT:
   return info;
}

int parse_scf_input(char *filename, scf_param *params)
{
   int info = 0;
   char key[100], value[100];
   FILE *fp = NULL;

   fp = fopen(filename,"r");
   if (!fp) {
      fprintf(stderr,"failed to open %s\n", filename);
      return -5;
   }

   params->hmatfname[0]='\0';
   params->smatfname[0]='\0';
   //params->shift = 0.0;
   params->nev     = 100;
   params->nshifts = 10;
   params->maxiter = 3;
   params->maxscf = 3;
   params->maxlan = 20;
   params->tol  = 1.0e-6;
   while (fscanf(fp,"%s %s", key, value) != EOF) {
      if (strncmp(key,"hmatrix",4) == 0) {
         strcpy(params->hmatfname,value);
      }
      else if (strncmp(key,"smatrix",4) == 0) {
         strcpy(params->smatfname,value);
      }
      else if (strncmp(key,"logfile",4) == 0) {
         strcpy(params->logprefix,value);
      }
      else if (strncmp(key,"nev",3) == 0) {
         params->nev = atoi(value);
      }
      else if (strncmp(key,"nshifts",3) == 0) {
         params->nshifts = atoi(value);
      }
      else if (strncmp(key,"maxiter",5) == 0) {
         params->maxiter = atoi(value);
      }
      else if (strncmp(key,"maxscf",5) == 0) {
         params->maxscf = atoi(value);
      }
      else if (strncmp(key,"maxlan",5) == 0) {
         params->maxlan = atoi(value);
      }
      else if (strncmp(key,"tol",3) == 0) {
         params->tol = atof(value);
      }
   } /* endwhile */

   fclose(fp);

EXIT:
   return info;
}

