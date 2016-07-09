#include "util.h"

void *tce_malloc(int length)
{
  void *p = malloc(length);
  if (!p) {
    fprintf(stderr,"\n Out of memory!\n");
    exit(1);
  }
  return p;
}

string mkString(char *s)
{
  string p = tce_malloc(strlen(s)+1);
  strcpy(p,s);
  return p;
}

string* mkIndexList(string *indices, int length){
  string *newlist = malloc(length * sizeof(string));
  int i=0;
  for(i=0;i<length;i++) {
    newlist[i] = mkString(indices[i]);
  }
  return newlist;

}

