

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


typedef char* string;

void *tce_malloc(int length);
string mkString(char *s);
string* mkIndexList(string *indices, int length);

#endif
