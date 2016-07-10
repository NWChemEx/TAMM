

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


typedef char* string;

typedef int bool;
#define true 1
#define false 0

void *tce_malloc(int length);
string mkString(char *s);
string* mkIndexList(string *indices, int length);

//Convert int to string
string int_str(int);

#endif
