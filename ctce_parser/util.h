

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

typedef struct tce_string_array_ *tce_string_array;

struct tce_string_array_{
	    string* list; int length;
};

void *tce_malloc(int length);
string mkString(char *s);
string* mkIndexList(string *indices, int length);

//Convert int to string
string int_str(int);

string combine_indices(string* indices, int count);
string combine_indexLists(string* upper, int upper_count, string* lower, int lower_count);

// Should be a string returned by combine_indexLists i.e., of the form
// upper_indices : lower_indices ex:- V,V,O : O,V,V
// (OR) could be simply a comma seperated list.
tce_string_array stringToList(string);
string* replicate_indices(string* indices, int len);
bool exists_index(string* list, int len, string x);
int count_index(string* list, int len, string x);
bool check_index_count(tce_string_array list, int count);
bool compare_index_lists(tce_string_array list1, tce_string_array list2);
bool check_array_usage(tce_string_array list1, tce_string_array list2);
bool exact_compare_index_lists(tce_string_array list1, tce_string_array list2);


void print_index_list(tce_string_array list1);

#endif
