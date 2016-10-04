

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


typedef char *ctce_string;

typedef int ctce_bool;
#define true 1
#define false 0

typedef struct tce_string_array_ *tce_string_array;

struct tce_string_array_ {
    ctce_string *list;
    int length;
};

void *tce_malloc(int length);

ctce_string mkString(char *s);

ctce_string *mkIndexList(ctce_string *indices, int length);

//Convert int to string
ctce_string int_str(int);

ctce_string combine_indices(ctce_string *indices, int count);

ctce_string combine_indexLists(ctce_string *upper, int upper_count, ctce_string *lower, int lower_count);

// Should be a string returned by combine_indexLists i.e., of the form
// upper_indices : lower_indices ex:- V,V,O : O,V,V
// (OR) could be simply a comma seperated list.
tce_string_array stringToList(ctce_string);

ctce_string *replicate_indices(ctce_string *indices, int len);

ctce_bool exists_index(ctce_string *list, int len, ctce_string x);

int count_index(ctce_string *list, int len, ctce_string x);

ctce_bool check_index_count(tce_string_array list, int count);

ctce_bool compare_index_lists(tce_string_array list1, tce_string_array list2);

ctce_bool check_array_usage(tce_string_array list1, tce_string_array list2);

ctce_bool exact_compare_index_lists(tce_string_array list1, tce_string_array list2);


void print_index_list(tce_string_array list1);

#endif
