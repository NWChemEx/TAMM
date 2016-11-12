

#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cstdlib>
#include <iostream>


typedef char *tamm_string;

typedef int tamm_bool;
#define true 1
#define false 0

typedef struct tce_string_array_ *tce_string_array;

struct tce_string_array_ {
    tamm_string *list;
    int length;
};

void *tce_malloc(int length);

tamm_string mkString(char *s);

tamm_string *mkIndexList(tamm_string *indices, int length);

//Convert int to string
tamm_string int_str(int);

tamm_string combine_indices(tamm_string *indices, int count);

tamm_string combine_indexLists(tamm_string *upper, int upper_count, tamm_string *lower, int lower_count);

// Should be a string returned by combine_indexLists i.e., of the form
// upper_indices : lower_indices ex:- V,V,O : O,V,V
// (OR) could be simply a comma seperated list.
tce_string_array stringToList(tamm_string);

tamm_string *replicate_indices(tamm_string *indices, int len);

tamm_bool exists_index(tamm_string *list, int len, tamm_string x);

int count_index(tamm_string *list, int len, tamm_string x);

tamm_bool check_index_count(tce_string_array list, int count);

tamm_bool compare_index_lists(tce_string_array list1, tce_string_array list2);

tamm_bool check_array_usage(tce_string_array list1, tce_string_array list2);

tamm_bool exact_compare_index_lists(tce_string_array list1, tce_string_array list2);


void print_index_list(tce_string_array list1);

#endif
