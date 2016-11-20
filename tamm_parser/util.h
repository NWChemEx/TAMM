

#ifndef UTIL_H_
#define UTIL_H_

#include <cstring>
#include <cassert>
#include <iostream>
#include <vector>

typedef char *tamm_string;

using tamm_string_array = std::vector<tamm_string>;
//typedef struct tce_string_array_ *tce_string_array;

//struct tce_string_array_ {
//    tamm_string *list;
//    int length;
//};

void *tce_malloc(int length);

tamm_string *mkIndexList(tamm_string *indices, int length);

tamm_string constcharToChar(const char* s);

tamm_string combine_indices(const tamm_string_array indices, int count);

tamm_string combine_indexLists(const tamm_string_array& upper, int upper_count, const tamm_string_array& lower, int lower_count);

// Should be a string returned by combine_indexLists i.e., of the form
// upper_indices : lower_indices ex:- V,V,O : O,V,V
// (OR) could be simply a comma seperated list.
tamm_string_array stringToList(tamm_string);

tamm_string_array replicate_indices(tamm_string *indices, int len);

bool exists_index(const tamm_string_array& list, int len, tamm_string x);

int count_index(tamm_string_array &list, int len, tamm_string x);

//bool check_index_count(tce_string_array list, int count);

bool compare_index_lists(const tamm_string_array& list1, const tamm_string_array& list2);

bool check_array_usage(tamm_string_array& list1, tamm_string_array& list2);

//bool exact_compare_index_lists(tce_string_array list1, tce_string_array list2);
//void print_index_list(tce_string_array list1);

#endif
