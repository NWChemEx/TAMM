//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------

#include "util.h"

void *tce_malloc(int length) {
    void *p = malloc(length);
    if (!p) {
        std::cerr << "\n Out of memory!\n";
        std::exit(EXIT_FAILURE);
    }
    return p;
}

tamm_string *mkIndexList(tamm_string *indices, int length) {
    tamm_string *newlist = (tamm_string *) malloc(length * sizeof(tamm_string));
    int i = 0;
    for (i = 0; i < length; i++) {
        newlist[i] = strdup(indices[i]);
    }
    return newlist;
}

tamm_string constcharToChar(const char* s){
    tamm_string val = new char[strlen(s)+1]();
    strncpy(val, s, strlen(s)+1);
    return val;
}


tamm_string combine_indexLists(const tamm_string_array& indices1, const tamm_string_array& indices2) {
    tamm_string upper = combine_indices(indices1);
    tamm_string lower = combine_indices(indices2);
    std::string s;
    s.append(upper);
    s.append(":");
    s.append(lower);
    return constcharToChar(s.c_str());

}


bool exists_index(const tamm_string_array &list, tamm_string x) {
    for (auto i: list) if (strcmp(i, x) == 0) return true;
    return false;
}

bool compare_index_lists(const tamm_string_array& alist1, const tamm_string_array& alist2) {
    int len1 = alist1.size();
    int len2 = alist2.size();
    if (len1 != len2) return false;
    for (auto i: alist1) {
        if (!exists_index(alist2, i)) return false;
    }
    return true;
}

bool check_array_usage(const tamm_string_array& list1, const tamm_string_array& list2) {
    int len1 = list1.size();
    int len2 = list2.size();
    if (len1 != len2) return false;

    for (int i = 0; i < len1; i++) {
        if (strcmp(list1[i], "N") != 0) if (strcmp(list1[i], list2[i]) != 0) return false;
    }
    return true;
}


int count_index(const tamm_string_array &list, tamm_string x) {
    int count = 0;
    for (auto i: list) {
        if (strcmp(i, x) == 0) count++;
    }
    return count;
}


tamm_string_array stringToList(const tamm_string s) {
    tamm_string str = strdup(s);
    tamm_string_array p;

    char* c = strtok(str, " ,:");
    while (c != nullptr) {
        p.push_back(c);
        c = strtok(nullptr, " ,:");
    }
    return p;
}

//Convert string array of indices to comma seperated string
tamm_string combine_indices(const tamm_string_array& indices) {
    if (indices.size() == 0) return "\0";
    std::string s;
    for (auto i: indices) {
        s.append(i);
        s.append(",");
    }
    return  constcharToChar(s.c_str());

}

//bool exact_compare_index_lists(tamm_string_array list1, tamm_string_array list2) {
//    int len1 = list1->length;
//    int len2 = list2->length;
//    if (len1 != len2) return false;
//    tamm_string *alist1 = list1->list;
//    tamm_string *alist2 = list2->list;
//    int i = 0;
//    for (i = 0; i < len1; i++) {
//        if (strcmp(alist1[i], alist2[i]) != 0) return false;
//    }
//    return true;
//}
//
//void print_index_list(tamm_string_array list1) {
//    int i = 0;
//    for (i = 0; i < list1->length; i++) std::cout << list1->list[i] << ",";
//
//}
