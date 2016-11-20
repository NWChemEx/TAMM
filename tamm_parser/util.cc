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

tamm_string int_str(int a) {
    const char* s = (std::to_string(a)).c_str();
    return constcharToChar(s);
}



tamm_string combine_indexLists(tamm_string *indices1, int count1, tamm_string *indices2, int count2) {
    tamm_string upper = combine_indices(indices1, count1);
    tamm_string lower = combine_indices(indices2, count2);
    std::string s;
    s.append(upper);
    s.append(":");
    s.append(lower);
    return constcharToChar(s.c_str());

}


tamm_string *replicate_indices(tamm_string *indices, int len) {
    tamm_string *newind = (tamm_string *)tce_malloc(len * sizeof(tamm_string));
    int i = 0;
    for (i = 0; i < len; i++) newind[i] = strdup(indices[i]);

    return newind;
}

bool exists_index(tamm_string *list, int len, tamm_string x) {
    int i = 0;
    for (i = 0; i < len; i++) if (strcmp(list[i], x) == 0) return true;
    return false;
}

bool compare_index_lists(tce_string_array list1, tce_string_array list2) {
    int len1 = list1->length;
    int len2 = list2->length;
    if (len1 != len2) return false;
    tamm_string *alist1 = list1->list;
    tamm_string *alist2 = list2->list;
    int i = 0;
    for (i = 0; i < len1; i++) {
        if (!exists_index(alist2, len2, alist1[i])) return false;
    }
    return true;
}

bool check_array_usage(tce_string_array list1, tce_string_array list2) {
    int len1 = list1->length;
    int len2 = list2->length;
    if (len1 != len2) return false;
    tamm_string *alist1 = list1->list;
    tamm_string *alist2 = list2->list;
    int i = 0;
    for (i = 0; i < len1; i++) {
        if (strcmp(alist1[i], "N") != 0) if (strcmp(alist1[i], alist2[i]) != 0) return false;
    }
    return true;
}


int count_index(tamm_string *list, int len, tamm_string x) {
    int count = 0, i = 0;
    for (i = 0; i < len; i++) {
        if (strcmp(list[i], x) == 0) count++;
    }
    return count;
}


tce_string_array stringToList(const tamm_string s) {
    tamm_string str = strdup(s);
    int len = strlen(str);
    tamm_string *list = (tamm_string *) tce_malloc(sizeof(tamm_string) * (len+1));
    int i = 0;

    char *c = strtok(str, " ,:");
    while (c != nullptr) {
        list[i] = c;
        i++;
        c = strtok(nullptr, " ,:");
    }

    free(str);
    str = strdup(s);
    free(list);
    len = i;
    i = 0;
    list = (tamm_string*) tce_malloc(sizeof(tamm_string) * (len+1));

    c = strtok(str, " ,:");
    while (c != nullptr) {
        list[i] = c;
        i++;
        c = strtok(nullptr, " ,:");
    }
    tce_string_array p = (tce_string_array) tce_malloc(sizeof(*p));
    p->list = list;
    p->length = len;

    return p;
}

//Convert string array of indices to comma seperated string
tamm_string combine_indices(tamm_string *indices, int count) {
    if (indices == nullptr) return "\0";
    std::string s;
    for (int i=0;i<count;i++) {
        s.append(indices[i]);
        s.append(",");
    }
    return  constcharToChar(s.c_str());

}

//bool exact_compare_index_lists(tce_string_array list1, tce_string_array list2) {
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
//void print_index_list(tce_string_array list1) {
//    int i = 0;
//    for (i = 0; i < list1->length; i++) std::cout << list1->list[i] << ",";
//
//}
