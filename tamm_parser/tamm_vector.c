#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ctce_vector.h"

void vector_init(ctce_vector *v) {
    v->data = NULL;
    v->size = 0;
    v->count = 0;
}

int vector_count(ctce_vector *v) {
    return v->count;
}

void vector_add(ctce_vector *v, void *e) {
    if (v->size == 0) {
        v->size = 10;
        v->data = malloc(sizeof(void *) * v->size);
        memset(v->data, '\0', sizeof(void *) * v->size);
    }

    if (v->size == v->count) {
        v->size *= 2;
        v->data = realloc(v->data, sizeof(void *) * v->size);
    }

    v->data[v->count] = e;
    v->count++;
}

void vector_set(ctce_vector *v, int index, void *e) {
    if (index >= v->count) {
        return;
    }

    v->data[index] = e;
}

void *vector_get(ctce_vector *v, int index) {
    if (index >= v->count) {
        return NULL;
    }

    return v->data[index];
}

void vector_delete(ctce_vector *v, int index) {
    if (index >= v->count) {
        return;
    }

    int i = 0, j = 0;
    for (i = index, j = index; i < v->count; i++) {
        v->data[j] = v->data[i];
        j++;
    }

    v->count--;
}

void vector_free(ctce_vector *v) {
    free(v->data);
}

//int main(void) {
//    vector v;
//    vector_init(&v);
//
//    vector_add(&v, "emil");
//    vector_add(&v, "hannes");
//    vector_add(&v, "lydia");
//    vector_add(&v, "olle");
//    vector_add(&v, "erik");
//
//    int i;
//    printf("first round:\n");
//    for (i = 0; i < vector_count(&v); i++) {
//        printf("%s\n", (char *) vector_get(&v, i));
//    }
//
//    vector_delete(&v, 1);
//    vector_delete(&v, 3);
//
//    printf("second round:\n");
//    for (i = 0; i < vector_count(&v); i++) {
//        printf("%s\n", (char *) vector_get(&v, i));
//    }
//
//    vector_free(&v);
//
//    return 0;
//}