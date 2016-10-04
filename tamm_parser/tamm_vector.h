#ifndef __CTCE_VECTOR_H__
#define __CTCE_VECTOR_H__

typedef struct vector_ {
    void **data;
    int size;
    int count;
} ctce_vector;

void vector_init(ctce_vector *);

int vector_count(ctce_vector *);

void vector_add(ctce_vector *, void *);

void vector_set(ctce_vector *, int, void *);

void *vector_get(ctce_vector *, int);

void vector_delete(ctce_vector *, int);

void vector_free(ctce_vector *);

#endif
