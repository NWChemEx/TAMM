#ifndef __TAMM_VECTOR_H__
#define __TAMM_VECTOR_H__

typedef struct vector_ {
    void **data;
    int size;
    int count;
} tamm_vector;

void vector_init(tamm_vector *);

int vector_count(tamm_vector *);

void vector_add(tamm_vector *, void *);

void vector_set(tamm_vector *, int, void *);

void *vector_get(tamm_vector *, int);

void vector_delete(tamm_vector *, int);

void vector_free(tamm_vector *);

#endif
