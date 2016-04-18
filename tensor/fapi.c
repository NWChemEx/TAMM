
#include "fapi.h"

void fget_hash_block_ma(void *big_array, double *buf, size_t size, void *hash_map, size_t key) 
{
  Integer isize = size;
  Integer ikey = key;
  get_hash_block_ma_(big_array, buf, &isize, hash_map, &ikey);
}

void fget_hash_block(void *darr, double *buf, size_t size, void *hash_map, size_t key) 
{
  Integer isize = size;
  Integer ikey = key;
  get_hash_block_(darr, buf, &isize, hash_map, &ikey);
}

void fadd_hash_block(void *darr, double *buf, size_t size, void *hash_map, size_t key) 
{
  Integer isize = size;
  Integer ikey = key;
  add_hash_block_(darr, buf, &isize, hash_map, &ikey);
}

void fget_hash_block_i(void *darr, double *buf, size_t size, void *hash_map, size_t key, 
                       size_t g3b, size_t g2b, size_t g1b, size_t g0b) 
{
  Integer isize = size;
  Integer ikey = key;
  Integer i3 = g3b, i2 = g2b, i1 = g1b, i0 = g0b;
  get_hash_block_i_(darr, buf, &isize, hash_map, &ikey, &i3, &i2, &i1, &i0);
}

