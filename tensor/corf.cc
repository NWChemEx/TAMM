#include "corf.h"

extern "C" {
void fname_and_create_(Integer *da, Integer *size);
void fdestroy_(Integer *da, Integer *l_a_offset);
}

namespace ctce {

  void CorFortran(int use_c, Assignment &as, add_fn fn) {
    if(use_c) {
      as.execute();
    }
    else {
      Integer da = as.tA().ga(), da_offset = as.tA().offset_index();
      Integer dc = as.tC().ga(), dc_offset = as.tC().offset_index();
      fn(&da, &da_offset, &dc, &dc_offset);
    }
  }

  void CorFortran(int use_c, Multiplication& m, mult_fn fn) {
    if(use_c) {
      m.execute();
    }
    else {
      Integer da = m.tA().ga(), da_offset = m.tA().offset_index();
      Integer db = m.tB().ga(), db_offset = m.tB().offset_index();
      Integer dc = m.tC().ga(), dc_offset = m.tC().offset_index();
      fn(&da, &da_offset, &db, &db_offset, &dc, &dc_offset);
    }
  }

  void CorFortran(int use_c, Tensor *tensor, offset_fn fn) {
    if(use_c) {
      tensor->create();
    }
    else {
      Integer k_a, l_a, size, d_a;
      fn(&l_a, &k_a, &size);
      fname_and_create_(&d_a, &size);
      tensor->attach(k_a, l_a, d_a);
    }
  }

  void destroy(Tensor *t) {
    if(t->allocated()) {
      t->destroy();
    }
    else if(t->attached()) {
      Integer d_a = t->ga(), l_a = t->offset_handle();
      fdestroy_(&d_a, &l_a);
      t->detach();
    }
    else {
      assert(0);
    }
  }

}; /*ctce*/


