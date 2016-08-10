#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"
#include "corf.h"
#include "equations.h"

namespace ctce {

  void schedule_levels(vector<vector<Tensor> *> &tensors,
                       vector<vector<Operation> *>&ops);

  extern "C" {

    void ccsd_t1_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t1, Integer *d_t2, Integer *d_v2, 
                      Integer *k_f1_offset, Integer *k_i0_offset,
                      Integer *k_t1_offset, Integer *k_t2_offset, Integer *k_v2_offset);

    void ccsd_t2_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t1, Integer *d_t2, Integer *d_v2, 
                      Integer *k_f1_offset, Integer *k_i0_offset,
                      Integer *k_t1_offset, Integer *k_t2_offset, Integer *k_v2_offset);

    void ccsd_e_cxx_(Integer *d_f1, Integer *d_i0, Integer *d_t1, Integer *d_t2, Integer *d_v2,
                     Integer *k_f1_offset, Integer *k_i0_offset,
                     Integer *k_t1_offset, Integer *k_t2_offset, Integer *k_v2_offset);
    void ccsd_e_(Integer *d_f1, Integer *d_i0, Integer *d_t1, Integer *d_t2, Integer *d_v2,
                     Integer *k_f1_offset, Integer *k_i0_offset,
                     Integer *k_t1_offset, Integer *k_t2_offset, Integer *k_v2_offset);

    void ccsd_et12_cxx_(Integer *d_e, 
                        Integer *d_f1, Integer *d_v2, 
                        Integer *d_r1, Integer *d_r2, 
                        Integer *d_t1, Integer *d_t2, 
                        Integer *k_e_offset,
                        Integer *k_f1_offset, Integer *k_v2_offset,
                        Integer *k_r1_offset, Integer *k_r2_offset, 
                        Integer *k_t1_offset, Integer *k_t2_offset) {
      // icsd_et12_cxx_(d_e, d_f1, d_v2, d_r1, d_r2, d_t1, d_t2,
      //                k_e_offset, k_f1_offset, k_v2_offset, 
      //                k_r1_offset, k_r2_offset, k_t1_offset, k_t2_offset);
      // return;
      
      Equations e_eqs, t1_eqs, t2_eqs;
      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

      ccsd_e_equations(e_eqs);
      ccsd_t1_equations(t1_eqs);
      ccsd_t2_equations(t2_eqs);

      std::vector<Tensor> e_tensors, t1_tensors, t2_tensors;
      std::vector<Operation> e_ops, t1_ops, t2_ops;

      tensors_and_ops(e_eqs,e_tensors, e_ops);
      tensors_and_ops(t1_eqs,t1_tensors, t1_ops);
      tensors_and_ops(t2_eqs,t2_tensors, t2_ops);

      {
        //setup e tensors
        Tensor *i0 = &e_tensors[0];
        Tensor *f = &e_tensors[1];
        Tensor *v = &e_tensors[2];
        Tensor *t1 = &e_tensors[3];
        Tensor *t2 = &e_tensors[4];
        
        v->set_dist(idist);
        t1->set_dist(dist_nwma);
        f->attach(*k_f1_offset, 0, *d_f1);
        i0->attach(*k_e_offset, 0, *d_e);
        t1->attach(*k_t1_offset, 0, *d_t1);
        t2->attach(*k_t2_offset, 0, *d_t2);
        v->attach(*k_v2_offset, 0, *d_v2);
      }

      {
        //setup t1 tensors
        Tensor *i0 = &t1_tensors[0];
        Tensor *f = &t1_tensors[1];
        Tensor *v = &t1_tensors[2];
        Tensor *t1 = &t1_tensors[3];
        Tensor *t2 = &t1_tensors[4];
        
        v->set_dist(idist);
        t1->set_dist(dist_nwma);
        f->attach(*k_f1_offset, 0, *d_f1);
        i0->attach(*k_r1_offset, 0, *d_r1);
        t1->attach(*k_t1_offset, 0, *d_t1);
        t2->attach(*k_t2_offset, 0, *d_t2);
        v->attach(*k_v2_offset, 0, *d_v2);
      }

      {
        //setup t2 tensors
        Tensor *i0 = &t2_tensors[0];
        Tensor *f = &t2_tensors[1];
        Tensor *v = &t2_tensors[2];
        Tensor *t1 = &t2_tensors[3];
        Tensor *t2 = &t2_tensors[4];
        
        v->set_dist(idist);
        t1->set_dist(dist_nwma);
        f->attach(*k_f1_offset, 0, *d_f1);
        i0->attach(*k_r2_offset, 0, *d_r2);
        t1->attach(*k_t1_offset, 0, *d_t1);
        t2->attach(*k_t2_offset, 0, *d_t2);
        v->attach(*k_v2_offset, 0, *d_v2);        
      }

#if 1
      vector<vector<Tensor> *> tensors(3);
      vector<vector<Operation> *> ops(3);
      tensors[0] = &e_tensors;
      tensors[1] = &t1_tensors;
      tensors[2] = &t2_tensors;
      ops[0] = &e_ops;
      ops[1] = &t1_ops;
      ops[2] = &t2_ops;
      schedule_levels(tensors, ops);

#else
      ccsd_e_cxx_(d_f1, d_e, d_t1, d_t2, d_v2,
                   k_f1_offset, k_e_offset,
                   k_t1_offset, k_t2_offset, k_v2_offset);
      ccsd_t1_cxx_(d_f1, d_r1, d_t1, d_t2, d_v2,
                   k_f1_offset, k_r1_offset,
                   k_t1_offset, k_t2_offset, k_v2_offset);
      ccsd_t2_cxx_(d_f1, d_r2, d_t1, d_t2, d_v2,
                   k_f1_offset, k_r2_offset,
                   k_t1_offset, k_t2_offset, k_v2_offset);
#endif

      {
        //un-setup e tensors
        Tensor *i0 = &e_tensors[0];
        Tensor *f = &e_tensors[1];
        Tensor *v = &e_tensors[2];
        Tensor *t1 = &e_tensors[3];
        Tensor *t2 = &e_tensors[4];
        
        f->detach();
        i0->detach();
        t1->detach();
        t2->detach();
        v->detach();
      }

      {
        //un-setup t1 tensors
        Tensor *i0 = &t1_tensors[0];
        Tensor *f = &t1_tensors[1];
        Tensor *v = &t1_tensors[2];
        Tensor *t1 = &t1_tensors[3];
        Tensor *t2 = &t1_tensors[4];
        
        f->detach();
        i0->detach();
        t1->detach();
        t2->detach();
        v->detach();
      }

      {
        //un-setup t2 tensors
        Tensor *i0 = &t2_tensors[0];
        Tensor *f = &t2_tensors[1];
        Tensor *v = &t2_tensors[2];
        Tensor *t1 = &t2_tensors[3];
        Tensor *t2 = &t2_tensors[4];

        f->detach();
        i0->detach();
        t1->detach();
        t2->detach();
        v->detach();
      }
    }
  }
}
