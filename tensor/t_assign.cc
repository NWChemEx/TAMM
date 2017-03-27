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
#include "tensor/t_assign.h"
#include <vector>
#include "tensor/stats.h"

#define USE_TIMER 1

using std::vector;

namespace tamm {

  void t_assign1(const Tensor& tC, const vector<IndexName>& c_ids_name,
               const Tensor& tA, const vector<IndexName>& a_ids_name,
               IterGroup<triangular>* out_itr,
               double coef, gmem::Handle sync_ga, int spos) {
  vector<size_t> order(tC.dim());
  vector<Index> c_ids = name2ids(tC,c_ids_name);
  vector<Index> a_ids = name2ids(tA,a_ids_name);
  
  for (int i = 0; i < tC.dim(); ++i) {
    order[i] = find(c_ids_name.begin(), c_ids_name.end(), a_ids_name[i]) - c_ids_name.begin() + 1;
  }

  int nprocs = gmem::ranks();
  int count = 0;

  int taskDim = 1;
  char taskStr[10] = "NXTASKA";
  gmem::Handle taskHandle;
  int sub;
  if (sync_ga.valid()) {
    taskHandle = sync_ga;
    sub = spos;
  } else {
    taskHandle =
        gmem::create(gmem::Int, 1, taskStr);  // global array for next task
    gmem::zero(taskHandle);                   // initialize to zero
    gmem::sync();
    sub = 0;
  }

  int next = static_cast<int>(gmem::atomic_fetch_add(taskHandle, sub, 1));

  vector<size_t> out_vec;  // out_vec = c_ids_v
  out_itr->reset();
  while (out_itr->next(&out_vec)) {
    if (next == count) {
      if ((tC.is_spatial_nonzero(out_vec)) && (tC.is_spin_nonzero(out_vec)) &&
          (tC.is_spin_restricted_nonzero(out_vec))) {
        vector<int> vtab1(IndexNum);
        for (int i = 0; i < tC.dim(); ++i) {
          assert(c_ids_name[i] < IndexNum);
          vtab1[c_ids_name[i]] = out_vec[i];
        }
        vector<size_t> c_ids_v = out_vec;

        size_t dimc = compute_size(c_ids_v);
        if (dimc <= 0) continue;
        // double* buf_c_sort = new double[dimc];
        // memset(buf_c_sort, 0, dimc * sizeof(double));

        vector<size_t> a_ids_v(tA.dim());
        for (int i = 0; i < tA.dim(); ++i) {
          assert(a_ids_name[i] < IndexNum);
          a_ids_v[i] = vtab1[a_ids_name[i]];
        }
        vector<size_t> a_value_r;
        tA.gen_restricted(a_ids_v, &a_value_r);

#if 1
        double* buf_a = new double[dimc];
        double* buf_a_sort = new double[dimc];
        assert(tA.dim() == a_ids_v.size());
        setValue(&a_ids, a_ids_v);
        setValueR(&a_ids, a_value_r);

        vector<size_t> a_svalue, a_svalue_r;
        vector<IndexName> a_name;
        int a_sign =
            sortByValueThenExtSymGroup(a_ids, &a_name, &a_svalue,
                                       &a_svalue_r);
#ifdef USE_TIMER
         getTimer.start();
        //ttimer::stats_var.getTimer.start();
#endif
        tA.get(a_svalue_r, buf_a, dimc);
#ifdef USE_TIMER
         getTimer.stop();
        //ttimer::stats_var.getTimer.stop();
#endif
#if 1
        vector<size_t> a_sort_ids = sort_ids(a_name, c_ids_name);
        tamm_sort(buf_a, buf_a_sort, a_svalue /*tA._value()*/,
                  a_sort_ids /*tA.sort_ids(a_name)*/,
                  static_cast<double>(a_sign)*coef /*(double)tA.sign()*/);
#else
        tamm_sort(buf_a, buf_a_sort, a_svalue /*tA._value()*/,
                  order,
                  static_cast<double>(a_sign)*coef /*(double)tA.sign()*/);
#endif
        delete[] buf_a;
#endif
#ifdef USE_TIMER
        addTimer.start();
        //ttimer::stats_var.addTimer.start();
#endif
        tC.add(out_vec, buf_a_sort, dimc);
#ifdef USE_TIMER
        addTimer.stop();
        //ttimer::stats_var.addTimer.stop();
#endif

        delete[] buf_a_sort;
      }
      next = static_cast<int>(gmem::atomic_fetch_add(taskHandle, sub, 1));
    }
    ++count;
  }

  if (!sync_ga.valid()) {
    gmem::sync();
    gmem::destroy(taskHandle);
  }
}


void t_assign2(const Tensor& tC, const vector<IndexName>& c_ids,
               const Tensor& tA, const vector<IndexName>& a_ids,
               IterGroup<triangular>* out_itr,
               double coef, gmem::Handle sync_ga, int spos) {
  vector<size_t> order(tC.dim());

  for (int i = 0; i < tC.dim(); ++i) {
    order[i] = find(c_ids.begin(), c_ids.end(), a_ids[i]) - c_ids.begin() + 1;
  }

  int nprocs = gmem::ranks();
  int count = 0;

  int taskDim = 1;
  char taskStr[10] = "NXTASKA";
  gmem::Handle taskHandle;
  int sub;
  if (sync_ga.valid()) {
    taskHandle = sync_ga;
    sub = spos;
  } else {
    taskHandle =
        gmem::create(gmem::Int, 1, taskStr);  // global array for next task
    gmem::zero(taskHandle);                   // initialize to zero
    gmem::sync();
    sub = 0;
  }

  int next = static_cast<int>(gmem::atomic_fetch_add(taskHandle, sub, 1));

  vector<size_t> out_vec;  // out_vec = c_ids_v
  out_itr->reset();
  while (out_itr->next(&out_vec)) {
    if (next == count) {
      vector<int> vtab1(IndexNum);
      for (int i = 0; i < tC.dim(); ++i) {
        assert(c_ids[i] < IndexNum);
        vtab1[c_ids[i]] = out_vec[i];
      }
      vector<size_t> a_ids_v(tA.dim());
      for (int i = 0; i < tA.dim(); ++i) {
        assert(a_ids[i] < IndexNum);
        a_ids_v[i] = vtab1[a_ids[i]];
      }
      if (tA.is_spatial_nonzero(out_vec) && tA.is_spin_nonzero(a_ids_v) &&
          tA.is_spin_restricted_nonzero(out_vec)) {
        size_t dimc = compute_size(out_vec);
        if (dimc <= 0) continue;
        vector<size_t> value_r;
        tA.gen_restricted(a_ids_v, &value_r);

        double* buf_a = new double[dimc];
        double* buf_a_sort = new double[dimc];
        // getTimer.start();
        tA.get(value_r, buf_a, dimc);
        // getTimer.stop();

        //        if (coef == -1.0) { // dscal
        //          for (int i = 0; i < dimc; ++i) buf_a[i] = -buf_a[i];
        //          tamm_sort(buf_a, buf_a_sort, a_ids_v, order, coef);
        //        }
        //        else {
        tamm_sort(buf_a, buf_a_sort, a_ids_v, order, coef);
        //        }
        // addTimer.start();
        tC.add(out_vec, buf_a_sort, dimc);
        // addTimer.stop();

        delete[] buf_a;
        delete[] buf_a_sort;
      }
      next = static_cast<int>(gmem::atomic_fetch_add(taskHandle, sub, 1));
    }
    ++count;
  }

  if (!sync_ga.valid()) {
    gmem::sync();
    gmem::destroy(taskHandle);
  }
}

void t_assign3(Assignment *a, gmem::Handle sync_ga, int spos) {
#ifdef USE_TIMER
   assignTimer.start();
  //ttimer::stats_var.assignTimer.start();
#endif
  t_assign1(a->tC(), a->cids(), a->tA(), a->aids(), &a->out_itr(), a->coef(),
      sync_ga, spos);
#ifdef USE_TIMER
   assignTimer.stop();
  //ttimer::stats_var.assignTimer.stop();
#endif
}



}  // namespace tamm
