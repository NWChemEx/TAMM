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
#include "tensor/t_mult.h"

#include <vector>

#include "tensor/stats.h"
#include "tensor/t_assign.h"

using std::vector;

namespace tamm {

void t_mult(double* a_c, const Tensor& tC, const Tensor& tA, const Tensor& tB,
            const double coef, const vector<IndexName>& sum_ids,
            IterGroup<triangular>* sum_itr, IterGroup<CopyIter>* cp_itr,
            const vector<size_t>& tid, Multiplication* m) {
  // vector<Integer>& vtab = Table::value();
  const vector<IndexName>& c_ids = id2name(m->c_ids);  // tC.name();
  const vector<IndexName>& a_ids = id2name(m->a_ids);  // tA.name();
  const vector<IndexName>& b_ids = id2name(m->b_ids);  // tB.name();

  vector<size_t> vtab(IndexNum);
  for (int i = 0; i < tC.dim(); ++i) {
    vtab[c_ids[i]] = tid[i];  // values used in genAntiIter
  }

  IterGroup<antisymm> ext_itr;
  genAntiIter(vtab, &ext_itr, tC, tA, tB);
  vector<size_t> ext_vec, sum_vec;
  ext_itr.reset();

  while (ext_itr.next(&ext_vec)) {
    if (tC.is_spatial_nonzero(ext_vec) && tC.is_spin_nonzero(ext_vec) &&
        tC.is_spin_restricted_nonzero(ext_vec)) {
      for (int i = 0; i < tC.dim(); ++i) {
        vtab[c_ids[i]] = ext_vec[i];
        // tC.setValueByName(c_ids[i], vtab[c_ids[i]]);
      }
      // const vector<size_t>& c_ids_v = tC.value();
      vector<size_t> c_ids_v(tC.dim());
      for (int i = 0; i < tC.dim(); i++) {
        c_ids_v[i] = vtab[c_ids[i]];
      }
      size_t dimc = compute_size(c_ids_v);
      assert(dimc == compute_size(tid));
      if (dimc == 0) continue;
      double* buf_c_sort = new double[dimc];
      memset(buf_c_sort, 0, dimc * sizeof(double));

      sum_itr->reset();
      bool ONE_TIME = sum_itr->empty();
      while (sum_itr->next(&sum_vec) || ONE_TIME) {
        ONE_TIME = false;
        for (int i = 0; i < sum_vec.size(); i++) {
          vtab[sum_ids[i]] = sum_vec[i];
        }
        vector<size_t> a_ids_v(tA.dim()), b_ids_v(tB.dim());
        for (int i = 0; i < tA.dim(); i++) {
          assert(a_ids[i] < IndexNum);
          a_ids_v[i] = vtab[a_ids[i]];
        }
        for (int i = 0; i < tB.dim(); i++) {
          assert(b_ids[i] < IndexNum);
          b_ids_v[i] = vtab[b_ids[i]];
        }
        // for (int i=0; i<sum_ids.size(); i++) {
        //   tA.setValueByName(sum_ids[i],sum_vec[i]);
        //   tB.setValueByName(sum_ids[i],sum_vec[i]);
        // }
        // vector<size_t> a_ids_v = tA.value();
        // vector<size_t> b_ids_v = tB.value();
        if (!tA.is_spatial_nonzero(a_ids_v)) continue;
        if (!tA.is_spin_nonzero(a_ids_v)) continue;

        vector<size_t> a_value_r, b_value_r;
        tA.gen_restricted(a_ids_v, &a_value_r);
        tB.gen_restricted(b_ids_v, &b_value_r);

        size_t dim_common = compute_size(sum_vec);
        size_t dima = compute_size(a_ids_v);
        if (dima <= 0) continue;
        size_t dimb = compute_size(b_ids_v);
        if (dimb <= 0) continue;
        size_t dima_sort = dima / dim_common;
        size_t dimb_sort = dimb / dim_common;

        setValue(&m->a_ids, a_ids_v);
        setValue(&m->b_ids, b_ids_v);
        setValueR(&m->a_ids, a_value_r);
        setValueR(&m->b_ids, b_value_r);

        double* buf_a = new double[dima];
        double* buf_a_sort = new double[dima];
        vector<size_t> a_svalue_r, b_svalue_r;
        vector<size_t> a_svalue, b_svalue;
        vector<IndexName> a_name;
        vector<IndexName> b_name;
        int a_sign =
            sortByValueThenExtSymGroup(m->a_ids, &a_name, &a_svalue,
                                       &a_svalue_r);
        // tA.sortByValueThenExtSymGroup();
        // if (tA.dim()==2) tA.get_ma = true;
        // tA.get(*d_a,a_svalue_r,a_name,buf_a,dima,*k_a_offset);
        // tA.get(*d_a,a_svalue_r,buf_a,dima,*k_a_offset);
        tA.get(a_svalue_r, buf_a, dima);
        // tA.get(*d_a,buf_a,dima,*k_a_offset);
        vector<size_t> a_sort_ids = sort_ids(a_name, m->a_mem_pos);
        tamm_sort(buf_a, buf_a_sort, a_svalue /*tA._value()*/,
                  a_sort_ids /*tA.sort_ids()*/,
                  static_cast<double>(a_sign) /*tA.sign()*/);
        delete[] buf_a;

        double* buf_b = new double[dimb];
        double* buf_b_sort = new double[dimb];
        int b_sign =
            sortByValueThenExtSymGroup(m->b_ids, &b_name, &b_svalue,
                                       &b_svalue_r);
        // tB.sortByValueThenExtSymGroup();
        // if (!tB.isIntermediate()) tB.get_i = true;
        // tB.get(*d_b,b_svalue_r,b_name,buf_b,dimb,*k_b_offset);
        // tB.get(*d_b,b_svalue_r,buf_b,dimb,*k_b_offset);
        tB.get(b_svalue_r, buf_b, dimb);
        // tB.get(*d_b,buf_b,dimb,*k_b_offset);
        // tce_sort(buf_b, buf_b_sort, tB._value(), tB.sort_ids(),
        // (double)tB.sign());
        vector<size_t> b_sort_ids = sort_ids(b_name, m->b_mem_pos);
        tamm_sort(buf_b, buf_b_sort, b_svalue /*tB._value()*/,
                  b_sort_ids /*tB.sort_ids(b_name)*/,
                  static_cast<double>(b_sign) /*(double)tB.sign()*/);
        delete[] buf_b;

        cdgemm('T', 'N', dima_sort, dimb_sort, dim_common, 1.0, buf_a_sort,
               dim_common, buf_b_sort, dim_common, 1.0, buf_c_sort, dima_sort);

        delete[] buf_a_sort;
        delete[] buf_b_sort;
      }  // sum_itr

      setValue(&m->c_ids, c_ids_v);
      setValueR(&m->c_ids, c_ids_v);
      // tC.sortByValueThenExtSymGroup();
      vector<IndexName> c_name;
      vector<size_t> c_svalue_r, c_svalue;
      sortByValueThenExtSymGroup(m->c_ids, &c_name, &c_svalue, &c_svalue_r);
      cp_itr->reset();
      vector<size_t> perm;
      while (cp_itr->next(&perm)) {
        cp_itr->fix_ids_for_copy(&perm);
        vector<IndexName> name;
        vector<size_t> value, value_r;
        // tC.orderIds(perm);
        orderIds(m->c_ids, perm, &name, &value, &value_r);
        if (compareVec<size_t>(tid, value)) {
          double sign = coef * cp_itr->sign();
          std::vector<size_t> cperm = mult_perm(name, m->c_mem_pos);
          std::vector<size_t> cmpval = getMemPosVal(m->c_ids, m->c_mem_pos);
          tamm_sortacc(buf_c_sort, a_c, cmpval, cperm, sign);
        }
      }  // cp_itr
      delete[] buf_c_sort;
    }  // if spatial check
  }    // ext_itr
}  // t_mult

void t_mult2(const Tensor& tC, const Tensor& tA, const Tensor& tB,
             const double coef, const vector<IndexName>& sum_ids,
             IterGroup<triangular>* sum_itr,
             IterGroup<CopyIter>* cp_itr, IterGroup<triangular>* out_itr,
             Multiplication* m) {
  const vector<IndexName>& c_name = id2name(m->c_ids);  // tC.name();
  vector<size_t> out_vec;
  out_itr->reset();
  double anti = 0, add = 0;
  while (out_itr->next(&out_vec)) {
    if (!tC.is_spatial_nonzero(out_vec)) continue;
    if (!tC.is_spin_nonzero(out_vec)) continue;
    if (!tC.is_spin_restricted_nonzero(out_vec)) continue;
    size_t dimc = compute_size(out_vec);
    if (dimc <= 0) continue;
    double* buf_c = new double[dimc];
    memset(buf_c, 0, dimc * sizeof(double));

    t_mult(buf_c, tC, tA, tB, coef, sum_ids, sum_itr, cp_itr, out_vec, m);
    // tce_add_hash_block_(d_c, buf_c, dimc, *k_c_offset, out_vec, c_name);
    tC.add(out_vec, buf_c, dimc);
    delete[] buf_c;
  }
}  // t_mult2

void t_mult3(const Tensor& tC, const Tensor& tA, const Tensor& tB,
             const double coef, const vector<IndexName>& sum_ids,
             IterGroup<triangular>* sum_itr,  IterGroup<CopyIter>* cp_itr,
             IterGroup<triangular>* out_itr, Multiplication* m,
             gmem::Handle sync_ga, int spos) {
  // vector<size_t>& vtab = Table::value();
  const vector<IndexName>& c_ids = id2name(m->c_ids);  // tC.name();
  const vector<IndexName>& a_ids = id2name(m->a_ids);  // tA.name();
  const vector<IndexName>& b_ids = id2name(m->b_ids);  // tB.name();

  // GA initialization
  int nprocs = gmem::ranks();
  int count = 0;
  int taskDim = 1;
  char taskStr[10] = "NXTASK";
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

  vector<size_t> out_vec, sum_vec;
  out_itr->reset();

  bool out_once = true;
  while (out_itr->next(&out_vec) || out_once) {
    out_once = false;
    if (next == count) {  // check if its my task
      if ((tC.is_spatial_nonzero(out_vec)) && (tC.is_spin_nonzero(out_vec)) &&
          (tC.is_spin_restricted_nonzero(out_vec))) {
        vector<int> vtab1(IndexNum);
        for (int i = 0; i < tC.dim(); i++) {
          assert(c_ids[i] < IndexNum);
          vtab1[c_ids[i]] = out_vec[i];
        }
        vector<size_t> c_ids_v = out_vec;

        size_t dimc = compute_size(c_ids_v);
        if (dimc <= 0) continue;
        double* buf_c_sort = new double[dimc];
        memset(buf_c_sort, 0, dimc * sizeof(double));

        sum_itr->reset();
        // bool ONE_TIME = sum_itr->empty();
        bool ONE_TIME = true;
        while (sum_itr->next(&sum_vec) || ONE_TIME) {
          ONE_TIME = false;
          for (int i = 0; i < sum_vec.size(); i++) {
            vtab1[sum_ids[i]] = sum_vec[i];
          }
          vector<size_t> a_ids_v(tA.dim()), b_ids_v(tB.dim());
          for (int i = 0; i < tA.dim(); i++) {
            assert(a_ids[i] < IndexNum);
            a_ids_v[i] = vtab1[a_ids[i]];
          }
          for (int i = 0; i < tB.dim(); i++) {
            assert(b_ids[i] < IndexNum);
            b_ids_v[i] = vtab1[b_ids[i]];
          }
          if (!tA.is_spatial_nonzero(a_ids_v)) continue;
          if (!tA.is_spin_nonzero(a_ids_v)) continue;

#if 0
          if (tC.dim()%2 != 0) {
            cout << "a_ids_v: " << a_ids_v << endl;
            cout << "b_ids_v: " << b_ids_v << endl;
          }
#endif  // debug stuff

          vector<size_t> a_value_r, b_value_r;
          tA.gen_restricted(a_ids_v, &a_value_r);
          tB.gen_restricted(b_ids_v, &b_value_r);

          size_t dim_common = compute_size(sum_vec);
          size_t dima = compute_size(a_ids_v);
          if (dima <= 0) continue;
          size_t dimb = compute_size(b_ids_v);
          if (dimb <= 0) continue;
          size_t dima_sort = dima / dim_common;
          size_t dimb_sort = dimb / dim_common;

          double* buf_a = new double[dima];
          double* buf_a_sort = new double[dima];
          assert(tA.dim() == a_ids_v.size());
          // tA.setValue(a_ids_v);
          // tB.setValue(b_ids_v);
          // tA.setValueR(a_value_r);
          // tB.setValueR(b_value_r);
          setValue(&m->a_ids, a_ids_v);
          setValue(&m->b_ids, b_ids_v);
          setValueR(&m->a_ids, a_value_r);
          setValueR(&m->b_ids, b_value_r);

          vector<size_t> a_svalue_r, b_svalue_r;
          vector<size_t> a_svalue, b_svalue;
          vector<IndexName> a_name;
          vector<IndexName> b_name;
          // int a_sign = tA.sortByValueThenExtSymGroup(a_name, a_svalue,
          // a_svalue_r);
          // int b_sign = tB.sortByValueThenExtSymGroup(b_name, b_svalue,
          // b_svalue_r);
          int a_sign =
              sortByValueThenExtSymGroup(m->a_ids, &a_name, &a_svalue,
                                         &a_svalue_r);
          int b_sign =
              sortByValueThenExtSymGroup(m->b_ids, &b_name, &b_svalue,
                                         &b_svalue_r);

#if 0
          if (tC.dim()%2 != 0) {
            cout << "a_ids_v: " << a_ids_v << endl;
            cout << "b_ids_v: " << b_ids_v << endl;
            cout << "a_value_r: " << a_value_r << endl;
            cout << "b_value_r: " << b_value_r << endl;
            cout << "a_svalue: " << a_svalue << endl;
            cout << "b_svalue: " << b_svalue << endl;
            cout << "a_svalue_r: " << a_svalue_r << endl;
            cout << "b_svalue_r: " << b_svalue_r << endl;
          }
#endif  // debug stuff

          // if (tA.dim() == 2) tA.get_ma = true;
          // tA.get(*d_a,a_svalue_r,a_name,buf_a,dima,*k_a_offset);
          getTimer.start();
          // tA.get(*d_a,a_svalue_r,buf_a,dima,*k_a_offset);
          tA.get(a_svalue_r, buf_a, dima);
          getTimer.stop();
          vector<size_t> a_sort_ids = sort_ids(a_name, m->a_mem_pos);
          // vector<size_t> a_sort_ids = tA.sort_ids(a_name);

          tamm_sort(buf_a, buf_a_sort, a_svalue /*tA._value()*/,
                    a_sort_ids /*tA.sort_ids(a_name)*/,
                    static_cast<double>(a_sign) /*(double)tA.sign()*/);
          delete[] buf_a;

          double* buf_b = new double[dimb];
          double* buf_b_sort = new double[dimb];
          // if (!tB.isIntermediate()) tB.get_i = true;
          // tB.get(*d_b,b_svalue_r,b_name,buf_b,dimb,*k_b_offset);
          getTimer.start();
          // tB.get(*d_b,b_svalue_r,buf_b,dimb,*k_b_offset);
          tB.get(b_svalue_r, buf_b, dimb);
          getTimer.stop();
          // vector<size_t> b_sort_ids = tB.sort_ids(b_name);
          vector<size_t> b_sort_ids = sort_ids(b_name, m->b_mem_pos);
          tamm_sort(buf_b, buf_b_sort, b_svalue /*tB._value()*/,
                    b_sort_ids /*tB.sort_ids(b_name)*/,
                    static_cast<double>(b_sign) /*(double)tB.sign()*/);
          delete[] buf_b;

          double beta = computeBeta(sum_ids, sum_vec);

          dgemmTimer.start();
          cdgemm('T', 'N', dima_sort, dimb_sort, dim_common, beta, buf_a_sort,
                 dim_common, buf_b_sort, dim_common, 1.0, buf_c_sort,
                 dima_sort);
          dgemmTimer.stop();

          delete[] buf_a_sort;
          delete[] buf_b_sort;
        }  // sum_itr

#if 1
        // tC.setValue(c_ids_v);
        // tC.setValueR(c_ids_v);
        setValue(&m->c_ids, c_ids_v);
        setValueR(&m->c_ids, c_ids_v);
#endif  // always
        vector<IndexName> c_name;
        vector<size_t> c_svalue_r, c_svalue;
        // tC.sortByValueThenExtSymGroup(c_name, c_svalue, c_svalue_r);
        sortByValueThenExtSymGroup(m->c_ids, &c_name, &c_svalue, &c_svalue_r);
        // vector<size_t> tid = tC._value();
        vector<size_t> tid = c_svalue;

        cp_itr->reset();
        vector<size_t> perm;
        bool out_cp_one_time = true;
        while (cp_itr->next(&perm) || out_cp_one_time) {
          out_cp_one_time = false;
          cp_itr->fix_ids_for_copy(&perm);
          vector<IndexName> name;
          vector<size_t> value, value_r;
          // tC.orderIds(perm, name, value, value_r);
          orderIds(m->c_ids, perm, &name, &value, &value_r);
          if (compareVec<size_t>(tid, value)) {
            double sign = coef * cp_itr->sign();
            double* buf_c = new double[dimc];
            // std::vector<size_t> cperm = tC.perm(name);
            std::vector<size_t> cperm = mult_perm(name, m->c_mem_pos);
            // std::vector<size_t> cmpval = getMemPosVal(tC.ids(),
            //                                           m->c_mem_pos);
            std::vector<size_t> cmpval = getMemPosVal(m->c_ids, m->c_mem_pos);
            tamm_sort(buf_c_sort, buf_c, cmpval, cperm, sign);

            addTimer.start();
            // tce_add_hash_block_(d_c, buf_c, dimc, *k_c_offset, value, name);
            tC.add(value, buf_c, dimc);
            addTimer.stop();

            delete[] buf_c;
          }
        }  // cp_itr
        delete[] buf_c_sort;
      }  // if spatial symmetry check

      next = static_cast<int>(gmem::atomic_fetch_add(taskHandle, sub, 1));
    }  // if next == count

    ++count;  // no matter my or not, increase by one
  }  // out_itr

  if (!sync_ga.valid()) {
    gmem::sync();               // sync, wait for all procs to finish
    gmem::destroy(taskHandle);  // free
  }
}  // t_mult3

void t_mult4(Multiplication* m, gmem::Handle sync_ga, int spos) {
  multTimer.start();
  if (m->tA().dim() == 0) {
    assert(m->tB().dim() != 0); /** @bug Cannot handle this case yet*/
    double coef;
    vector<size_t> id;
    m->tA().get(id, &coef, 1);
    coef *= m->coef();
    Assignment as(&m->tC(), &m->tB(), coef, id2name(m->c_ids),
                  id2name(m->b_ids));
    t_assign3(as, sync_ga, spos);
  } else if (m->tB().dim() == 0) {
    assert(m->tA().dim() != 0); /** @bug Cannot handle this case yet*/
    double coef;
    vector<size_t> id;
    m->tB().get(id, &coef, 1);
    coef *= m->coef();
    Assignment as(&m->tC(), &m->tA(), coef, id2name(m->c_ids),
                  id2name(m->a_ids));
    t_assign3(as, sync_ga, spos);
    return;
  } else {
    t_mult3(m->tC(), m->tA(), m->tB(), m->coef(), m->sum_ids(), &m->sum_itr(),
            &m->cp_itr(), &m->out_itr(), m, sync_ga, spos);
  }
  multTimer.stop();
}

}  // namespace tamm
