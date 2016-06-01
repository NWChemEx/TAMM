#include "t_assign.h"
using namespace std;

namespace ctce {

  extern "C" {

    void t_assign2(
        Integer* d_a, Integer* k_a_offset,
        Integer* d_c, Integer* k_c_offset,
        Tensor& tC, const vector<IndexName> &c_ids,
	Tensor& tA, const vector<IndexName> &a_ids,
	IterGroup<triangular>& out_itr, double coef) {

      //const vector<IndexName>& c_ids = cids;//tC.name();
      //const vector<IndexName>& a_ids = aids;//tA.name();
      //vector<Integer>& vtab = Table::value();
      vector<Integer> order(tC.dim());

      for (int i = 0; i < tC.dim(); ++i) order[i] = find(c_ids.begin(), c_ids.end(), a_ids[i]) - c_ids.begin() + 1;

      int nprocs = GA_Nnodes();
      int count = 0;

      int taskDim = 1;
      char taskStr[10] = "NXTASKA";
      int taskHandle = NGA_Create(C_INT, 1, &taskDim, taskStr, NULL); // global array for next task
      GA_Zero(taskHandle); // initialize to zero

      GA_Sync();

      int sub = 0;
      int next = NGA_Read_inc(taskHandle, &sub, 1);

      vector<Integer> out_vec; // out_vec = c_ids_v
      out_itr.reset();
      while (out_itr.next(out_vec)) {

        if (next == count) {

	  vector<int> vtab1(IndexNum);
          for (int i = 0; i < tC.dim(); ++i) {
	    assert(c_ids[i] < IndexNum);
            vtab1[c_ids[i]] = out_vec[i];
          }
	  vector<Integer> a_ids_v(tA.dim());
          for (int i = 0; i < tA.dim(); ++i) {
	    assert(a_ids[i] < IndexNum);
            a_ids_v[i] = vtab1[a_ids[i]];
          }
          if (is_spatial_nonzero(out_vec, tA.irrep()) &&
              is_spin_nonzero(a_ids_v) &&
              is_spin_restricted_nonzero(out_vec, 2 * tC.dim())) {

            Integer dimc = compute_size(out_vec); if (dimc <= 0) continue;
	    vector<Integer> value_r;
            tA.gen_restricted(a_ids_v, value_r);

            double* buf_a = new double[dimc];
            double* buf_a_sort = new double[dimc];
            tA.get2(*d_a, value_r, buf_a, dimc, *k_a_offset); // get2 is for t_assign use

            if (coef == -1.0) { // dscal
              for (int i = 0; i < dimc; ++i) buf_a[i] = -buf_a[i];
            }
            tce_sort(buf_a, buf_a_sort, a_ids_v, order, 1.0);
            tce_add_hash_block_(d_c, buf_a_sort, dimc, *k_c_offset, out_vec, c_ids);

            delete [] buf_a;
            delete [] buf_a_sort;
          }
          int sub = 0;
          next = NGA_Read_inc(taskHandle, &sub, 1);
        }
        ++count;
      }

      GA_Sync();
      GA_Destroy(taskHandle);
    }

    void t_assign3(
        Integer* d_a, Integer* k_a_offset,
        Integer* d_c, Integer* k_c_offset, Assignment& a) {
      t_assign2(d_a, k_a_offset, d_c, k_c_offset, a.tC(), a.cids(), a.tA(), a.aids(), a.out_itr(), a.coef());
    }

  } // extern C

}; // namespace ctce
