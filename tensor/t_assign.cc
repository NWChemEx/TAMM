#include "stats.h"
#include "t_assign.h"

using namespace std;

namespace ctce {

void t_assign2(
  Tensor& tC, const vector<IndexName> &c_ids,
	Tensor& tA, const vector<IndexName> &a_ids,
	IterGroup<triangular>& out_itr, double coef,
  gmem::Handle sync_ga,
  int spos) {

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
  if(sync_ga.valid()) {
    taskHandle = sync_ga;
    sub = spos;
  }
  else {
    taskHandle = gmem::create(gmem::Int, 1, taskStr); // global array for next task
    gmem::zero(taskHandle); // initialize to zero
    gmem::sync();
    sub = 0;
  }

  int next = (int) gmem::atomic_fetch_add(taskHandle, sub, 1);

  vector<size_t> out_vec; // out_vec = c_ids_v
  out_itr.reset();
  while (out_itr.next(out_vec)) {

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
      if (tA.is_spatial_nonzero(out_vec) &&
          tA.is_spin_nonzero(a_ids_v) &&
          tA.is_spin_restricted_nonzero(out_vec)) {

        size_t dimc = compute_size(out_vec); if (dimc <= 0) continue;
        vector<size_t> value_r;
        tA.gen_restricted(a_ids_v, value_r);

        double* buf_a = new double[dimc];
        double* buf_a_sort = new double[dimc];
        getTimer.start();
        tA.get(value_r, buf_a, dimc);
        getTimer.stop();

//        if (coef == -1.0) { // dscal
//          for (int i = 0; i < dimc; ++i) buf_a[i] = -buf_a[i];
//          ctce_sort(buf_a, buf_a_sort, a_ids_v, order, coef);
//        }
//        else {
          ctce_sort(buf_a, buf_a_sort, a_ids_v, order, coef);
//        }
        addTimer.start();
        tC.add(out_vec, buf_a_sort, dimc);
        addTimer.stop();

        delete [] buf_a;
        delete [] buf_a_sort;
      }
      next = (int) gmem::atomic_fetch_add(taskHandle, sub, 1);
    }
    ++count;
  }

  if(!sync_ga.valid()) {
    gmem::sync();
    gmem::destroy(taskHandle);
  }
}

void t_assign3(Assignment& a, gmem::Handle  sync_ga, int spos) {
  assignTimer.start();
  t_assign2(a.tC(), a.cids(), a.tA(), a.aids(), a.out_itr(),
            a.coef(), sync_ga, spos);
  assignTimer.stop();
}

} // namespace ctce
