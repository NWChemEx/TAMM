#include "stats.h"
#include "t_mult.h"
using namespace std;

namespace ctce {

  extern "C" {

    void t_mult(Integer* d_a, Integer* k_a_offset,
        Integer* d_b, Integer* k_b_offset, double *a_c,
        Tensor &tC, Tensor &tA, Tensor &tB, const double coef,
        const vector<IndexName>& sum_ids,
        IterGroup <triangular>& sum_itr,
        IterGroup<CopyIter>& cp_itr,
		const vector<Integer>& tid,
		Multiplication& m) {

      //vector<Integer>& vtab = Table::value();
      const vector<IndexName>& c_ids = id2name(m.c_ids);//tC.name();
      const vector<IndexName>& a_ids = id2name(m.a_ids);//tA.name();
      const vector<IndexName>& b_ids = id2name(m.b_ids);//tB.name();

      vector<Integer> vtab(IndexNum);
      for (int i = 0; i < tC.dim(); ++i) {
        vtab[c_ids[i]] = tid[i]; // values used in genAntiIter
      }

      IterGroup<antisymm> ext_itr;
      genAntiIter(vtab, ext_itr,tC,tA,tB);
      vector<Integer> ext_vec, sum_vec;
      ext_itr.reset();

      while (ext_itr.next(ext_vec)) {
        if (tC.is_spatial_nonzero(ext_vec) &&
            tC.is_spin_nonzero(ext_vec) &&
            tC.is_spin_restricted_nonzero(ext_vec)) {

          for (int i = 0; i < tC.dim(); ++i) {
            vtab[c_ids[i]] = ext_vec[i];
            //tC.setValueByName(c_ids[i], vtab[c_ids[i]]);
          }
          //const vector<Integer>& c_ids_v = tC.value();
	  vector<Integer> c_ids_v(tC.dim());
	  for(int i=0; i<tC.dim(); i++) {
	    c_ids_v[i] = vtab[c_ids[i]];
	  }
          Integer dimc = compute_size(c_ids_v);
	  assert(dimc == compute_size(tid));
          if (dimc == 0) continue;
          double* buf_c_sort = new double[dimc];
          memset(buf_c_sort, 0, dimc * sizeof(double));

          sum_itr.reset();
          bool ONE_TIME = sum_itr.empty();
          while (sum_itr.next(sum_vec) || ONE_TIME) {

            ONE_TIME = false;
	    for(int i=0; i<sum_vec.size(); i++) {
	      vtab[sum_ids[i]] = sum_vec[i];
	    }
	    vector<Integer> a_ids_v(tA.dim()), b_ids_v(tB.dim());
	    for(int i=0; i<tA.dim(); i++) {
	      assert(a_ids[i] < IndexNum);
	      a_ids_v[i] = vtab[a_ids[i]];
	    }
	    for(int i=0; i<tB.dim(); i++) {
	      assert(b_ids[i] < IndexNum);
	      b_ids_v[i] = vtab[b_ids[i]];
	    }
            // for (int i=0; i<sum_ids.size(); i++) {
            //   tA.setValueByName(sum_ids[i],sum_vec[i]);
            //   tB.setValueByName(sum_ids[i],sum_vec[i]);
            // }
            // vector<Integer> a_ids_v = tA.value();
            // vector<Integer> b_ids_v = tB.value();
            if(!tA.is_spatial_nonzero(a_ids_v))  continue;
            if(!tA.is_spin_nonzero(a_ids_v)) continue;

	    vector<Integer> a_value_r, b_value_r;
	    tA.gen_restricted(a_ids_v, a_value_r);
	    tB.gen_restricted(b_ids_v, b_value_r);

            Integer dim_common = compute_size(sum_vec);
            Integer dima = compute_size(a_ids_v); if (dima<=0) continue;
            Integer dimb = compute_size(b_ids_v); if (dimb<=0) continue;
            Integer dima_sort = dima / dim_common;
            Integer dimb_sort = dimb / dim_common;

	    setValue(m.a_ids, a_ids_v);
	    setValue(m.b_ids, b_ids_v);
	    setValueR(m.a_ids, a_value_r);
	    setValueR(m.b_ids, b_value_r);

            double* buf_a = new double[dima];
            double* buf_a_sort = new double[dima];
	    vector<Integer> a_svalue_r, b_svalue_r;
	    vector<Integer> a_svalue, b_svalue;
	    vector<IndexName> a_name;
	    vector<IndexName> b_name;
	    int a_sign = sortByValueThenExtSymGroup(m.a_ids, a_name, a_svalue, a_svalue_r);
            //tA.sortByValueThenExtSymGroup();
            // if (tA.dim()==2) tA.get_ma = true;
	    //tA.get(*d_a,a_svalue_r,a_name,buf_a,dima,*k_a_offset);
	    // tA.get(*d_a,a_svalue_r,buf_a,dima,*k_a_offset);
	    tA.get(a_svalue_r,buf_a,dima);
            //tA.get(*d_a,buf_a,dima,*k_a_offset);
	    vector<Integer> a_sort_ids = sort_ids(a_name, m.a_mem_pos);
            tce_sort(buf_a, buf_a_sort, a_svalue /*tA._value()*/, a_sort_ids/*tA.sort_ids()*/, (double)a_sign /*tA.sign()*/);
            delete [] buf_a;

            double* buf_b = new double[dimb];
            double* buf_b_sort = new double[dimb];
	    int b_sign = sortByValueThenExtSymGroup(m.b_ids, b_name, b_svalue, b_svalue_r);
            //tB.sortByValueThenExtSymGroup();
            // if (!tB.isIntermediate()) tB.get_i = true;
	    //tB.get(*d_b,b_svalue_r,b_name,buf_b,dimb,*k_b_offset);
	    // tB.get(*d_b,b_svalue_r,buf_b,dimb,*k_b_offset);
	    tB.get(b_svalue_r,buf_b,dimb);
            //tB.get(*d_b,buf_b,dimb,*k_b_offset);
            //tce_sort(buf_b, buf_b_sort, tB._value(), tB.sort_ids(), (double)tB.sign());
	    vector<Integer> b_sort_ids = sort_ids(b_name, m.b_mem_pos);
	    tce_sort(buf_b, buf_b_sort, b_svalue/*tB._value()*/, b_sort_ids /*tB.sort_ids(b_name)*/, (double)b_sign /*(double)tB.sign()*/);
            delete [] buf_b;

            cdgemm('T','N', dima_sort, dimb_sort, dim_common, 1.0, buf_a_sort,
                dim_common, buf_b_sort, dim_common, 1.0, buf_c_sort, dima_sort);

            delete [] buf_a_sort;
            delete [] buf_b_sort;
          } // sum_itr

	  setValue(m.c_ids, c_ids_v);
	  setValueR(m.c_ids, c_ids_v);
          //tC.sortByValueThenExtSymGroup();
	  vector<IndexName> c_name;
	  vector<Integer> c_svalue_r, c_svalue;
	  sortByValueThenExtSymGroup(m.c_ids, c_name, c_svalue, c_svalue_r);
          cp_itr.reset();
          vector<Integer> perm;
          while (cp_itr.next(perm)) {
            cp_itr.fix_ids_for_copy(perm);
	    vector<IndexName> name;
	    vector<Integer> value, value_r;
            //tC.orderIds(perm);
	    orderIds(m.c_ids, perm, name, value, value_r);
            if (compareVec<Integer>(tid,value)) {
              double sign = coef * cp_itr.sign();
	      std::vector<Integer> cperm = mult_perm(name, m.c_mem_pos);
	      std::vector<Integer> cmpval = getMemPosVal(m.c_ids, m.c_mem_pos);
              sortacc(buf_c_sort, a_c, cmpval, cperm, sign);
            }
          } // cp_itr
          delete [] buf_c_sort;
        } // if spatial check
      } // ext_itr
    } // t_mult


    void t_mult2(Integer* d_a, Integer* k_a_offset, Integer* d_b, Integer* k_b_offset,
        Integer* d_c, Integer* k_c_offset,
        Tensor& tC, Tensor& tA, Tensor& tB, const double coef,
        const vector<IndexName>& sum_ids,
        IterGroup<triangular>& sum_itr,
        IterGroup<CopyIter>& cp_itr,
		 IterGroup<triangular>& out_itr,
		 Multiplication& m) {

      const vector<IndexName>& c_name = id2name(m.c_ids);//tC.name();
      vector<Integer> out_vec;
      out_itr.reset();
      double anti=0, add=0;
      while (out_itr.next(out_vec)) {
        if (!tC.is_spatial_nonzero(out_vec)) continue;
        if (!tC.is_spin_nonzero(out_vec)) continue;
        if (!tC.is_spin_restricted_nonzero(out_vec)) continue;
        Integer dimc = compute_size(out_vec);
        if (dimc <= 0) continue;
        double* buf_c = new double[dimc];
        memset(buf_c, 0, dimc * sizeof(double));

        t_mult(d_a,k_a_offset,d_b,k_b_offset,buf_c,
	       tC,tA,tB,coef,sum_ids,sum_itr,cp_itr,out_vec, m);
        tce_add_hash_block_(d_c, buf_c, dimc, *k_c_offset, out_vec, c_name);
        delete [] buf_c;
      }
    } // t_mult2

    void t_mult3(Integer* d_a, Integer* k_a_offset, Integer* d_b, Integer* k_b_offset,
        Integer* d_c, Integer* k_c_offset,
        Tensor& tC, Tensor& tA, Tensor& tB, const double coef,
        const vector<IndexName>& sum_ids,
        IterGroup<triangular>& sum_itr,
        IterGroup<CopyIter>& cp_itr,
		 IterGroup<triangular>& out_itr,
		 Multiplication& m, int sync_ga, int spos) {

      //vector<Integer>& vtab = Table::value();
      const vector<IndexName>& c_ids = id2name(m.c_ids);//tC.name();
      const vector<IndexName>& a_ids = id2name(m.a_ids);//tA.name();
      const vector<IndexName>& b_ids = id2name(m.b_ids);//tB.name();

      // GA initialization
      int nprocs = GA_Nnodes();
      int count = 0;
      int taskDim = 1;
      char taskStr[10] = "NXTASK";
      int taskHandle;
      int sub;
      if(sync_ga) {
        taskHandle = sync_ga;
        sub = spos;
      }
      else {
        taskHandle = NGA_Create(C_INT,1,&taskDim,taskStr,NULL); // global array for next task
        GA_Zero(taskHandle); // initialize to zero
        GA_Sync();
        sub = 0;
      }

      int next = NGA_Read_inc(taskHandle, &sub, 1);

      vector<Integer> out_vec, sum_vec;
      out_itr.reset();

      bool out_once = true;
      while (out_itr.next(out_vec) || out_once) {
        out_once = false;
        if (next==count) { // check if its my task

          if ( (tC.is_spatial_nonzero(out_vec)) &&
              (tC.is_spin_nonzero(out_vec)) &&
              (tC.is_spin_restricted_nonzero(out_vec)) ) {

	    if(tC.dim()%2!=0) {
	      cout<<"Computing output tensor block: "<<out_vec<<endl;
	    }

	    vector<int> vtab1(IndexNum);
            for (int i=0; i<tC.dim(); i++) {
	      assert(c_ids[i] < IndexNum);
	      vtab1[c_ids[i]] = out_vec[i];
            }
	    vector<Integer> c_ids_v = out_vec;

            Integer dimc = compute_size(c_ids_v); if (dimc<=0) continue;
            double* buf_c_sort = new double[dimc];
            memset(buf_c_sort, 0, dimc*sizeof(double));

            sum_itr.reset();
            //bool ONE_TIME = sum_itr.empty();
            bool ONE_TIME = true;
            while (sum_itr.next(sum_vec) || ONE_TIME) {

              ONE_TIME = false;
	      for(int i=0; i<sum_vec.size(); i++) {
		vtab1[sum_ids[i]] = sum_vec[i];
	      }
	      vector<Integer> a_ids_v(tA.dim()), b_ids_v(tB.dim());
	      for(int i=0; i<tA.dim(); i++) {
		assert(a_ids[i] < IndexNum);
		a_ids_v[i] = vtab1[a_ids[i]];
	      }
	      for(int i=0; i<tB.dim(); i++) {
		assert(b_ids[i] < IndexNum);
		b_ids_v[i] = vtab1[b_ids[i]];
	      }
              if (!tA.is_spatial_nonzero(a_ids_v)) continue;
              if (!tA.is_spin_nonzero(a_ids_v)) continue;

#if 0
	      if(tC.dim()%2!=0) {
		cout<<"a_ids_v: "<<a_ids_v<<endl;
		cout<<"b_ids_v: "<<b_ids_v<<endl;
	      }
#endif
	      
	      vector<Integer> a_value_r, b_value_r;
	      tA.gen_restricted(a_ids_v, a_value_r);
	      tB.gen_restricted(b_ids_v, b_value_r);

              Integer dim_common = compute_size(sum_vec);
              Integer dima = compute_size(a_ids_v); if (dima<=0) continue;
              Integer dimb = compute_size(b_ids_v); if (dimb<=0) continue;
              Integer dima_sort = dima / dim_common;
              Integer dimb_sort = dimb / dim_common;

              double* buf_a = new double[dima];
              double* buf_a_sort = new double[dima];
	      assert(tA.dim() == a_ids_v.size());
	      // tA.setValue(a_ids_v);
	      // tB.setValue(b_ids_v);
	      // tA.setValueR(a_value_r);
	      // tB.setValueR(b_value_r);
	      setValue(m.a_ids, a_ids_v);
	      setValue(m.b_ids, b_ids_v);
	      setValueR(m.a_ids, a_value_r);
	      setValueR(m.b_ids, b_value_r);

	      vector<Integer> a_svalue_r, b_svalue_r;
	      vector<Integer> a_svalue, b_svalue;
	      vector<IndexName> a_name;
	      vector<IndexName> b_name;
              // int a_sign = tA.sortByValueThenExtSymGroup(a_name, a_svalue, a_svalue_r);
              // int b_sign = tB.sortByValueThenExtSymGroup(b_name, b_svalue, b_svalue_r);
              int a_sign = sortByValueThenExtSymGroup(m.a_ids, a_name, a_svalue, a_svalue_r);
              int b_sign = sortByValueThenExtSymGroup(m.b_ids, b_name, b_svalue, b_svalue_r);

#if 0
	      if(tC.dim()%2!=0) {
		cout<<"a_ids_v: "<<a_ids_v<<endl;
		cout<<"b_ids_v: "<<b_ids_v<<endl;
		cout<<"a_value_r: "<<a_value_r<<endl;
		cout<<"b_value_r: "<<b_value_r<<endl;
		cout<<"a_svalue: "<<a_svalue<<endl;
		cout<<"b_svalue: "<<b_svalue<<endl;
		cout<<"a_svalue_r: "<<a_svalue_r<<endl;
		cout<<"b_svalue_r: "<<b_svalue_r<<endl;
	      }
#endif

              // if (tA.dim()==2) tA.get_ma = true;
              //tA.get(*d_a,a_svalue_r,a_name,buf_a,dima,*k_a_offset);
              getTimer.start();
              // tA.get(*d_a,a_svalue_r,buf_a,dima,*k_a_offset);
              tA.get(a_svalue_r,buf_a,dima);
              getTimer.stop();
	      vector<Integer> a_sort_ids = sort_ids(a_name, m.a_mem_pos);
	      //vector<Integer> a_sort_ids = tA.sort_ids(a_name);

              tce_sort(buf_a, buf_a_sort, a_svalue/*tA._value()*/, a_sort_ids/*tA.sort_ids(a_name)*/, (double)a_sign /*(double)tA.sign()*/);
              delete [] buf_a;

              double* buf_b = new double[dimb];
              double* buf_b_sort = new double[dimb];
              // if (!tB.isIntermediate()) tB.get_i = true;
              //tB.get(*d_b,b_svalue_r,b_name,buf_b,dimb,*k_b_offset);
              getTimer.start();
              // tB.get(*d_b,b_svalue_r,buf_b,dimb,*k_b_offset);
              tB.get(b_svalue_r,buf_b,dimb);
              getTimer.stop();
	      //vector<Integer> b_sort_ids = tB.sort_ids(b_name);
	      vector<Integer> b_sort_ids = sort_ids(b_name, m.b_mem_pos);
              tce_sort(buf_b, buf_b_sort, b_svalue/*tB._value()*/, b_sort_ids /*tB.sort_ids(b_name)*/, (double)b_sign /*(double)tB.sign()*/);
              delete [] buf_b;

              double beta = computeBeta(sum_ids,sum_vec);

              dgemmTimer.start();
              cdgemm('T','N',dima_sort, dimb_sort, dim_common, beta, buf_a_sort,
                  dim_common, buf_b_sort, dim_common, 1.0, buf_c_sort, dima_sort);
              dgemmTimer.stop();

              delete [] buf_a_sort;
              delete [] buf_b_sort;
            } // sum_itr

#if 1	
	  // tC.setValue(c_ids_v);
	  // tC.setValueR(c_ids_v);
	    setValue(m.c_ids, c_ids_v);
	    setValueR(m.c_ids, c_ids_v);
#endif
	  vector<IndexName> c_name;
	  vector<Integer> c_svalue_r, c_svalue;
	  // tC.sortByValueThenExtSymGroup(c_name, c_svalue, c_svalue_r);
	  sortByValueThenExtSymGroup(m.c_ids, c_name, c_svalue, c_svalue_r);
	  //vector<Integer> tid = tC._value();
	  vector<Integer> tid = c_svalue;

            cp_itr.reset();
            vector<Integer> perm;
            bool out_cp_one_time = true;
            while (cp_itr.next(perm) || out_cp_one_time) {
              out_cp_one_time = false;
              cp_itr.fix_ids_for_copy(perm);
	      vector<IndexName> name;
	      vector<Integer> value, value_r;
              //tC.orderIds(perm, name, value, value_r);
	      orderIds(m.c_ids, perm, name, value, value_r);
              if (compareVec<Integer>(tid, value)) {
                double sign = coef * cp_itr.sign();
                double* buf_c = new double[dimc];
		//std::vector<Integer> cperm = tC.perm(name);
		std::vector<Integer> cperm = mult_perm(name, m.c_mem_pos);
		// std::vector<Integer> cmpval = getMemPosVal(tC.ids(), m.c_mem_pos);
		std::vector<Integer> cmpval = getMemPosVal(m.c_ids, m.c_mem_pos);
                tce_sort(buf_c_sort, buf_c, cmpval, cperm, sign);

                addTimer.start();
                tce_add_hash_block_(d_c, buf_c, dimc, *k_c_offset, value, name);
                addTimer.stop();

                delete [] buf_c;
              }
            } // cp_itr
            delete [] buf_c_sort;
          } // if spatial symmetry check

          next = NGA_Read_inc(taskHandle, &sub, 1);  // get my next task

        } // if next == count

        ++count; // no matter my or not, increase by one

      } // out_itr

      if(sync_ga==0) {
        GA_Sync(); // sync, wait for all procs to finish
        GA_Destroy(taskHandle); // free
      }
    } // t_mult3

    void t_mult4(Integer* d_a, Integer* k_a_offset, Integer* d_b, Integer* k_b_offset,
                 Integer* d_c, Integer* k_c_offset, Multiplication& m, int sync_ga, int spos) {
      multTimer.start();
      t_mult3(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset,
          m.tC(), m.tA(), m.tB(), m.coef(), m.sum_ids(),
	      m.sum_itr(), m.cp_itr(), m.out_itr(), m, sync_ga, spos);
      multTimer.stop();
    }

  } /*extern "C"*/

}; // namespace ctce
