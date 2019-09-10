
#ifndef CCSD_T_GPU_ALL_FUSED_HPP_
#define CCSD_T_GPU_ALL_FUSED_HPP_

#include "tamm/tamm.hpp"
// using namespace tamm;

void initmemmodule();
void dev_mem_s(size_t,size_t,size_t,size_t,size_t,size_t);
void dev_mem_d(size_t,size_t,size_t,size_t,size_t,size_t);

void total_fused_ccsd_t(size_t base_size_h1b, size_t base_size_h2b, size_t base_size_h3b, 
						size_t base_size_p4b, size_t base_size_p5b, size_t base_size_p6b,
						// 
						double* host_d1_t2_all, double* host_d1_v2_all,
						double* host_d2_t2_all, double* host_d2_v2_all,
						double* host_s1_t2_all, double* host_s1_v2_all,
						// 
						size_t size_d1_t2_all, size_t size_d1_v2_all,
						size_t size_d2_t2_all, size_t size_d2_v2_all,
						size_t size_s1_t2_all, size_t size_s1_v2_all,
						// 
						size_t* list_d1_sizes, 
						size_t* list_d2_sizes, 
						size_t* list_s1_sizes, 
						// 
						std::vector<int> vec_d1_flags,
						std::vector<int> vec_d2_flags,
						std::vector<int> vec_s1_flags,
						// 
						size_t size_noab, size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
						size_t size_nvab, size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
                                          size_t size_max_dim_s1_t2, size_t size_max_dim_s1_v2, 
						// 
						double factor, 
						double* host_evl_sortedh1, double* host_evl_sortedh2, double* host_evl_sortedh3, 
						double* host_evl_sortedp4, double* host_evl_sortedp5, double* host_evl_sortedp6,
						double* final_energy_4, double* final_energy_5);

template<typename T>
void ccsd_t_gpu_all_fused(ExecutionContext& ec,
                   const TiledIndexSpace& MO,
                   const Index noab, const Index nvab,
                   std::vector<int>& k_spin,
                   std::vector<size_t>& k_offset,
                   std::vector<T>& a_c, //not used
                   Tensor<T>& d_t1, 
                   Tensor<T>& d_t2, //d_a
                   Tensor<T>& d_v2, //d_b
                   std::vector<T>& k_evl_sorted,
                   std::vector<size_t>& k_range,
                   size_t t_h1b, size_t t_h2b, size_t t_h3b,
                   size_t t_p4b, size_t t_p5b, size_t t_p6b,
                   std::vector<T>& k_abufs1, std::vector<T>& k_bbufs1,
                   std::vector<T>& k_abuf1, std::vector<T>& k_bbuf1,
                   std::vector<T>& k_abuf2, std::vector<T>& k_bbuf2,                   
                   double& factor, std::vector<double>& energy_l, int usedevice) {

    // initmemmodule();
    size_t abufs1_size = k_abufs1.size();
    size_t bbufs1_size = k_bbufs1.size();

    size_t abuf_size1 = k_abuf1.size();
    size_t bbuf_size1 = k_bbuf1.size();
    size_t abuf_size2 = k_abuf2.size();
    size_t bbuf_size2 = k_bbuf2.size();

    Eigen::Matrix<size_t, 9,6, Eigen::RowMajor> a3;
    Eigen::Matrix<size_t, 9,6, Eigen::RowMajor> a3_s1;
    Eigen::Matrix<size_t, 9,6, Eigen::RowMajor> a3_d2;
    a3.setZero();
    a3_d2.setZero();
    a3_s1.setZero();

    a3_s1(0,0)=t_p4b;
    a3_s1(0,1)=t_p5b;
    a3_s1(0,2)=t_p6b;
    a3_s1(0,3)=t_h1b;
    a3_s1(0,4)=t_h2b;
    a3_s1(0,5)=t_h3b;

    a3_s1(1,0)=t_p4b;
    a3_s1(1,1)=t_p5b;
    a3_s1(1,2)=t_p6b;
    a3_s1(1,3)=t_h2b;
    a3_s1(1,4)=t_h1b;
    a3_s1(1,5)=t_h3b;

    a3_s1(2,0)=t_p4b;
    a3_s1(2,1)=t_p5b;
    a3_s1(2,2)=t_p6b;
    a3_s1(2,3)=t_h3b;
    a3_s1(2,4)=t_h1b;
    a3_s1(2,5)=t_h2b;

    a3_s1(3,0)=t_p5b;
    a3_s1(3,1)=t_p4b;
    a3_s1(3,2)=t_p6b;
    a3_s1(3,3)=t_h1b;
    a3_s1(3,4)=t_h2b;
    a3_s1(3,5)=t_h3b;

    a3_s1(4,0)=t_p5b;
    a3_s1(4,1)=t_p4b;
    a3_s1(4,2)=t_p6b;
    a3_s1(4,3)=t_h2b;
    a3_s1(4,4)=t_h1b;
    a3_s1(4,5)=t_h3b;

    a3_s1(5,0)=t_p5b;
    a3_s1(5,1)=t_p4b;
    a3_s1(5,2)=t_p6b;
    a3_s1(5,3)=t_h3b;
    a3_s1(5,4)=t_h1b;
    a3_s1(5,5)=t_h2b;

    a3_s1(6,0)=t_p6b;
    a3_s1(6,1)=t_p4b;
    a3_s1(6,2)=t_p5b;
    a3_s1(6,3)=t_h1b;
    a3_s1(6,4)=t_h2b;
    a3_s1(6,5)=t_h3b;

    a3_s1(7,0)=t_p6b;
    a3_s1(7,1)=t_p4b;
    a3_s1(7,2)=t_p5b;
    a3_s1(7,3)=t_h2b;
    a3_s1(7,4)=t_h1b;
    a3_s1(7,5)=t_h3b;

    a3_s1(8,0)=t_p6b;
    a3_s1(8,1)=t_p4b;
    a3_s1(8,2)=t_p5b;
    a3_s1(8,3)=t_h3b;
    a3_s1(8,4)=t_h1b;
    a3_s1(8,5)=t_h2b;

    for (auto ia6=0; ia6<8; ia6++){
      if(a3_s1(ia6,0) != 0) {
      for (auto ja6=ia6+1;ja6<9;ja6++) { //TODO: ja6 start ?
      if((a3_s1(ia6,0) == a3_s1(ja6,0)) && (a3_s1(ia6,1) == a3_s1(ja6,1))
       && (a3_s1(ia6,2) == a3_s1(ja6,2)) && (a3_s1(ia6,3) == a3_s1(ja6,3))
       && (a3_s1(ia6,4) == a3_s1(ja6,4)) && (a3_s1(ia6,5) == a3_s1(ja6,5)))
       {
        a3_s1(ja6,0)=0;
        a3_s1(ja6,1)=0;
        a3_s1(ja6,2)=0;
        a3_s1(ja6,3)=0;
        a3_s1(ja6,4)=0;
        a3_s1(ja6,5)=0;
      }
      } 
      }
    }
      
  
    a3(0,0)=t_p4b;
    a3(0,1)=t_p5b;
    a3(0,2)=t_p6b;
    a3(0,3)=t_h1b;
    a3(0,4)=t_h2b;
    a3(0,5)=t_h3b;

    a3(1,0)=t_p4b;
    a3(1,1)=t_p5b;
    a3(1,2)=t_p6b;
    a3(1,3)=t_h2b;
    a3(1,4)=t_h1b;
    a3(1,5)=t_h3b;

    a3(2,0)=t_p4b;
    a3(2,1)=t_p5b;
    a3(2,2)=t_p6b;
    a3(2,3)=t_h3b;
    a3(2,4)=t_h1b;
    a3(2,5)=t_h2b;

    a3(3,0)=t_p5b;
    a3(3,1)=t_p6b;
    a3(3,2)=t_p4b;
    a3(3,3)=t_h1b;
    a3(3,4)=t_h2b;
    a3(3,5)=t_h3b;

    a3(4,0)=t_p5b;
    a3(4,1)=t_p6b;
    a3(4,2)=t_p4b;
    a3(4,3)=t_h2b;
    a3(4,4)=t_h1b;
    a3(4,5)=t_h3b;

    a3(5,0)=t_p5b;
    a3(5,1)=t_p6b;
    a3(5,2)=t_p4b;
    a3(5,3)=t_h3b;
    a3(5,4)=t_h1b;
    a3(5,5)=t_h2b;

    a3(6,0)=t_p4b;
    a3(6,1)=t_p6b;
    a3(6,2)=t_p5b;
    a3(6,3)=t_h1b;
    a3(6,4)=t_h2b;
    a3(6,5)=t_h3b;

    a3(7,0)=t_p4b;
    a3(7,1)=t_p6b;
    a3(7,2)=t_p5b;
    a3(7,3)=t_h2b;
    a3(7,4)=t_h1b;
    a3(7,5)=t_h3b;

    a3(8,0)=t_p4b;
    a3(8,1)=t_p6b;
    a3(8,2)=t_p5b;
    a3(8,3)=t_h3b;
    a3(8,4)=t_h1b;
    a3(8,5)=t_h2b;

    // auto notset=1;

    for (auto ia6=0; ia6<8; ia6++){
      if(a3(ia6,0) != 0) {
      for (auto ja6=ia6+1;ja6<9;ja6++) { //TODO: ja6 start ?
      if((a3(ia6,0) == a3(ja6,0)) && (a3(ia6,1) == a3(ja6,1))
       && (a3(ia6,2) == a3(ja6,2)) && (a3(ia6,3) == a3(ja6,3))
       && (a3(ia6,4) == a3(ja6,4)) && (a3(ia6,5) == a3(ja6,5)))
       {
        a3(ja6,0)=0;
        a3(ja6,1)=0;
        a3(ja6,2)=0;
        a3(ja6,3)=0;
        a3(ja6,4)=0;
        a3(ja6,5)=0;
      }
      } 
      }
    }

    a3_d2(0,0)=t_p4b;
    a3_d2(0,1)=t_p5b;
    a3_d2(0,2)=t_p6b;
    a3_d2(0,3)=t_h1b;
    a3_d2(0,4)=t_h2b;
    a3_d2(0,5)=t_h3b;

    a3_d2(1,0)=t_p4b;
    a3_d2(1,1)=t_p5b;
    a3_d2(1,2)=t_p6b;
    a3_d2(1,3)=t_h2b;
    a3_d2(1,4)=t_h3b;
    a3_d2(1,5)=t_h1b;

    a3_d2(2,0)=t_p4b;
    a3_d2(2,1)=t_p5b;
    a3_d2(2,2)=t_p6b;
    a3_d2(2,3)=t_h1b;
    a3_d2(2,4)=t_h3b;
    a3_d2(2,5)=t_h2b;

    a3_d2(3,0)=t_p5b;
    a3_d2(3,1)=t_p4b;
    a3_d2(3,2)=t_p6b;
    a3_d2(3,3)=t_h1b;
    a3_d2(3,4)=t_h2b;
    a3_d2(3,5)=t_h3b;

    a3_d2(4,0)=t_p5b;
    a3_d2(4,1)=t_p4b;
    a3_d2(4,2)=t_p6b;
    a3_d2(4,3)=t_h2b;
    a3_d2(4,4)=t_h3b;
    a3_d2(4,5)=t_h1b;

    a3_d2(5,0)=t_p5b;
    a3_d2(5,1)=t_p4b;
    a3_d2(5,2)=t_p6b;
    a3_d2(5,3)=t_h1b;
    a3_d2(5,4)=t_h3b;
    a3_d2(5,5)=t_h2b;

    a3_d2(6,0)=t_p6b;
    a3_d2(6,1)=t_p4b;
    a3_d2(6,2)=t_p5b;
    a3_d2(6,3)=t_h1b;
    a3_d2(6,4)=t_h2b;
    a3_d2(6,5)=t_h3b;

    a3_d2(7,0)=t_p6b;
    a3_d2(7,1)=t_p4b;
    a3_d2(7,2)=t_p5b;
    a3_d2(7,3)=t_h2b;
    a3_d2(7,4)=t_h3b;
    a3_d2(7,5)=t_h1b;

    a3_d2(8,0)=t_p6b;
    a3_d2(8,1)=t_p4b;
    a3_d2(8,2)=t_p5b;
    a3_d2(8,3)=t_h1b;
    a3_d2(8,4)=t_h3b;
    a3_d2(8,5)=t_h2b;


    for (auto ia6=0; ia6<8; ia6++){
      if(a3_d2(ia6,0) != 0) {
      for (auto ja6=ia6+1;ja6<9;ja6++) { //TODO: ja6 start ?
      if((a3_d2(ia6,0) == a3_d2(ja6,0)) && (a3_d2(ia6,1) == a3_d2(ja6,1))
       && (a3_d2(ia6,2) == a3_d2(ja6,2)) && (a3_d2(ia6,3) == a3_d2(ja6,3))
       && (a3_d2(ia6,4) == a3_d2(ja6,4)) && (a3_d2(ia6,5) == a3_d2(ja6,5)))
       {
        a3_d2(ja6,0)=0;
        a3_d2(ja6,1)=0;
        a3_d2(ja6,2)=0;
        a3_d2(ja6,3)=0;
        a3_d2(ja6,4)=0;
        a3_d2(ja6,5)=0;
      }
      } 
      }
    }

    Index p4b,p5b,p6b,h1b,h2b,h3b;
    const size_t max_dima = abuf_size1 /  (9*noab);
    const size_t max_dimb = bbuf_size1 /  (9*noab);
    const size_t max_dima2 = abuf_size2 / (9*nvab);
    const size_t max_dimb2 = bbuf_size2 / (9*nvab);

    const size_t s1_max_dima = abufs1_size / 9;
    const size_t s1_max_dimb = bbufs1_size / 9;

    //singles
    std::vector<int> sd_t_s1_exec(9*9,0);
    std::vector<size_t> sd_t_s1_args(9*9*6);
    size_t s1c = 0;
    size_t s1b = 0;
    size_t s1e = 0;

    //doubles 1
    std::vector<int> sd_t_d1_exec(9*9*noab,0);
    std::vector<size_t> sd_t_d1_args(9*9*7*noab);
    size_t d1c = 0;
    size_t d1b = 0;
    size_t d1e = 0;

    //doubles 2
    std::vector<int> sd_t_d2_exec(9*9*nvab,0);
    std::vector<size_t> sd_t_d2_args(9*9*7*nvab);
    size_t d2b=0;
    size_t d2c=0;
    size_t d2e=0;
        
    for (auto ia6=0; ia6<9; ia6++){ 

      p4b=a3_s1(ia6,0);
      p5b=a3_s1(ia6,1);
      p6b=a3_s1(ia6,2);
      h1b=a3_s1(ia6,3);
      h2b=a3_s1(ia6,4);
      h3b=a3_s1(ia6,5);


      if( (p5b<=p6b) && (h2b<=h3b) && p4b!=0){ 
      if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         +k_spin[h1b]+k_spin[h2b]+k_spin[h3b]!=12){

          //  cout << "spin1,2 = ";
          //  dprint(k_spin[p4b]+k_spin[p5b]+k_spin[p6b], k_spin[h1b]+k_spin[h2b]+k_spin[h3b]);

         if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         == k_spin[h1b]+k_spin[h2b]+k_spin[h3b]) {

           if(k_spin[p4b] == k_spin[h1b]){

           size_t dim_common = 1;
           size_t dima_sort = k_range[p4b]*k_range[h1b];
           size_t dima = dim_common * dima_sort;
           size_t dimb_sort=k_range[p5b]*k_range[p6b]*
                        k_range[h2b]*k_range[h3b];
           size_t dimb = dim_common * dimb_sort;
          if(dima>0 && dimb>0){

          //  cout << "spin1,2 = ";
          //  dprint(k_spin[p4b]+k_spin[p5b]+k_spin[p6b], k_spin[h1b]+k_spin[h2b]+k_spin[h3b]);

            std::vector<T> k_a(dima);
            std::vector<T> k_a_sort(dima);
            //TODO 
            IndexVector bids = {p4b-noab,h1b};
            d_t1.get(bids,k_a);

            const int ndim = 2;
            int perm[ndim]={1,0};
            int size[ndim]={k_range[p4b],k_range[h1b]};
            
            // create a plan (shared_ptr)
            auto plan = hptt::create_plan(perm, ndim, 1, &k_a[0], size, NULL, 0, &k_a_sort[0],
                                          NULL, hptt::ESTIMATE, 1, NULL, true);
            plan->execute();

            std::vector<T> k_b_sort(dimb);
            d_v2.get({p5b,p6b,h2b,h3b},k_b_sort); //h3b,h2b,p6b,p5b

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
      && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
     {
        // dprint(1);
        // sd_t_s1_1_cuda(k_range[h1b],k_range[h2b],
        //             k_range[h3b],k_range[p4b],
        //             k_range[p5b],k_range[p6b],
        //             &a_c[0],&k_a_sort[0],&k_b_sort[0]);

       sd_t_s1_exec[s1e] = 1;

       sd_t_s1_args[s1c++] = k_range[h1b];
       sd_t_s1_args[s1c++] = k_range[h2b];
       sd_t_s1_args[s1c++] = k_range[h3b];
       sd_t_s1_args[s1c++] = k_range[p4b];
       sd_t_s1_args[s1c++] = k_range[p5b];
       sd_t_s1_args[s1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abufs1.begin() + s1b*s1_max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbufs1.begin() + s1b*s1_max_dimb);
       s1b++;                    
       
     }
     s1e++;

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
        // dprint(2);
          //  sd_t_s1_2_cuda(k_range[h1b],k_range[h2b],
          //           k_range[h3b],k_range[p4b],
          //           k_range[p5b],k_range[p6b],
          //           &a_c[0],&k_a_sort[0],&k_b_sort[0]);
       sd_t_s1_exec[s1e] = 1;

       sd_t_s1_args[s1c++] = k_range[h1b];
       sd_t_s1_args[s1c++] = k_range[h2b];
       sd_t_s1_args[s1c++] = k_range[h3b];
       sd_t_s1_args[s1c++] = k_range[p4b];
       sd_t_s1_args[s1c++] = k_range[p5b];
       sd_t_s1_args[s1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abufs1.begin() + s1b*s1_max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbufs1.begin() + s1b*s1_max_dimb);
       s1b++;                              
      }
      s1e++;

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
      //  dprint(3);
      //  sd_t_s1_3_cuda(k_range[h1b],k_range[h2b],
      //               k_range[h3b],k_range[p4b],
      //               k_range[p5b],k_range[p6b],
      //               &a_c[0],&k_a_sort[0],&k_b_sort[0]);
       sd_t_s1_exec[s1e] = 1;

       sd_t_s1_args[s1c++] = k_range[h1b];
       sd_t_s1_args[s1c++] = k_range[h2b];
       sd_t_s1_args[s1c++] = k_range[h3b];
       sd_t_s1_args[s1c++] = k_range[p4b];
       sd_t_s1_args[s1c++] = k_range[p5b];
       sd_t_s1_args[s1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abufs1.begin() + s1b*s1_max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbufs1.begin() + s1b*s1_max_dimb);
       s1b++;                          
     }
     s1e++;

     if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
      && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
        // dprint(4);
        // sd_t_s1_4_cuda(k_range[h1b],k_range[h2b],
        //         k_range[h3b],k_range[p4b],
        //         k_range[p5b],k_range[p6b],
        //         &a_c[0],&k_a_sort[0],&k_b_sort[0]);
       sd_t_s1_exec[s1e] = 1;

       sd_t_s1_args[s1c++] = k_range[h1b];
       sd_t_s1_args[s1c++] = k_range[h2b];
       sd_t_s1_args[s1c++] = k_range[h3b];
       sd_t_s1_args[s1c++] = k_range[p4b];
       sd_t_s1_args[s1c++] = k_range[p5b];
       sd_t_s1_args[s1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abufs1.begin() + s1b*s1_max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbufs1.begin() + s1b*s1_max_dimb);
       s1b++;                            
      }
      s1e++;

    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b)) 
      {
        // dprint(5);
      //  sd_t_s1_5_cuda(k_range[h1b],k_range[h2b],
      //               k_range[h3b],k_range[p4b],
      //               k_range[p5b],k_range[p6b],
      //               &a_c[0],&k_a_sort[0],&k_b_sort[0]);
       sd_t_s1_exec[s1e] = 1;

       sd_t_s1_args[s1c++] = k_range[h1b];
       sd_t_s1_args[s1c++] = k_range[h2b];
       sd_t_s1_args[s1c++] = k_range[h3b];
       sd_t_s1_args[s1c++] = k_range[p4b];
       sd_t_s1_args[s1c++] = k_range[p5b];
       sd_t_s1_args[s1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abufs1.begin() + s1b*s1_max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbufs1.begin() + s1b*s1_max_dimb);
       s1b++;                          
     }
     s1e++;
  
    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
      //  dprint(6);
          // sd_t_s1_6_cuda(k_range[h1b],k_range[h2b],
          //                k_range[h3b],k_range[p4b],
          //                k_range[p5b],k_range[p6b],
          //              &a_c[0],&k_a_sort[0],&k_b_sort[0]);
       sd_t_s1_exec[s1e] = 1;

       sd_t_s1_args[s1c++] = k_range[h1b];
       sd_t_s1_args[s1c++] = k_range[h2b];
       sd_t_s1_args[s1c++] = k_range[h3b];
       sd_t_s1_args[s1c++] = k_range[p4b];
       sd_t_s1_args[s1c++] = k_range[p5b];
       sd_t_s1_args[s1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abufs1.begin() + s1b*s1_max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbufs1.begin() + s1b*s1_max_dimb);
       s1b++;                              
     }
     s1e++;

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
      && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
      // dprint(7);
        // sd_t_s1_7_cuda(k_range[h1b],k_range[h2b],
        //             k_range[h3b],k_range[p4b],
        //             k_range[p5b],k_range[p6b],
        //             &a_c[0],&k_a_sort[0],&k_b_sort[0]);
       sd_t_s1_exec[s1e] = 1;

       sd_t_s1_args[s1c++] = k_range[h1b];
       sd_t_s1_args[s1c++] = k_range[h2b];
       sd_t_s1_args[s1c++] = k_range[h3b];
       sd_t_s1_args[s1c++] = k_range[p4b];
       sd_t_s1_args[s1c++] = k_range[p5b];
       sd_t_s1_args[s1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abufs1.begin() + s1b*s1_max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbufs1.begin() + s1b*s1_max_dimb);
       s1b++;                            
     }
     s1e++;

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
      && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b)) 
      {
        // dprint(8);
        // sd_t_s1_8_cuda(k_range[h1b],k_range[h2b],
        //             k_range[h3b],k_range[p4b],
        //             k_range[p5b],k_range[p6b],
        //             &a_c[0],&k_a_sort[0],&k_b_sort[0]);
       sd_t_s1_exec[s1e] = 1;

       sd_t_s1_args[s1c++] = k_range[h1b];
       sd_t_s1_args[s1c++] = k_range[h2b];
       sd_t_s1_args[s1c++] = k_range[h3b];
       sd_t_s1_args[s1c++] = k_range[p4b];
       sd_t_s1_args[s1c++] = k_range[p5b];
       sd_t_s1_args[s1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abufs1.begin() + s1b*s1_max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbufs1.begin() + s1b*s1_max_dimb);
       s1b++;                            
     }
     s1e++;

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
      && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
      {
        // dprint(9);
        //  sd_t_s1_9_cuda(k_range[h1b],k_range[h2b],
        //             k_range[h3b],k_range[p4b],
        //             k_range[p5b],k_range[p6b],
        //             &a_c[0],&k_a_sort[0],&k_b_sort[0]);
       sd_t_s1_exec[s1e] = 1;

       sd_t_s1_args[s1c++] = k_range[h1b];
       sd_t_s1_args[s1c++] = k_range[h2b];
       sd_t_s1_args[s1c++] = k_range[h3b];
       sd_t_s1_args[s1c++] = k_range[p4b];
       sd_t_s1_args[s1c++] = k_range[p5b];
       sd_t_s1_args[s1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abufs1.begin() + s1b*s1_max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbufs1.begin() + s1b*s1_max_dimb);
       s1b++;                                        
     }
     s1e++;

    } //if(dima>0 && dimb>0)
    } //spin
    }} //spin checks
    } //if( (p5b<=p6b) && (h2b<=h3b) && p4b!=0)
    
      p4b=a3(ia6,0);
      p5b=a3(ia6,1);
      p6b=a3(ia6,2);
      h1b=a3(ia6,3);
      h2b=a3(ia6,4);
      h3b=a3(ia6,5);

    if( (p4b<=p5b) && (h2b<=h3b) && p4b!=0){ 
      if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         +k_spin[h1b]+k_spin[h2b]+k_spin[h3b]!=12){
         if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         == k_spin[h1b]+k_spin[h2b]+k_spin[h3b]) {

          //  size_t dimc=k_range[p4b]*k_range[p5b]*k_range[p6b]*
          //            k_range[h1b]*k_range[h2b]*k_range[h3b];

          for (Index h7b=0;h7b<noab;h7b++){

            if(k_spin[p4b]+k_spin[p5b]
              == k_spin[h1b]+k_spin[h7b]) {

                size_t dim_common = k_range[h7b];
                size_t dima_sort = k_range[p4b]*k_range[p5b]*k_range[h1b];
                size_t dima = dim_common*dima_sort;
                size_t dimb_sort = k_range[p6b]*k_range[h2b]*k_range[h3b];
                size_t dimb = dim_common*dimb_sort;
                
                if(dima > 0 && dimb > 0) {
                  std::vector<T> k_a(dima);
                  std::vector<T> k_a_sort(dima);

              // cout << "spin1,2 = ";
             //dprint(k_spin[p4b]+k_spin[p5b], k_spin[h1b]+k_spin[h7b]);

              // cout << "h7b,h1b=";
              ////dprint(h7b,h1b);

                  //TODO
                  if(h7b<h1b) {

                    d_t2.get({p4b-noab,p5b-noab,h7b,h1b},k_a); //h1b,h7b,p5b-noab,p4b-noab
                    int perm[4]={3,1,0,2}; //3,1,0,2
                    int size[4]={k_range[p4b],k_range[p5b],k_range[h7b],k_range[h1b]};
                    // int size[4]={k_range[h7b],k_range[p4b],k_range[p5b],k_range[h1b]}; //1,3,2,0
                    // int size[4]={k_range[h1b],k_range[p5b],k_range[p4b],k_range[h7b]}; //0,2,3,1
                    
                    auto plan = hptt::create_plan
                    (perm, 4, -1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }
                  if(h1b<=h7b){

                    d_t2.get({p4b-noab,p5b-noab,h1b,h7b},k_a); //h7b,h1b,p5b-noab,p4b-noab
                    int perm[4]={2,1,0,3}; //2,1,0,3
                    // int size[4]={k_range[p4b],k_range[p5b],k_range[h1b],k_range[h7b]};
                    int size[4]={k_range[p4b],k_range[p5b],k_range[h1b],k_range[h7b]};
                    // int size[4]={k_range[h7b],k_range[p4b],k_range[p5b],k_range[h1b]}; //0,3,2,1
                    // int size[4]={k_range[h1b],k_range[p5b],k_range[p4b],k_range[h7b]}; //1,2,3,0
                    
                    auto plan = hptt::create_plan
                    (perm, 4, 1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }


    std::vector<T> k_b_sort(dimb);
    if(h7b <= p6b){
      d_v2.get({h7b,p6b,h2b,h3b},k_b_sort); //h3b,h2b,p6b,h7b

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
     {
      //dprint(1);
      //  cout << k_a_sort << endl;
       sd_t_d1_exec[d1e] = 1;

       sd_t_d1_args[d1c++] = k_range[h1b];
       sd_t_d1_args[d1c++] = k_range[h2b];
       sd_t_d1_args[d1c++] = k_range[h3b];
       sd_t_d1_args[d1c++] = k_range[h7b];
       sd_t_d1_args[d1c++] = k_range[p4b];
       sd_t_d1_args[d1c++] = k_range[p5b];
       sd_t_d1_args[d1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + d1b*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + d1b*max_dimb);
       d1b++;
     }
     d1e++;

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
       //dprint(2);
        // cout << k_a_sort << endl;
       sd_t_d1_exec[d1e] = 1;

       sd_t_d1_args[d1c++] = k_range[h1b];
       sd_t_d1_args[d1c++] = k_range[h2b];
       sd_t_d1_args[d1c++] = k_range[h3b];
       sd_t_d1_args[d1c++] = k_range[h7b];
       sd_t_d1_args[d1c++] = k_range[p4b];
       sd_t_d1_args[d1c++] = k_range[p5b];
       sd_t_d1_args[d1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + d1b*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + d1b*max_dimb);
       d1b++;
      }
      d1e++;

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
     {
      // //dprint(3);
       sd_t_d1_exec[d1e] = 1;

       sd_t_d1_args[d1c++] = k_range[h1b];
       sd_t_d1_args[d1c++] = k_range[h2b];
       sd_t_d1_args[d1c++] = k_range[h3b];
       sd_t_d1_args[d1c++] = k_range[h7b];
       sd_t_d1_args[d1c++] = k_range[p4b];
       sd_t_d1_args[d1c++] = k_range[p5b];
       sd_t_d1_args[d1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + d1b*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + d1b*max_dimb);
       d1b++;
     }
     d1e++;

    if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
        ////dprint(4);
       sd_t_d1_exec[d1e] = 1;

       sd_t_d1_args[d1c++] = k_range[h1b];
       sd_t_d1_args[d1c++] = k_range[h2b];
       sd_t_d1_args[d1c++] = k_range[h3b];
       sd_t_d1_args[d1c++] = k_range[h7b];
       sd_t_d1_args[d1c++] = k_range[p4b];
       sd_t_d1_args[d1c++] = k_range[p5b];
       sd_t_d1_args[d1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + d1b*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + d1b*max_dimb);
       d1b++;
      }
      d1e++;

    if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
        ////dprint(5);
       sd_t_d1_exec[d1e] = 1;

       sd_t_d1_args[d1c++] = k_range[h1b];
       sd_t_d1_args[d1c++] = k_range[h2b];
       sd_t_d1_args[d1c++] = k_range[h3b];
       sd_t_d1_args[d1c++] = k_range[h7b];
       sd_t_d1_args[d1c++] = k_range[p4b];
       sd_t_d1_args[d1c++] = k_range[p5b];
       sd_t_d1_args[d1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + d1b*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + d1b*max_dimb);
       d1b++;
     }
     d1e++;
  
   if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
      // //dprint(6);
       sd_t_d1_exec[d1e] = 1;

       sd_t_d1_args[d1c++] = k_range[h1b];
       sd_t_d1_args[d1c++] = k_range[h2b];
       sd_t_d1_args[d1c++] = k_range[h3b];
       sd_t_d1_args[d1c++] = k_range[h7b];
       sd_t_d1_args[d1c++] = k_range[p4b];
       sd_t_d1_args[d1c++] = k_range[p5b];
       sd_t_d1_args[d1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + d1b*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + d1b*max_dimb);
       d1b++;
     }
     d1e++;

    if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
        ////dprint(7);
       sd_t_d1_exec[d1e] = 1;

       sd_t_d1_args[d1c++] = k_range[h1b];
       sd_t_d1_args[d1c++] = k_range[h2b];
       sd_t_d1_args[d1c++] = k_range[h3b];
       sd_t_d1_args[d1c++] = k_range[h7b];
       sd_t_d1_args[d1c++] = k_range[p4b];
       sd_t_d1_args[d1c++] = k_range[p5b];
       sd_t_d1_args[d1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + d1b*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + d1b*max_dimb);
       d1b++;
     }
     d1e++;

    if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
        ////dprint(8);
       sd_t_d1_exec[d1e] = 1;

       sd_t_d1_args[d1c++] = k_range[h1b];
       sd_t_d1_args[d1c++] = k_range[h2b];
       sd_t_d1_args[d1c++] = k_range[h3b];
       sd_t_d1_args[d1c++] = k_range[h7b];
       sd_t_d1_args[d1c++] = k_range[p4b];
       sd_t_d1_args[d1c++] = k_range[p5b];
       sd_t_d1_args[d1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + d1b*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + d1b*max_dimb);
       d1b++;
     }
     d1e++;

    if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
      {
        ////dprint(9);
       sd_t_d1_exec[d1e] = 1;

       sd_t_d1_args[d1c++] = k_range[h1b];
       sd_t_d1_args[d1c++] = k_range[h2b];
       sd_t_d1_args[d1c++] = k_range[h3b];
       sd_t_d1_args[d1c++] = k_range[h7b];
       sd_t_d1_args[d1c++] = k_range[p4b];
       sd_t_d1_args[d1c++] = k_range[p5b];
       sd_t_d1_args[d1c++] = k_range[p6b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + d1b*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + d1b*max_dimb);
       d1b++;
     }
     d1e++;

    } //if(h7b <= p6b)
    } //if(dima > 0 && dimb > 0)
    } //kspin
    } //h7b
    }} //spin checks
    } //if( (p4b<=p5b) && (h2b<=h3b) && p4b!=0)

    //} //end ia6

    //--------------------------------BEGIN DOUBLES-2----------------------------//

    // int p4b,p5b,p6b,h1b,h2b,h3b;
    // for (auto ia6=0; ia6<9; ia6++){ 
      p4b=a3_d2(ia6,0);
      p5b=a3_d2(ia6,1);
      p6b=a3_d2(ia6,2);
      h1b=a3_d2(ia6,3);
      h2b=a3_d2(ia6,4);
      h3b=a3_d2(ia6,5);

      if( (p5b<=p6b) && (h1b<=h2b) && p4b!=0){ 
      if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         +k_spin[h1b]+k_spin[h2b]+k_spin[h3b]!=12){
         if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         == k_spin[h1b]+k_spin[h2b]+k_spin[h3b]) {

          //  size_t dimc=k_range[p4b]*k_range[p5b]*k_range[p6b]*
          //            k_range[h1b]*k_range[h2b]*k_range[h3b];

          for (Index p7b=noab;p7b<noab+nvab;p7b++){

            if(k_spin[p4b]+k_spin[p7b]
              == k_spin[h1b]+k_spin[h2b]) {

                size_t dim_common = k_range[p7b];
                size_t dima_sort = k_range[p4b]*k_range[h1b]*k_range[h2b];
                size_t dima = dim_common*dima_sort;
                size_t dimb_sort = k_range[p5b]*k_range[p6b]*k_range[h3b];
                size_t dimb = dim_common*dimb_sort;
                
                if(dima > 0 && dimb > 0) {
                  std::vector<T> k_a(dima);
                  std::vector<T> k_a_sort(dima);

                  if(p7b<p4b) {

                    d_t2.get({p7b-noab,p4b-noab,h1b,h2b},k_a); //h2b,h1b,p4b-noab,p7b-noab
                    // for (auto x=0;x<dima;x++) k_a_sort[x] = -1 * k_a[x];
                    int perm[4]={3,2,1,0};
                    int size[4]={k_range[p7b],k_range[p4b],k_range[h1b],k_range[h2b]};
                    
                    auto plan = hptt::create_plan
                    (perm, 4, -1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }
                  if(p4b<=p7b) {

                    d_t2.get({p4b-noab,p7b-noab,h1b,h2b},k_a); //h2b,h1b,p7b-noab,p4b-noab
                    int perm[4]={3,2,0,1}; //0,1,3,2
                    int size[4]={k_range[p4b],k_range[p7b],k_range[h1b],k_range[h2b]};
                    
                    auto plan = hptt::create_plan
                    (perm, 4, 1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }

    std::vector<T> k_b_sort(dimb);
    if(h3b <= p7b){
      d_v2.get({p5b,p6b,h3b,p7b},k_b_sort); //p7b,h3b,p6b,p5b


    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
     {
       sd_t_d2_exec[d2e] = 1;

       sd_t_d2_args[d2c++] = k_range[h1b];
       sd_t_d2_args[d2c++] = k_range[h2b];
       sd_t_d2_args[d2c++] = k_range[h3b];
       sd_t_d2_args[d2c++] = k_range[p4b];
       sd_t_d2_args[d2c++] = k_range[p5b];
       sd_t_d2_args[d2c++] = k_range[p6b];
       sd_t_d2_args[d2c++] = k_range[p7b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + d2b*max_dima2);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + d2b*max_dimb2);
       d2b++;
  
     }
     d2e++;

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
       sd_t_d2_exec[d2e] = 1;

       sd_t_d2_args[d2c++] = k_range[h1b];
       sd_t_d2_args[d2c++] = k_range[h2b];
       sd_t_d2_args[d2c++] = k_range[h3b];
       sd_t_d2_args[d2c++] = k_range[p4b];
       sd_t_d2_args[d2c++] = k_range[p5b];
       sd_t_d2_args[d2c++] = k_range[p6b];
       sd_t_d2_args[d2c++] = k_range[p7b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + d2b*max_dima2);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + d2b*max_dimb2);
       d2b++;
      }
      d2e++;

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b)) 
     {
       sd_t_d2_exec[d2e] = 1;

       sd_t_d2_args[d2c++] = k_range[h1b];
       sd_t_d2_args[d2c++] = k_range[h2b];
       sd_t_d2_args[d2c++] = k_range[h3b];
       sd_t_d2_args[d2c++] = k_range[p4b];
       sd_t_d2_args[d2c++] = k_range[p5b];
       sd_t_d2_args[d2c++] = k_range[p6b];
       sd_t_d2_args[d2c++] = k_range[p7b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + d2b*max_dima2);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + d2b*max_dimb2);
       d2b++;
     }
     d2e++;

    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
       sd_t_d2_exec[d2e] = 1;

       sd_t_d2_args[d2c++] = k_range[h1b];
       sd_t_d2_args[d2c++] = k_range[h2b];
       sd_t_d2_args[d2c++] = k_range[h3b];
       sd_t_d2_args[d2c++] = k_range[p4b];
       sd_t_d2_args[d2c++] = k_range[p5b];
       sd_t_d2_args[d2c++] = k_range[p6b];
       sd_t_d2_args[d2c++] = k_range[p7b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + d2b*max_dima2);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + d2b*max_dimb2);
       d2b++;
      }

    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
       sd_t_d2_exec[d2e] = 1;

       sd_t_d2_args[d2c++] = k_range[h1b];
       sd_t_d2_args[d2c++] = k_range[h2b];
       sd_t_d2_args[d2c++] = k_range[h3b];
       sd_t_d2_args[d2c++] = k_range[p4b];
       sd_t_d2_args[d2c++] = k_range[p5b];
       sd_t_d2_args[d2c++] = k_range[p6b];
       sd_t_d2_args[d2c++] = k_range[p7b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + d2b*max_dima2);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + d2b*max_dimb2);
       d2b++;
     }
     d2e++;
  
    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b))
     {
       sd_t_d2_exec[d2e] = 1;

       sd_t_d2_args[d2c++] = k_range[h1b];
       sd_t_d2_args[d2c++] = k_range[h2b];
       sd_t_d2_args[d2c++] = k_range[h3b];
       sd_t_d2_args[d2c++] = k_range[p4b];
       sd_t_d2_args[d2c++] = k_range[p5b];
       sd_t_d2_args[d2c++] = k_range[p6b];
       sd_t_d2_args[d2c++] = k_range[p7b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + d2b*max_dima2);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + d2b*max_dimb2);
       d2b++;
     }
     d2e++;

    if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
       sd_t_d2_exec[d2e] = 1;

       sd_t_d2_args[d2c++] = k_range[h1b];
       sd_t_d2_args[d2c++] = k_range[h2b];
       sd_t_d2_args[d2c++] = k_range[h3b];
       sd_t_d2_args[d2c++] = k_range[p4b];
       sd_t_d2_args[d2c++] = k_range[p5b];
       sd_t_d2_args[d2c++] = k_range[p6b];
       sd_t_d2_args[d2c++] = k_range[p7b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + d2b*max_dima2);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + d2b*max_dimb2);
       d2b++;
     }
     d2e++;

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
       sd_t_d2_exec[d2e] = 1;

       sd_t_d2_args[d2c++] = k_range[h1b];
       sd_t_d2_args[d2c++] = k_range[h2b];
       sd_t_d2_args[d2c++] = k_range[h3b];
       sd_t_d2_args[d2c++] = k_range[p4b];
       sd_t_d2_args[d2c++] = k_range[p5b];
       sd_t_d2_args[d2c++] = k_range[p6b];
       sd_t_d2_args[d2c++] = k_range[p7b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + d2b*max_dima2);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + d2b*max_dimb2);
       d2b++;
     }
     d2e++;

    if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b))
      {
       sd_t_d2_exec[d2e] = 1;

       sd_t_d2_args[d2c++] = k_range[h1b];
       sd_t_d2_args[d2c++] = k_range[h2b];
       sd_t_d2_args[d2c++] = k_range[h3b];
       sd_t_d2_args[d2c++] = k_range[p4b];
       sd_t_d2_args[d2c++] = k_range[p5b];
       sd_t_d2_args[d2c++] = k_range[p6b];
       sd_t_d2_args[d2c++] = k_range[p7b];
       
       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + d2b*max_dima2);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + d2b*max_dimb2);
       d2b++;
     }
     d2e++;

    } //if(h3b <= p7b)
    } //if(dima > 0 && dimb > 0)
    } //if(k_spin[p4b]+k_spin[p7b] == k_spin[h1b]+k_spin[h2b])
    } //end p7b
    }} //spin checks
    } //if( (p5b<=p6b) && (h1b<=h2b) && p4b!=0)

    } //end ia6

total_fused_ccsd_t(k_range[t_h1b],k_range[t_h2b],
                        k_range[t_h3b],k_range[t_p4b],
                        k_range[t_p5b],k_range[t_p6b],
                        k_abuf1.data(), k_bbuf1.data(),
                        k_abuf2.data(), k_bbuf2.data(),
                        k_abufs1.data(), k_bbufs1.data(),
                        abuf_size1,bbuf_size1,
                        abuf_size2,bbuf_size2,
                        abufs1_size, bbufs1_size,
                        sd_t_d1_args.data(), 
                        sd_t_d2_args.data(), 
                        sd_t_s1_args.data(), 
                        sd_t_d1_exec,
                        sd_t_d2_exec,
                        sd_t_s1_exec,
                        noab, max_dima,max_dimb,
                        nvab, max_dima2,max_dimb2,
                              s1_max_dima, s1_max_dimb, 
                        factor, 
                        &k_evl_sorted[k_offset[t_h1b]],
                        &k_evl_sorted[k_offset[t_h2b]],
                        &k_evl_sorted[k_offset[t_h3b]],
                        &k_evl_sorted[k_offset[t_p4b]],
                        &k_evl_sorted[k_offset[t_p5b]],
                        &k_evl_sorted[k_offset[t_p6b]],
                        &energy_l[0], &energy_l[1]);

} //end double_gpu_fused_driver

#endif //CCSD_T_GPU_ALL_FUSED_HPP_