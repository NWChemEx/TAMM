
#pragma once

// using namespace tamm;
extern double ccsdt_d1_t2_GetTime;
extern double ccsdt_d1_v2_GetTime;
extern double ccsdt_d2_t2_GetTime;
extern double ccsdt_d2_v2_GetTime;
extern double ccsd_t_data_per_rank;

#if defined(USE_CUDA)
//TensorGen target-centric CUDA kernels
void jk_ccsd_t_d1_1(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d1_2(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d1_3(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d1_4(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d1_5(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d1_6(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d1_7(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d1_8(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d1_9(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);

void jk_ccsd_t_d2_1(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d2_2(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d2_3(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d2_4(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d2_5(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d2_6(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d2_7(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d2_8(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void jk_ccsd_t_d2_9(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);

#if defined(USE_NV_TC)
void driver_ccsd_t_d1_1(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d1_2(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d1_3(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d1_4(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d1_5(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d1_6(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d1_7(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d1_8(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d1_9(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2);

void driver_ccsd_t_d2_1(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d2_2(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d2_3(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d2_4(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d2_5(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d2_6(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d2_7(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d2_8(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2);
void driver_ccsd_t_d2_9(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_p7, double* host_t3, double* host_t2, double* host_v2);
#endif

#endif

//target-centric CPU kernels
void sd_t_d1_1_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_2_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_3_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_4_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_5_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_6_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_7_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_8_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_9_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);

void sd_t_d2_1_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_2_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_3_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_4_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_5_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_6_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_7_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_8_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_9_cpu(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);

//NWChem source-centric GPU kernels
void sd_t_d1_1_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_2_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_3_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_4_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_5_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_6_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_7_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_8_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d1_9_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);

void sd_t_d2_1_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_2_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_3_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_4_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_5_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_6_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_7_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_8_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_d2_9_cuda(size_t,size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);

#define USE_A100

template<typename T>
void ccsd_t_doubles_unfused(ExecutionContext& ec,
                   const TiledIndexSpace& MO,
                   const Index noab, const Index nvab,
                   std::vector<int>& k_spin,
                   std::vector<T>& a_c,
                   Tensor<T>& d_t2, //d_a
                   Tensor<T>& d_v2, //d_b
                   std::vector<T>& p_evl_sorted,
                   std::vector<size_t>& k_range,
                   size_t t_h1b, size_t t_h2b, size_t t_h3b,
                   size_t t_p4b, size_t t_p5b, size_t t_p6b,
                   int has_gpu,
                   bool is_restricted, bool use_nwc_gpu_kernels) {

    auto rank = ec.pg().rank();

    Eigen::Matrix<size_t, 9,6, Eigen::RowMajor> a3;
    a3.setZero();

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

    Index p4b,p5b,p6b,h1b,h2b,h3b;
    for (auto ia6=0; ia6<9; ia6++){ 
      p4b=a3(ia6,0);
      p5b=a3(ia6,1);
      p6b=a3(ia6,2);
      h1b=a3(ia6,3);
      h2b=a3(ia6,4);
      h3b=a3(ia6,5);
    
      // cout << "p456,h123= ";
      ////print_varlist(p4b,p5b,p6b,h1b,h2b,h3b);

      // if ((has_gpu==1)&&(notset==1)) {
      //  dev_mem_d(k_range[t_h1b],k_range[t_h2b],
      //               k_range[t_h3b],k_range[t_p4b],
      //               k_range[t_p5b],k_range[t_p6b]);
      //  notset=0;
      // }


    if( (p4b<=p5b) && (h2b<=h3b) && p4b!=0 ){ 
      if((!is_restricted) || (k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         +k_spin[h1b]+k_spin[h2b]+k_spin[h3b]!=12)) {
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
             //print_varlist(k_spin[p4b]+k_spin[p5b], k_spin[h1b]+k_spin[h7b]);

              // cout << "h7b,h1b=";
              ////print_varlist(h7b,h1b);

                  //TODO
                  if(h7b<h1b) {
                    {
                      TimerGuard tg_total{&ccsdt_d1_t2_GetTime};
                      ccsd_t_data_per_rank += dima;
                      d_t2.get({p4b-noab,p5b-noab,h7b,h1b},k_a); //h1b,h7b,p5b-noab,p4b-noab
                    }
                    int perm[4]={3,1,0,2}; //3,1,0,2
                    int size[4]={(int)k_range[p4b],(int)k_range[p5b],(int)k_range[h7b],(int)k_range[h1b]};
                    // int size[4]={k_range[h7b],k_range[p4b],k_range[p5b],k_range[h1b]}; //1,3,2,0
                    // int size[4]={k_range[h1b],k_range[p5b],k_range[p4b],k_range[h7b]}; //0,2,3,1
                    
                    auto plan = hptt::create_plan
                    (perm, 4, -1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }
                  if(h1b<=h7b){
                    {
                      TimerGuard tg_total{&ccsdt_d1_t2_GetTime};
                      ccsd_t_data_per_rank += dima;
                      d_t2.get({p4b-noab,p5b-noab,h1b,h7b},k_a); //h7b,h1b,p5b-noab,p4b-noab
                    }
                    int perm[4]={2,1,0,3}; //2,1,0,3
                    // int size[4]={k_range[p4b],k_range[p5b],k_range[h1b],k_range[h7b]};
                    int size[4]={(int)k_range[p4b],(int)k_range[p5b],(int)k_range[h1b],(int)k_range[h7b]};
                    // int size[4]={k_range[h7b],k_range[p4b],k_range[p5b],k_range[h1b]}; //0,3,2,1
                    // int size[4]={k_range[h1b],k_range[p5b],k_range[p4b],k_range[h7b]}; //1,2,3,0
                    
                    auto plan = hptt::create_plan
                    (perm, 4, 1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }


    if(debug && rank==0) {
      std::cout << "h7b =" << h7b << " h1b,h2b,h3b,p4b,p5b,p6b : t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b" << std::endl;
      print_varlist(h1b,h2b,h3b,p4b,p5b,p6b);
      print_varlist(t_h1b,t_h2b,t_h3b,t_p4b,t_p5b,t_p6b);
    }
    
    std::vector<T> k_b_sort(dimb);
    if(h7b <= p6b) {
      {
        TimerGuard tg_total{&ccsdt_d1_v2_GetTime};
        ccsd_t_data_per_rank += dimb;
        d_v2.get({h7b,p6b,h2b,h3b},k_b_sort); //h3b,h2b,p6b,h7b
      }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
     {
      //print_varlist(1);
      //  cout << k_a_sort << endl;
      if(has_gpu) {
        if(use_nwc_gpu_kernels) 
          sd_t_d1_1_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
        #if defined(USE_CUDA)                  
        else {
          // printf ("[%s] calls jk_ccsd_t_d1_1()\n", __func__);
        #if defined(USE_NV_TC)
          // void driver_ccsd_t_d1_1(int size_h3, int size_h2, int size_h1, int size_p6, int size_p5, int size_p4, int size_h7, double* host_t3, double* host_t2, double* host_v2) 
          driver_ccsd_t_d1_1((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
        #else 
          // void jk_ccsd_t_d1_1(size_t size_h3, size_t size_h2, size_t size_h1, size_t size_p6, size_t size_p5, size_t size_p4, size_t size_h7, double* host_t3, double* host_t2, double* host_v2);
          jk_ccsd_t_d1_1(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
        #endif
        }
        #endif
      }
      else
        sd_t_d1_1_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[h7b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
       //print_varlist(2);
        // cout << k_a_sort << endl;
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d1_2_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);  
          #if defined(USE_CUDA)                  
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d1_2((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d1_2(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else
           sd_t_d1_2_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[h7b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);        
      }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
     {
      // //print_varlist(3);
      if(has_gpu) {
        if(use_nwc_gpu_kernels) 
          sd_t_d1_3_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);     
        #if defined(USE_CUDA)        
        else {
        #if defined(USE_NV_TC)
          driver_ccsd_t_d1_3((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
        #else
          jk_ccsd_t_d1_3(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
        #endif
        }
        #endif
      }
      else         
        sd_t_d1_3_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[h7b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                 
     }

    if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
        ////print_varlist(4);
        if(has_gpu) {
        if(use_nwc_gpu_kernels) 
          sd_t_d1_4_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);    
        #if defined(USE_CUDA)    
        else {
        #if defined(USE_NV_TC)
          driver_ccsd_t_d1_4((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
        #else
          jk_ccsd_t_d1_4(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
        #endif
        }
        #endif
        }
        else   
          sd_t_d1_4_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[h7b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                         
      }

    if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
        ////print_varlist(5);
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d1_5_cuda(k_range[h1b],k_range[h2b],
                      k_range[h3b],k_range[h7b],k_range[p4b],
                      k_range[p5b],k_range[p6b],
                      &a_c[0],&k_a_sort[0],&k_b_sort[0]);         
          #if defined(USE_CUDA)           
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d1_5((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d1_5(k_range[t_h3b],k_range[t_h2b],
                      k_range[t_h1b],k_range[t_p6b],
                      k_range[t_p5b],k_range[t_p4b],k_range[h7b],
                      &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else 
          sd_t_d1_5_cpu(k_range[t_h1b],k_range[t_h2b],
                      k_range[t_h3b],k_range[h7b],k_range[t_p4b],
                      k_range[t_p5b],k_range[t_p6b],
                      &a_c[0],&k_a_sort[0],&k_b_sort[0]);                        
     }
  
   if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
      // //print_varlist(6);
      if(has_gpu) {
        if(use_nwc_gpu_kernels) 
            sd_t_d1_6_cuda(k_range[h1b],k_range[h2b],
                      k_range[h3b],k_range[h7b],k_range[p4b],
                      k_range[p5b],k_range[p6b],
                      &a_c[0],&k_a_sort[0],&k_b_sort[0]);  
          #if defined(USE_CUDA)                
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d1_6((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d1_6(k_range[t_h3b],k_range[t_h2b],
                      k_range[t_h1b],k_range[t_p6b],
                      k_range[t_p5b],k_range[t_p4b],k_range[h7b],
                      &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
      }
      else      
          sd_t_d1_6_cpu(k_range[t_h1b],k_range[t_h2b],
                      k_range[t_h3b],k_range[h7b],k_range[t_p4b],
                      k_range[t_p5b],k_range[t_p6b],
                      &a_c[0],&k_a_sort[0],&k_b_sort[0]);                      
     }

    if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
        ////print_varlist(7);
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d1_7_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);  
          #if defined(USE_CUDA)                  
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d1_7((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d1_7(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else    
          sd_t_d1_7_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[h7b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                        
     }

    if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
        ////print_varlist(8);
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d1_8_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #if defined(USE_CUDA) 
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d1_8((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d1_8(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else  
          sd_t_d1_8_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[h7b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                          
     }

    if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
         {
        ////print_varlist(9);
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d1_9_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #if defined(USE_CUDA)                
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d1_9((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d1_9(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[h7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else           
          sd_t_d1_9_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[h7b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                 
         }

          } // if(h7b <= p6b)
         } // if(dima > 0 && dimb > 0)
        } // spin_p45 == spin_h17
       } // h7b loop 
      } // spin_p456 == spin_h123
     } // spin_p+spin_h != 12 
    } // if( (p4b<=p5b) && (h2b<=h3b) && p4b!=0 )
  } //end ia6

    //--------------------------------BEGIN DOUBLES-2----------------------------//

  #if 1
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
    a3(1,4)=t_h3b;
    a3(1,5)=t_h1b;

    a3(2,0)=t_p4b;
    a3(2,1)=t_p5b;
    a3(2,2)=t_p6b;
    a3(2,3)=t_h1b;
    a3(2,4)=t_h3b;
    a3(2,5)=t_h2b;

    a3(3,0)=t_p5b;
    a3(3,1)=t_p4b;
    a3(3,2)=t_p6b;
    a3(3,3)=t_h1b;
    a3(3,4)=t_h2b;
    a3(3,5)=t_h3b;

    a3(4,0)=t_p5b;
    a3(4,1)=t_p4b;
    a3(4,2)=t_p6b;
    a3(4,3)=t_h2b;
    a3(4,4)=t_h3b;
    a3(4,5)=t_h1b;

    a3(5,0)=t_p5b;
    a3(5,1)=t_p4b;
    a3(5,2)=t_p6b;
    a3(5,3)=t_h1b;
    a3(5,4)=t_h3b;
    a3(5,5)=t_h2b;

    a3(6,0)=t_p6b;
    a3(6,1)=t_p4b;
    a3(6,2)=t_p5b;
    a3(6,3)=t_h1b;
    a3(6,4)=t_h2b;
    a3(6,5)=t_h3b;

    a3(7,0)=t_p6b;
    a3(7,1)=t_p4b;
    a3(7,2)=t_p5b;
    a3(7,3)=t_h2b;
    a3(7,4)=t_h3b;
    a3(7,5)=t_h1b;

    a3(8,0)=t_p6b;
    a3(8,1)=t_p4b;
    a3(8,2)=t_p5b;
    a3(8,3)=t_h1b;
    a3(8,4)=t_h3b;
    a3(8,5)=t_h2b;


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

    // int p4b,p5b,p6b,h1b,h2b,h3b;
    for (auto ia6=0; ia6<9; ia6++){ 
      p4b=a3(ia6,0);
      p5b=a3(ia6,1);
      p6b=a3(ia6,2);
      h1b=a3(ia6,3);
      h2b=a3(ia6,4);
      h3b=a3(ia6,5);
    
      if( (p5b<=p6b) && (h1b<=h2b) && p4b!=0 ) { 
      if((!is_restricted) || (k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         +k_spin[h1b]+k_spin[h2b]+k_spin[h3b]!=12)) {
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
                    {
                      TimerGuard tg_total{&ccsdt_d2_t2_GetTime};
                      ccsd_t_data_per_rank += dima;
                      d_t2.get({p7b-noab,p4b-noab,h1b,h2b},k_a); //h2b,h1b,p4b-noab,p7b-noab
                    }
                    // for (auto x=0;x<dima;x++) k_a_sort[x] = -1 * k_a[x];
                    int perm[4]={3,2,1,0};
                    int size[4]={(int)k_range[p7b],(int)k_range[p4b],(int)k_range[h1b],(int)k_range[h2b]};
                    
                    auto plan = hptt::create_plan
                    (perm, 4, -1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }
                  if(p4b<=p7b) {
                    {
                      TimerGuard tg_total{&ccsdt_d2_t2_GetTime};
                      ccsd_t_data_per_rank += dima;
                      d_t2.get({p4b-noab,p7b-noab,h1b,h2b},k_a); //h2b,h1b,p7b-noab,p4b-noab
                    }
                    int perm[4]={3,2,0,1}; //0,1,3,2
                    int size[4]={(int)k_range[p4b],(int)k_range[p7b],(int)k_range[h1b],(int)k_range[h2b]};
                    
                    auto plan = hptt::create_plan
                    (perm, 4, 1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }

    std::vector<T> k_b_sort(dimb);
    if(h3b <= p7b) {
      {
        TimerGuard tg_total{&ccsdt_d2_v2_GetTime};
        ccsd_t_data_per_rank += dimb;      
        d_v2.get({p5b,p6b,h3b,p7b},k_b_sort); //p7b,h3b,p6b,p5b
      }


    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
     {
       if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d2_1_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);     
          #if defined(USE_CUDA)              
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d2_1((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d2_1(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else
          sd_t_d2_1_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);        
     }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d2_2_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #if defined(USE_CUDA)
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d2_2((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d2_2(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else          
          sd_t_d2_2_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                  
      }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b)) 
     {
       if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d2_3_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);        
          #if defined(USE_CUDA)           
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d2_3((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d2_3(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else                
          sd_t_d2_3_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);            
     }

    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d2_4_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);  
          #if defined(USE_CUDA)                  
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d2_4((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d2_4(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else          
          sd_t_d2_4_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);              
      }

    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d2_5_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);     
          #if defined(USE_CUDA)               
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d2_5((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d2_5(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else   
          sd_t_d2_5_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                         
     }
  
    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b))
     {
       if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d2_6_cuda(k_range[h1b],k_range[h2b],
                      k_range[h3b],k_range[p4b],
                      k_range[p5b],k_range[p6b],k_range[p7b],
                      &a_c[0],&k_a_sort[0],&k_b_sort[0]); 
          #if defined(USE_CUDA)                  
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d2_6((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d2_6(k_range[t_h3b],k_range[t_h2b],
                      k_range[t_h1b],k_range[t_p6b],
                      k_range[t_p5b],k_range[t_p4b],k_range[p7b],
                      &a_c[0],&k_a_sort[0],&k_b_sort[0]);    
          #endif
          }
          #endif
        }
        else                    
          sd_t_d2_6_cpu(k_range[t_h1b],k_range[t_h2b],
                      k_range[t_h3b],k_range[t_p4b],
                      k_range[t_p5b],k_range[t_p6b],k_range[p7b],
                      &a_c[0],&k_a_sort[0],&k_b_sort[0]);           
     }

    if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d2_7_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]); 
          #if defined(USE_CUDA)                   
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d2_7((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d2_7(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else     
          sd_t_d2_7_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                       
     }

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d2_8_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);   
          #if defined(USE_CUDA)                 
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d2_8((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d2_8(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else  
          sd_t_d2_8_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                          
     }

    if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b))
        {
        if(has_gpu) {
          if(use_nwc_gpu_kernels) 
            sd_t_d2_9_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);    
          #if defined(USE_CUDA)      
          else {
          #if defined(USE_NV_TC)
            driver_ccsd_t_d2_9((int)k_range[t_h3b],(int)k_range[t_h2b],
                    (int)k_range[t_h1b],(int)k_range[t_p6b],
                    (int)k_range[t_p5b],(int)k_range[t_p4b],(int)k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #else
            jk_ccsd_t_d2_9(k_range[t_h3b],k_range[t_h2b],
                    k_range[t_h1b],k_range[t_p6b],
                    k_range[t_p5b],k_range[t_p4b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
          #endif
          }
          #endif
        }
        else         
          sd_t_d2_9_cpu(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);                   
        }

         } // if(h3b <= p7b)
        } // if(dima > 0 && dimb > 0)
       } // spin_p47 == spin_h12
      } // p7b loop 
     } // spin_p456 == spin_h123
    } // spin_p+spin_h != 12 
   } // if( (p5b<=p6b) && (h1b<=h2b) && p4b!=0 )
  } //end ia6  
  #endif
} //end double_gpu_driver

