
#ifndef CCSD_T_SINGLES_GPU_HPP_
#define CCSD_T_SINGLES_GPU_HPP_



// #include "header.hpp"
#include "tamm/tamm.hpp"
using namespace tamm;


void initmemmodule();
void dev_mem_s(size_t,size_t,size_t,size_t,size_t,size_t);
void dev_mem_d(size_t,size_t,size_t,size_t,size_t,size_t);

void sd_t_s1_1_cuda(size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_s1_2_cuda(size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_s1_3_cuda(size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_s1_4_cuda(size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_s1_5_cuda(size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_s1_6_cuda(size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_s1_7_cuda(size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_s1_8_cuda(size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);
void sd_t_s1_9_cuda(size_t,size_t,size_t,size_t,size_t,size_t,double*,double*,double*);


// template <typename Arg, typename... Args>
// void dprint(Arg&& arg, Args&&... args)
// {
//     cout << std::forward<Arg>(arg);
//     ((cout << ',' << std::forward<Args>(args)), ...);
//     cout << "\n";
// }

// template<typename T>
using T=double;
void ccsd_t_singles_gpu(ExecutionContext& ec,
                   const TiledIndexSpace& MO,
                   const Index noab, const Index nvab,
                   std::vector<int>& k_spin,
                   std::vector<T>& a_c,
                   Tensor<T>& d_t1, 
                   Tensor<T>& d_v2,
                   std::vector<T>& p_evl_sorted,
                   std::vector<size_t>& k_range, 
                   size_t t_h1b, size_t t_h2b, size_t t_h3b, 
                   size_t t_p4b, size_t t_p5b, size_t t_p6b, int usedevice=1) {

    initmemmodule();

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
    a3(3,1)=t_p4b;
    a3(3,2)=t_p6b;
    a3(3,3)=t_h1b;
    a3(3,4)=t_h2b;
    a3(3,5)=t_h3b;

    a3(4,0)=t_p5b;
    a3(4,1)=t_p4b;
    a3(4,2)=t_p6b;
    a3(4,3)=t_h2b;
    a3(4,4)=t_h1b;
    a3(4,5)=t_h3b;

    a3(5,0)=t_p5b;
    a3(5,1)=t_p4b;
    a3(5,2)=t_p6b;
    a3(5,3)=t_h3b;
    a3(5,4)=t_h1b;
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
    a3(7,4)=t_h1b;
    a3(7,5)=t_h3b;

    a3(8,0)=t_p6b;
    a3(8,1)=t_p4b;
    a3(8,2)=t_p5b;
    a3(8,3)=t_h3b;
    a3(8,4)=t_h1b;
    a3(8,5)=t_h2b;

    // auto notset=1;

    // cout << "a3 = " << a3 << endl; correct
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

    // cout << "a3 = " << a3 << endl; correct

    Index p4b,p5b,p6b,h1b,h2b,h3b;

    for (auto ia6=0; ia6<9; ia6++){ 
      p4b=a3(ia6,0);
      p5b=a3(ia6,1);
      p6b=a3(ia6,2);
      h1b=a3(ia6,3);
      h2b=a3(ia6,4);
      h3b=a3(ia6,5);

      // cout << "p456,h123= ";
      // dprint(p4b,p5b,p6b,h1b,h2b,h3b);
    

      // if ((usedevice==1)&&(notset==1)) {
      //  dev_mem_s(k_range[t_h1b],k_range[t_h2b],
      //               k_range[t_h3b],k_range[t_p4b],
      //               k_range[t_p5b],k_range[t_p6b]);
      //  notset=0;
      // }

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
        sd_t_s1_1_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
       
     }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
        // dprint(2);
           sd_t_s1_2_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
      }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
      //  dprint(3);
       sd_t_s1_3_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

     if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
      && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
        // dprint(4);
        sd_t_s1_4_cuda(k_range[h1b],k_range[h2b],
                k_range[h3b],k_range[p4b],
                k_range[p5b],k_range[p6b],
                &a_c[0],&k_a_sort[0],&k_b_sort[0]);
      }

    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b)) 
      {
        // dprint(5);
       sd_t_s1_5_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }
  
    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
      //  dprint(6);
          sd_t_s1_6_cuda(k_range[h1b],k_range[h2b],
                         k_range[h3b],k_range[p4b],
                         k_range[p5b],k_range[p6b],
                       &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
      && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
      // dprint(7);
        sd_t_s1_7_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
      && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b)) 
      {
        // dprint(8);
        sd_t_s1_8_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
      && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
      {
        // dprint(9);
         sd_t_s1_9_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

    }
    }
         }
    }
    }
         
  } //for ia6

} //end ccsd_t_singles

#endif //CCSD_T_SINGLES_GPU_HPP_