
#ifndef CCSD_T_SINGLES_GPU_HPP_
#define CCSD_T_SINGLES_GPU_HPP_



// #include "header.hpp"
#include "tamm/tamm.hpp"
using namespace tamm;


void initmemmodule();
void dev_mem_s(int h1d, int h2d, int h3d, int p4d, int p5d,int p6d);
void dev_mem_d(int h1d, int h2d, int h3d, int p4d, int p5d,int p6d);

// template<typename T>
using T=double;
void ccsd_t_singles_gpu(ExecutionContext& ec,
                   const TiledIndexSpace& MO,
                   std::vector<T>& a_c,
                   Tensor<T>& d_t1, 
                   Tensor<T>& d_v2,
                   std::vector<T>& p_evl_sorted,
                   std::vector<int>& k_range, int t_h1b, int t_h2b,
                   int t_h3b, int t_p4b, int t_p5b, 
                   int t_p6b, int usedevice=1) {

    initmemmodule();

    Eigen::Matrix<int, 9,6, Eigen::RowMajor> a3;
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

    auto notset=1;

    for (auto ia6=0; ia6<8; ia6++){
      if(a3(ia6,1) != 0) {
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

    int p4b,p5b,p6b,h1b,h2b,h3b;
   

    for (auto ia6=0; ia6<8; ia6++){ 
      p4b=a3(ia6,0);
      p5b=a3(ia6,1);
      p6b=a3(ia6,2);
      h1b=a3(ia6,3);
      h2b=a3(ia6,4);
      h3b=a3(ia6,5);
    

      if ((usedevice==1)&&(notset==1)) {
       dev_mem_s(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b]);
       notset=0;
      }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
      && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
     {
        // call sd_t_s1_1_cuda(k_range[h1b],k_range[h2b],
        //             k_range[h3b],k_range[p4b],
        //             k_range[p5b],k_range[p6b],
        //             a_c,&k_a_sort[0],&k_b_sort[0]);
        ;
     }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
          //  sd_t_s1_2_cuda(k_range[h1b],k_range[h2b],
          //           k_range[h3b],k_range[p4b],
          //           k_range[p5b],k_range[p6b],
          //           a_c,&k_a_sort[0],&k_b_sort[0]);
      }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
      //  sd_t_s1_3_cuda(k_range[h1b],k_range[h2b],
      //               k_range[h3b],k_range[p4b],
      //               k_range[p5b],k_range[p6b],
      //               a_c,&k_a_sort[0],&k_b_sort[0]);
     }

     if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
      && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
        // sd_t_s1_4_cuda(k_range[h1b],k_range[h2b],
        //         k_range[h3b],k_range[p4b],
        //         k_range[p5b],k_range[p6b],
        //         a_c,&k_a_sort[0],&k_b_sort[0]);
      }

    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b)) 
      {
      //  sd_t_s1_5_cuda(k_range[h1b],k_range[h2b],
      //               k_range[h3b],k_range[p4b],
      //               k_range[p5b],k_range[p6b],
      //               a_c,&k_a_sort[0],&k_b_sort[0]);
     }
  
    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
      && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
          // sd_t_s1_6_cuda(k_range[h1b],k_range[h2b],
          //                k_range[h3b],k_range[p4b],
          //                k_range[p5b],k_range[p6b],
          //              a_c,&k_a_sort[0],&k_b_sort[0]);
     }

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
      && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {

        // sd_t_s1_7_cuda(k_range[h1b],k_range[h2b],
        //             k_range[h3b],k_range[p4b],
        //             k_range[p5b],k_range[p6b],
        //             a_c,&k_a_sort[0],&k_b_sort[0]);
     }

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
      && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b)) 
      {
        // sd_t_s1_8_cuda(k_range[h1b],k_range[h2b],
        //             k_range[h3b],k_range[p4b],
        //             k_range[p5b],k_range[p6b],
        //             a_c,&k_a_sort[0],&k_b_sort[0]);
     }

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
      && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
      {
        //  sd_t_s1_9_cuda(k_range[h1b],k_range[h2b],
        //             k_range[h3b],k_range[p4b],
        //             k_range[p5b],k_range[p6b],
        //             a_c,&k_a_sort[0],&k_b_sort[0]);
     }

    }

} //end ccsd_t_singles

#endif //CCSD_T_SINGLES_GPU_HPP_