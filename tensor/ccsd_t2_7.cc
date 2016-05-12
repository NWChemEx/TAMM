#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"

namespace ctce {

  extern "C" {
    // something is wrong with t2_7_1 :(
    /* i1 ( h6 p3 h1 p5 )_v + = 1 * v ( h6 p3 h1 p5 )_v */
    void ccsd_t2_7_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      //std::cout << "ccsd_t2_7_1!" << std::endl;
      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
      Tensor tC = Tensor4(P3B,H1B,P5B,H6B,0,1,2,3,iV_tensor, dist_nw, dim_ov);
      Tensor tA = Tensor4(H6B,P3B,H1B,P5B,0,1,2,3,V_tensor, idist, dim_n);
      Assignment a = Assignment(tC,tA,1.0, ivec(P3B,H1B,P5B,H6B), ivec(H6B,P3B,H1B,P5B));
      // t_assign4(d_a, k_a_offset, d_c, k_c_offset, a);
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a);
    } // t2_7_1

    // what is peta?
    /* i1 ( h6 p3 h1 p5 )_vt + = -1 * Sum ( p7 ) * t ( p7 h1 )_t * v ( h6 p3 p5 p7 )_v */
    void ccsd_t2_7_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      std::cout << "ccsd_t2_7_2!" << std::endl;
      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;
      Tensor tC = Tensor4(P3B,H1B,P5B,H6B,0,1,2,3,iVT_tensor, dist_nw, dim_ov);
      Tensor tA = Tensor2(P7B,H1B,0,1,T_tensor, dist_nwma, dim_ov);
      Tensor tB = Tensor4(H6B,P3B,P5B,P7B,0,1,2,2,V_tensor, idist, dim_n);
      Multiplication m = Multiplication(tC,tA,tB,-1.0);
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset,m);
    } // t2_7_2

  } // extern C
}; // namespace ctce

