#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"

namespace ctce {

  static Assignment a_t2_6_1, a_t2_6_2_1; 
  static Multiplication m_t2_6_2_2, m_t2_6_2, m_t2_6_3, m_t2_6, m_t2_8;

  extern "C" {

    void gen_expr_t2_68_cxx_() {

      static bool set_t2_68 = true;
      Tensor tC, tA, tB;

      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

      if (set_t2_68) {

        //std::cout << "set t2 6-8.\n";

        /* i1 ( h9 h11 h1 h2 )_v + = -1 * v ( h9 h11 h1 h2 )_v */
        tC = Tensor4(H9B,H11B,H1B,H2B,0,0,1,1,iV_tensor, dist_nw, dim_ov);
        tA = Tensor4(H9B,H11B,H1B,H2B,0,0,1,1,V_tensor, idist, dim_n);
        a_t2_6_1 = Assignment(tC,tA,-1.0);

        /* i2 ( h9 h11 h1 p8 )_v + = 1 * v ( h9 h11 h1 p8 )_v */
        tC = Tensor4(H9B,H11B,H1B,P8B,0,0,1,2,iV_tensor, dist_nw, dim_ov);
        tA = Tensor4(H9B,H11B,H1B,P8B,0,0,1,2,V_tensor, idist, dim_n);
        a_t2_6_2_1 = Assignment(tC,tA,1.0);

        /* i2 ( h9 h11 h1 p8 )_vt + = 1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h9 h11 p6 p8 )_v */
        tC = Tensor4(H9B,H11B,H1B,P8B,0,0,1,2,iV_tensor, dist_nw, dim_ov);
        tA = Tensor2(P6B,H1B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H9B,H11B,P6B,P8B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_6_2_2 = Multiplication(tC,tA,tB,0.5);

        /* i1 ( h9 h11 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( p8 ) * t ( p8 h1 )_t * i2 ( h9 h11 h2 p8 )_v */
        tC = Tensor4(H9B,H11B,H1B,H2B,0,0,1,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P8B,H1B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H9B,H11B,H2B,P8B,0,0,1,2,iV_tensor, dist_nw, dim_ov);
        m_t2_6_2 = Multiplication(tC,tA,tB,1.0);

        /* i1 ( h9 h11 h1 h2 )_vt + = -1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( h9 h11 p5 p6 )_v */
        tC = Tensor4(H9B,H11B,H1B,H2B,0,0,1,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor4(P5B,P6B,H1B,H2B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor4(H9B,H11B,P5B,P6B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_6_3 = Multiplication(tC,tA,tB,-0.5);

        /* i0 ( p3 p4 h1 h2 )_vt + = -1/2 * Sum ( h11 h9 ) * t ( p3 p4 h9 h11 )_t * i1 ( h9 h11 h1 h2 )_v */
        tC = Tensor4(P3B,P4B,H1B,H2B,0,0,1,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor4(P3B,P4B,H9B,H11B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor4(H9B,H11B,H1B,H2B,0,0,1,1,iV_tensor, dist_nw, dim_ov);
        m_t2_6 = Multiplication(tC,tA,tB,-0.5);

        /* i0 ( p3 p4 h1 h2 )_vt + = 1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( p3 p4 p5 p6 )_v */
        tC = Tensor4(P3B,P4B,H1B,H2B,0,0,1,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor4(P5B,P6B,H1B,H2B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor4(P3B,P4B,P5B,P6B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_8 = Multiplication(tC,tA,tB,0.5);

        set_t2_68 = false;
      }
    }

    void ccsd_t2_6_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_6_1);
    } // t2_6_1

    void ccsd_t2_6_2_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_6_2_1);
    } // t2_6_2_1

    void ccsd_t2_6_2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_6_2_2);
    } // t2_6_2_2

    void ccsd_t2_6_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_6_2);
    } // t2_6_2

    void ccsd_t2_6_3_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_6_3);
    } // t2_6_3

    void ccsd_t2_6_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_6);
    } // t2_6

    void ccsd_t2_8_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_8);
    } // t2_8

  } // extern C
}; // namespace ctce

