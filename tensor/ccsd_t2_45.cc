#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "expression.h"

namespace ctce {

  static Assignment a_t2_4_1, a_t2_4_2_1, a_t2_5_1;
  static Multiplication m_t2_4_2_2, m_t2_4_2, m_t2_4_3, m_t2_4_4, m_t2_4, m_t2_5_2, m_t2_5_3, m_t2_5;

  extern "C" {

    void gen_expr_t2_45_cxx_() {

      static bool set_t2_45 = true;
      Tensor tC, tA, tB;

      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

      if (set_t2_45) {

        //std::cout << "set t2 4-5.\n";

        /* i1 ( h9 h1 )_f + = 1 * f ( h9 h1 )_f */
        tC = Tensor2(H9B,H1B,0,1,iF_tensor, dist_nw, dim_ov);
        tA = Tensor2(H9B,H1B,0,1,F_tensor, dist_nw, dim_n);
        a_t2_4_1 = Assignment(tC,tA,1.0);

        /* i2 ( h9 p8 )_f + = 1 * f ( h9 p8 )_f */
        tC = Tensor2(H9B,P8B,0,1,iF_tensor, dist_nw, dim_ov);
        tA = Tensor2(H9B,P8B,0,1,F_tensor, dist_nw, dim_n);
        a_t2_4_2_1 = Assignment(tC,tA,1.0);

        /* i2 ( h9 p8 )_vt + = 1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h9 p6 p8 )_v */
        tC = Tensor2(H9B,P8B,0,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P6B,H7B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H7B,H9B,P6B,P8B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_4_2_2 = Multiplication(tC,tA,tB, 1.0);

        /* i1 ( h9 h1 )_ft + = 1 * Sum ( p8 ) * t ( p8 h1 )_t * i2 ( h9 p8 )_f */
        tC = Tensor2(H9B,H1B,0,1,iTF_tensor, dist_nw, dim_ov);
        tA = Tensor2(P8B,H1B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor2(H9B,P8B,0,1,iF_tensor, dist_nw, dim_ov);
        m_t2_4_2 = Multiplication(tC,tA,tB, 1.0);

        /* i1 ( h9 h1 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h9 h1 p6 )_v */
        tC = Tensor2(H9B,H1B,0,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P6B,H7B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H7B,H9B,H1B,P6B,0,0,1,2,V_tensor, idist, dim_n);
        m_t2_4_3 = Multiplication(tC,tA,tB, -1.0);

        /* i1 ( h9 h1 )_vt + = -1/2 * Sum ( h8 p6 p7 ) * t ( p6 p7 h1 h8 )_t * v ( h8 h9 p6 p7 )_v */
        tC = Tensor2(H9B,H1B,0,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor4(P6B,P7B,H1B,H8B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor4(H8B,H9B,P6B,P7B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_4_4 = Multiplication(tC,tA,tB,-0.5);

        /* i0 ( p3 p4 h1 h2 )_tf + = -1 * P( 2 ) * Sum ( h9 ) * t ( p3 p4 h1 h9 )_t * i1 ( h9 h2 )_f */
        tC = Tensor4(P3B,P4B,H1B,H2B,0,0,1,1,iTF_tensor, dist_nw, dim_ov);
        tA = Tensor4(P3B,P4B,H1B,H9B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor2(H9B,H2B,0,1,iF_tensor, dist_nwma, dim_ov);
        //tB.get_ma = true;
        m_t2_4 = Multiplication(tC,tA,tB,-1.0);

        /* i1 ( p3 p5 )_f + = 1 * f ( p3 p5 )_f */
        tC = Tensor2(P3B,P5B,0,1,iF_tensor, dist_nw, dim_ov);
        tA = Tensor2(P3B,P5B,0,1,F_tensor, dist_nw, dim_n);
        a_t2_5_1 = Assignment(tC,tA,1.0);

        /* i1 ( p3 p5 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 p3 p5 p6 )_v */
        tC = Tensor2(P3B,P5B,0,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P6B,H7B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H7B,P3B,P5B,P6B,0,1,2,2,V_tensor, idist, dim_n);
        m_t2_5_2 = Multiplication(tC,tA,tB,-1.0);

        /* i1 ( p3 p5 )_vt + = -1/2 * Sum ( h7 h8 p6 ) * t ( p3 p6 h7 h8 )_t * v ( h7 h8 p5 p6 )_v */
        tC = Tensor2(P3B,P5B,0,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor4(P3B,P6B,H7B,H8B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor4(H7B,H8B,P5B,P6B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_5_3 = Multiplication(tC,tA,tB,-0.5);

        /* i0 ( p3 p4 h1 h2 )_tf + = 1 * P( 2 ) * Sum ( p5 ) * t ( p3 p5 h1 h2 )_t * i1 ( p4 p5 )_f */
        tC = Tensor4(P3B,P4B,H1B,H2B,0,0,1,1,iTF_tensor, dist_nw, dim_ov);
        tA = Tensor4(P3B,P5B,H1B,H2B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor2(P4B,P5B,0,1,iF_tensor, dist_nwma, dim_ov);
        //tB.get_ma = true;
        m_t2_5 = Multiplication(tC,tA,tB,1.0);

        set_t2_45 = false;
      }
    }

    void ccsd_t2_4_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_4_1);
    } // t2_4_1

    void ccsd_t2_4_2_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_4_2_1);
    } // t2_4_2_1

    void ccsd_t2_4_2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_4_2_2);
    } // t2_4_2_2

    void ccsd_t2_4_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_4_2);
    } // t2_4_2

    void ccsd_t2_4_3_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_4_3);
    } // t2_4_3

    void ccsd_t2_4_4_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_4_4);
    } // t2_4_4

    void ccsd_t2_4_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_4);
    } // t2_4

    void ccsd_t2_5_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_5_1);
    } // t2_5_1

    void ccsd_t2_5_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_5_2);
    } // t2_5_2

    void ccsd_t2_5_3_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_5_3);
    } // t2_5_3

    void ccsd_t2_5_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_5);
    } // t2_5

  } // extern C
}; // namespace ctce

