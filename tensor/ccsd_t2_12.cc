#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "expression.h"

namespace ctce {

  static Assignment a_t2_1, a_t2_2_1, a_t2_2_2_1, a_t2_2_2_2_1, a_t2_2_4_1, a_t2_2_5_1;
  static Multiplication m_t2_2, m_t2_2_2, m_t2_2_2_2, m_t2_2_2_2_2, m_t2_2_2_3, m_t2_2_4, m_t2_2_4_2, m_t2_2_5, m_t2_2_5_2, m_t2_2_6;
  static Tensor t2_2_1;

  extern "C" {

    void gen_expr_t2_12_cxx_() {

      static bool set_t2_12 = true;
      Tensor tC, tA, tB;

      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

      if (set_t2_12) {

        //std::cout << "set t2 1-2.\n";
	//std::cout<<"INTORB===="<<Variables::intorb()<<endl;

        // a_t2_1 : i0 ( p3 p4 h1 h2 )_v + = 1 * v ( p3 p4 h1 h2 )_v 
        tC = Tensor4(P3B,P4B,H1B,H2B,0,0,1,1,iV_tensor, dist_nw, dim_ov);
        tA = Tensor4(P3B,P4B,H1B,H2B,0,0,1,1,V_tensor, idist, dim_n);
        a_t2_1 = Assignment(tC,tA,1.0,ivec(P3B,P4B,H1B,H2B),ivec(P3B,P4B,H1B,H2B));

        // m_t2_2 : i0 ( p3 p4 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( h10 ) * t ( p3 h10 )_t * i1 ( h10 p4 h1 h2 )_v
        tC = Tensor4(P3B,P4B,H1B,H2B,0,0,1,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P3B,H10B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(P4B,H10B,H1B,H2B,0,1,2,2,iV_tensor, dist_nw, dim_ov); // ?
        m_t2_2 = Multiplication(tC,tA,tB,-1.0);

        // a_t2_2_1 : i1 ( h10 p3 h1 h2 )_v + = 1 * v ( h10 p3 h1 h2 )_v
        tC = Tensor4(P3B,H10B,H1B,H2B,0,1,2,2,iV_tensor, idist, dim_n);
        tA = Tensor4(H10B,P3B,H1B,H2B,0,1,2,2,V_tensor, idist, dim_n);
        a_t2_2_1 = Assignment(tC,tA,1.0, ivec(P3B,H10B,H1B,H2B), ivec(H10B,P3B,H1B,H2B));

        // m_t2_2_2 : i1 ( h10 p3 h1 h2 )_vt + = 1/2 * Sum ( h11 ) * t ( p3 h11 )_t * i2 ( h10 h11 h1 h2 )_v
        tC = Tensor4(P3B,H10B,H1B,H2B,0,1,2,2,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P3B,H11B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H10B,H11B,H1B,H2B,0,0,1,1,iV_tensor, dist_nw, dim_ov);
        m_t2_2_2 = Multiplication(tC,tA,tB,0.5);

        // a_t2_2_2_1 : i2 ( h10 h11 h1 h2 )_v + = -1 * v ( h10 h11 h1 h2 )_v
        tC = Tensor4(H10B,H11B,H1B,H2B,0,0,1,1,iV_tensor, idist, dim_n);
        tA = Tensor4(H10B,H11B,H1B,H2B,0,0,1,1,V_tensor, idist, dim_n);
        a_t2_2_2_1 = Assignment(tC,tA,-1.0, ivec(H10B,H11B,H1B,H2B), ivec(H10B,H11B,H1B,H2B));

        // m_t2_2_2_2 : i2 ( h10 h11 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i3 ( h10 h11 h2 p5 )_v
        tC = Tensor4(H10B,H11B,H1B,H2B,0,0,1,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P5B,H1B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H10B,H11B,H2B,P5B,0,0,1,2,iV_tensor, dist_nw, dim_ov);
        m_t2_2_2_2 = Multiplication(tC,tA,tB,1.0);

        // a_t2_2_2_2_1 : i3 ( h10 h11 h1 p5 )_v + = 1 * v ( h10 h11 h1 p5 )_v
        tC = Tensor4(H10B,H11B,H1B,P5B,0,0,1,2,iV_tensor, idist, dim_n);
        tA = Tensor4(H10B,H11B,H1B,P5B,0,0,1,2,V_tensor, idist, dim_n);
        a_t2_2_2_2_1 = Assignment(tC,tA,1.0, ivec(H10B,H11B,H1B,P5B), ivec(H10B,H11B,H1B,P5B));

        // m_t2_2_2_2_2 : i3 ( h10 h11 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h10 h11 p5 p6 )_v 
        tC = Tensor4(H10B,H11B,H1B,P5B,0,0,1,2,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P6B,H1B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H10B,H11B,P5B,P6B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_2_2_2_2 = Multiplication(tC,tA,tB,-0.5);

        // m_t2_2_2_3 : i2 ( h10 h11 h1 h2 )_vt + = -1/2 * Sum ( p7 p8 ) * t ( p7 p8 h1 h2 )_t * v ( h10 h11 p7 p8 )_v 
        tC = Tensor4(H10B,H11B,H1B,H2B,0,0,1,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor4(P7B,P8B,H1B,H2B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor4(H10B,H11B,P7B,P8B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_2_2_3 = Multiplication(tC,tA,tB,-0.5);

        // m_t2_2_4 : i1 ( h10 p3 h1 h2 )_ft + = -1 * Sum ( p5 ) * t ( p3 p5 h1 h2 )_t * i2 ( h10 p5 )_f
        tC = Tensor4(P3B,H10B,H1B,H2B,0,1,2,2,iTF_tensor, dist_nw, dim_ov);
        tA = Tensor4(P3B,P5B,H1B,H2B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor2(H10B,P5B,0,1,iF_tensor, dist_nw, dim_ov);
        m_t2_2_4 = Multiplication(tC,tA,tB,-1.0);

        // a_t2_2_4_1 : i2 ( h10 p5 )_f + = 1 * f ( h10 p5 )_f
        tC = Tensor2(H10B,P5B,0,1,iF_tensor, dist_nw, dim_ov);
        tA = Tensor2(H10B,P5B,0,1,F_tensor, dist_nw, dim_n);
        a_t2_2_4_1 = Assignment(tC,tA,1.0, ivec(H10B,P5B), ivec(H10B,P5B));

        // m_t2_2_4_2 : i2 ( h10 p5 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h10 p5 p6 )_v
        tC = Tensor2(H10B,P5B,0,1,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P6B,H7B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H7B,H10B,P5B,P6B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_2_4_2 = Multiplication(tC,tA,tB,-1.0);

        // m_t2_2_5 : i1 ( h10 p3 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( h7 p9 ) * t ( p3 p9 h1 h7 )_t * i2 ( h7 h10 h2 p9 )_v
        tC = Tensor4(P3B,H10B,H1B,H2B,0,1,2,2,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor4(P3B,P9B,H1B,H7B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor4(H7B,H10B,H2B,P9B,0,0,1,2,iV_tensor, dist_nw, dim_ov);
        m_t2_2_5 = Multiplication(tC,tA,tB,1.0);

        // a_t2_2_5_1 : i2 ( h7 h10 h1 p9 )_v + = 1 * v ( h7 h10 h1 p9 )_v
        tC = Tensor4(H7B,H10B,H1B,P9B,0,0,1,2,iV_tensor, idist, dim_n);
        tA = Tensor4(H7B,H10B,H1B,P9B,0,0,1,2,V_tensor, idist, dim_n);
        a_t2_2_5_1 = Assignment(tC,tA,1.0, ivec(H7B,H10B,H1B,P9B), ivec(H7B,H10B,H1B,P9B));

        // m_t2_2_5_2 : i2 ( h7 h10 h1 p9 )_vt + = 1 * Sum ( p5 ) * t ( p5 h1 )_t * v ( h7 h10 p5 p9 )_v
        tC = Tensor4(H7B,H10B,H1B,P9B,0,0,1,2,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor2(P5B,H1B,0,1,T_tensor, dist_nwma, dim_ov);
        tB = Tensor4(H7B,H10B,P5B,P9B,0,0,1,1,V_tensor, idist, dim_n);
        m_t2_2_5_2 = Multiplication(tC,tA,tB,1.0);

        // m_t2_2_6 : i1 ( h10 p3 h1 h2 )_vt + = 1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( h10 p3 p5 p6 )_v
        tC = Tensor4(P3B,H10B,H1B,H2B,0,1,2,2,iVT_tensor, dist_nw, dim_ov);
        tA = Tensor4(P5B,P6B,H1B,H2B,0,0,1,1,T_tensor, dist_nw, dim_ov);
        tB = Tensor4(H10B,P3B,P5B,P6B,0,1,2,2,V_tensor, idist, dim_n);
        m_t2_2_6 = Multiplication(tC,tA,tB,0.5);

	//OFFSET_ccsd_t2_2_1: i1 ( h10 p3 h1 h2 )_v
	t2_2_1 = Tensor4(P3B,H10B,H1B,H2B,0,1,2,2,iVT_tensor,dist_nw,dim_ov);

        set_t2_12 = false;
      }
    }

    void ccsd_t2_2_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1) {
      t2_2_1.create(k_i1_offset, d_i1);
    }

    void ccsd_t2_2_1_deletefile_cxx_() {
      t2_2_1.destroy();
    }

    void ccsd_t2_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_1);
    }
    void ccsd_t2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2);
    }
    void ccsd_t2_2_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_2_1);
    }
    void ccsd_t2_2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2_2);
    }
    void ccsd_t2_2_2_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_2_2_1);
    }
    void ccsd_t2_2_2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2_2_2);
    }
    void ccsd_t2_2_2_2_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_2_2_2_1);
    }
    void ccsd_t2_2_2_2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2_2_2_2);
    }
    void ccsd_t2_2_2_3_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2_2_3);
    }

    /* no use : t2_2_3, t2_2_3_1, t2_2_3_2 */

    void ccsd_t2_2_4_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2_4);
    } // t2_2_4
    void ccsd_t2_2_4_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_2_4_1);
    } // t2_2_4_1
    void ccsd_t2_2_4_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2_4_2);
    } // t2_2_4_2
    void ccsd_t2_2_5_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2_5);
    } // t2_2_5
    void ccsd_t2_2_5_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t2_2_5_1);
    } // t2_2_5_1
    void ccsd_t2_2_5_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2_5_2);
    } // t2_2_5_2
    void ccsd_t2_2_6_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t2_2_6);
    } // t2_2_6

    // no use : t2_3

  } // extern C
}; // namespace ctce

