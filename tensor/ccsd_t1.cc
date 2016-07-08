#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"

namespace ctce {

  static Assignment a_t1_1, a_t1_2_1, a_t1_2_2_1, a_t1_3_1, a_t1_5_1, a_t1_6_1;
  static Multiplication m_t1_2_2_2, m_t1_2_2, m_t1_2_3, m_t1_2_4, m_t1_2, m_t1_3_2, m_t1_3, m_t1_4, m_t1_5_2, m_t1_5, m_t1_6_2, m_t1_6, m_t1_7;
  static Tensor t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1;

  extern "C" {

    void gen_expr_t1_cxx_() {

      static bool set_t1 = true;
      Tensor tC, tA, tB;

      DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;

      if (set_t1) {

	Tensor i0 = Tensor2(TV,TO,dist_nw);
	Tensor f = Tensor2(TN,TN, dist_nw);
	Tensor v = Tensor4(TN,TN,TN,TN,idist);
	Tensor t1 = Tensor2(TV,TO,dist_nwma);
	Tensor t2 = Tensor4(TV,TV,TO,TO,dist_nw);

	Tensor i1_2 = Tensor2(TO,TO,dist_nw);
	Tensor i2_2_2 = Tensor2(TO,TV,dist_nw);
	Tensor i1_3 = Tensor2(TV,TV,dist_nw);
	Tensor i1_5 = Tensor2(TO,TV,dist_nw);
	Tensor i1_6 = Tensor4(TO,TO,TO,TV,dist_nw);

	/*
     i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f                                                         DONE
     i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f                         DONE
         i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f                                                     DONE
         i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f                      DONE
             i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f                                                 DONE
             i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7 p3 p5 )_v         DONE
         i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1 p4 )_v             NOPE
         i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v  NOPE
     i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f                          DONE
         i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f                                                     DONE
         i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3 p4 )_v             NOPE
     i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1 p3 )_v                 NOPE
     i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8 p7 )_f                 DONE
         i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f                                                     DONE
         i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5 p7 )_v              NOPE
     i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4 h5 h1 p3 )_v     NOPE
         i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v                                         DONE
         i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5 p3 p6 )_v          NOPE
     i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 p2 p3 p4 )_v      DONE
	*/

        /* i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f */
        a_t1_1 = Assignment(i0,f,1.0,ivec(P2B,H1B), ivec(P2B,H1B));

        /* i1 ( h7 h1 )_f + = 1 * f ( h7 h1 )_f */
        a_t1_2_1 = Assignment(i1_2,f,1.0, ivec(H7B,H1B), ivec(H7B,H1B));

        /* i2 ( h7 p3 )_f + = 1 * f ( h7 p3 )_f */
        a_t1_2_2_1 = Assignment(i2_2_2,f,1.0, ivec(H7B,P3B), ivec(H7B,P3B));

        /* i2 ( h7 p3 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h7 p3 p5 )_v */
        m_t1_2_2_2 = Multiplication(i2_2_2,ivec(H7B,P3B),
				    t1,ivec(P5B,H6B),
				    v,ivec(H6B,H7B,P3B,P5B),
				    -1.0);

        /* i1 ( h7 h1 )_ft + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i2 ( h7 p3 )_f */
        m_t1_2_2 = Multiplication(i1_2,ivec(H7B,H1B),
				  t1,ivec(P3B,H1B),
				  i2_2_2,ivec(H7B,P3B),
				  1.0);

        /* i1 ( h7 h1 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h7 h1 p4 )_v */
        m_t1_2_3 = Multiplication(i1_2,ivec(H7B,H1B),
				  t1,ivec(P4B,H5B),
				  v,ivec(H5B,H7B,H1B,P4B),-1.0);

        /* i1 ( h7 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h7 p3 p4 )_v */
        m_t1_2_4 = Multiplication(i1_2,ivec(H7B,H1B),
				  t2,ivec(P3B,P4B,H1B,H5B),
				  v,ivec(H5B,H7B,P3B,P4B),-0.5);

        /* i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f */
        m_t1_2 = Multiplication(i0,ivec(P2B,H1B),
				t1,ivec(P2B,H7B),
				i1_2,ivec(H7B,H1B),-1.0);

        /* i1 ( p2 p3 )_f + = 1 * f ( p2 p3 )_f */
        a_t1_3_1 = Assignment(i1_3,f,1.0,ivec(P2B,P3B),ivec(P2B,P3B));

        /* i1 ( p2 p3 )_vt + = -1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 p2 p3 p4 )_v */
        m_t1_3_2 = Multiplication(i1_3,ivec(P2B,P3B),
				  t1,ivec(P4B,H5B),
				  v,ivec(H5B,P2B,P3B,P4B),-1.0);

        /* i0 ( p2 h1 )_tf + = 1 * Sum ( p3 ) * t ( p3 h1 )_t * i1 ( p2 p3 )_f */
        m_t1_3 = Multiplication(i0,ivec(P2B,H1B),t1,ivec(P3B,H1B),i1_3,ivec(P2B,P3B),1.0);

        /* i0 ( p2 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 h1 p3 )_v */
        m_t1_4 = Multiplication(i0,ivec(P2B,H1B),t1,ivec(P3B,H4B),v,ivec(H4B,P2B,H1B,P3B),-1.0);

        /* i1 ( h8 p7 )_f + = 1 * f ( h8 p7 )_f */
        a_t1_5_1 = Assignment(i1_5,f,1.0,ivec(H8B,P7B), ivec(H8B,P7B));

        /* i1 ( h8 p7 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 p5 p7 )_v */
        m_t1_5_2 = Multiplication(i1_5,ivec(H8B,P7B),t1,ivec(P5B,H6B),v,ivec(H6B,H8B,P5B,P7B),1.0);

        /* i0 ( p2 h1 )_tf + = 1 * Sum ( p7 h8 ) * t ( p2 p7 h1 h8 )_t * i1 ( h8 p7 )_f */
        m_t1_5 = Multiplication(i0,ivec(P2B,H1B),t2,ivec(P2B,P7B,H1B,H8B),i1_5,ivec(H8B,P7B),1.0);

        /* i1 ( h4 h5 h1 p3 )_v + = 1 * v ( h4 h5 h1 p3 )_v */
        a_t1_6_1 = Assignment(i1_6,v,1.0, ivec(H4B,H5B,H1B,P3B), ivec(H4B,H5B,H1B,P3B));

        /* i1 ( h4 h5 h1 p3 )_vt + = -1 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h4 h5 p3 p6 )_v */
#if 0
	Tensor tc = Tensor4(H4B,H5B,H1B,P3B,0,0,1,2,iV_tensor, dist_nw, dim_ov);
	Tensor ta = Tensor2(P6B,H1B, 0, 1, iV_tensor, dist_nwma, dim_ov);
	Tensor tb = Tensor4(H4B,H5B,P3B,P6B,0,0,1,1,iV_tensor, idist, dim_n);
        m_t1_6_2 = Multiplication(tc,ta,tb,-1.0);
#else
        m_t1_6_2 = Multiplication(i1_6,ivec(H4B,H5B,H1B,P3B),
				  t1,ivec(P6B,H1B),v,ivec(H4B,H5B,P3B,P6B),-1.0);
#endif

        /* i0 ( p2 h1 )_vt + = -1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4 h5 h1 p3 )_v */
        m_t1_6 = Multiplication(i0,ivec(P2B,H1B),t2,ivec(P2B,P3B,H4B,H5B),i1_6,ivec(H4B,H5B,H1B,P3B),-0.5);

        /* i0 ( p2 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 p2 p3 p4 )_v */
#if 0
        m_t1_7 = Multiplication(i0,ivec(P2B,H1B),t2,ivec(P3B,P4B,H1B,H5B),v,ivec(H5B,P2B,P3B,P4B),-0.5);
#else
	Tensor tc = Tensor2(P2B,H1B, 0, 1, iVT_tensor, dist_nw, dim_ov);
	Tensor ta = Tensor4(P3B,P4B,H4B,H5B,0,0,1,1,iT_tensor, dist_nwma, dim_ov);
	Tensor tb = Tensor4(H5B,P2B,P3B,P4B,0,1,1,1,iV_tensor, idist, dim_n);
        m_t1_7 = Multiplication(tc,ta,tb,-0.50);
#endif

        set_t1 = false;
        
        //OFFSET_ccsd_t1_2_1: i1 ( h7 h1 )_f
        t1_2_1 = Tensor2(H7B,H1B,0,1,iF_tensor,dist_nw,dim_ov);

        //OFFSET_ccsd_t1_2_2_1: i2 ( h7 p3 )_f
        t1_2_2_1 = Tensor2(H7B,P3B,0,1,iF_tensor,dist_nw,dim_ov);

        //OFFSET_ccsd_t1_3_1: i1 ( p2 p3 )_f
        t1_3_1 = Tensor2(P2B,P3B,0,1,iF_tensor,dist_nw,dim_ov);

        //OFFSET_ccsd_t1_5_1: i1 ( h8 p7 )_f
        t1_5_1 = Tensor2(H8B,P7B,0,1,iF_tensor,dist_nw,dim_ov);

        //OFFSET_ccsd_t1_6_1: i1 ( h4 h5 h1 p3 )_v
        t1_6_1 = Tensor4(H4B,H5B,H1B,P3B,0,0,1,2,iV_tensor, dist_nw, dim_ov);

      }
    }

    void ccsd_t1_2_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      t1_2_1.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_2_1_deletefile_cxx_() {
      t1_2_1.destroy();
    }

    void ccsd_t1_2_2_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      t1_2_2_1.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_2_2_1_deletefile_cxx_() {
      t1_2_2_1.destroy();
    }

    void ccsd_t1_3_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      t1_3_1.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_3_1_deletefile_cxx_() {
      t1_3_1.destroy();
    }

    void ccsd_t1_5_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      t1_5_1.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_5_1_deletefile_cxx_() {
      t1_5_1.destroy();
    }

    void ccsd_t1_6_1_createfile_cxx_(Integer *k_i1_offset, Integer *d_i1, Integer *size_i1) {
      t1_6_1.create(k_i1_offset, d_i1, size_i1);
    }

    void ccsd_t1_6_1_deletefile_cxx_() {
      t1_6_1.destroy();
    }

    void ccsd_t1_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_1);
    } // t1_1

    void ccsd_t1_2_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_2_1);
    } // t1_2_1

    void ccsd_t1_2_2_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_2_2_1);
    } // t1_2_2_1

    void ccsd_t1_2_2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2_2_2);
    } // t1_2_2_2

    void ccsd_t1_2_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2_2);
    } // t1_2_2

    void ccsd_t1_2_3_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2_3);
    } // t1_2_3

    void ccsd_t1_2_4_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2_4);
    } // t1_2_4

    void ccsd_t1_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_2);
    } // t1_2

    void ccsd_t1_3_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_3_1);
    } // t1_3_1

    void ccsd_t1_3_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_3_2);
    } // t1_3_2

    void ccsd_t1_3_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_3);
    } // t1_3

    void ccsd_t1_4_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_4);
    } // t1_4

    void ccsd_t1_5_1_cxx_(Integer *d_a, Integer *k_a_offset, Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_5_1);
    } // t1_5_1

    void ccsd_t1_5_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_5_2);
    } // t1_5_2

    void ccsd_t1_5_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_5);
    } // t1_5

    void ccsd_t1_6_1_cxx_(Integer *d_a, Integer *k_a_offset, 
        Integer *d_c, Integer *k_c_offset) {
      t_assign3(d_a, k_a_offset, d_c, k_c_offset, a_t1_6_1);
    } // t1_6_1

    void ccsd_t1_6_2_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_6_2);
    } // t1_6_2

    void ccsd_t1_6_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_6);
    } // t1_6

    void ccsd_t1_7_cxx_(Integer *d_a, Integer *k_a_offset,
        Integer *d_b, Integer *k_b_offset, Integer *d_c, Integer *k_c_offset) {
      t_mult4(d_a, k_a_offset, d_b, k_b_offset, d_c, k_c_offset, m_t1_7);
    } // t1_7

  } // extern C

}; // namespace ctce

