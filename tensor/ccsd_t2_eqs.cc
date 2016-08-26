extern "C" {
#include "ctce_parser.h"
};

#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"
#include "corf.h"
#include "equations.h"

/*
 *i0 ( p3 p4 h1 h2 )_v + = 1 * v ( p3 p4 h1 h2 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( h10 ) * t ( p3 h10 )_t * i1 ( h10 p4 h1 h2 )_v
 *    i1 ( h10 p3 h1 h2 )_v + = 1 * v ( h10 p3 h1 h2 )_v
 *    i1 ( h10 p3 h1 h2 )_vt + = 1/2 * Sum ( h11 ) * t ( p3 h11 )_t * i2 ( h10 h11 h1 h2 )_v
 *        i2 ( h10 h11 h1 h2 )_v + = -1 * v ( h10 h11 h1 h2 )_v
 *        i2 ( h10 h11 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i3 ( h10 h11 h2 p5 )_v
 *            i3 ( h10 h11 h1 p5 )_v + = 1 * v ( h10 h11 h1 p5 )_v
 *            i3 ( h10 h11 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h10 h11 p5 p6 )_v
 *        i2 ( h10 h11 h1 h2 )_vt + = -1/2 * Sum ( p7 p8 ) * t ( p7 p8 h1 h2 )_t * v ( h10 h11 p7 p8 )_v
 *    i1 ( h10 p3 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i2 ( h10 p3 h2 p5 )_v
 *        i2 ( h10 p3 h1 p5 )_v + = 1 * v ( h10 p3 h1 p5 )_v
 *        i2 ( h10 p3 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h10 p3 p5 p6 )_v
 *    i1 ( h10 p3 h1 h2 )_ft + = -1 * Sum ( p5 ) * t ( p3 p5 h1 h2 )_t * i2 ( h10 p5 )_f
 *        i2 ( h10 p5 )_f + = 1 * f ( h10 p5 )_f
 *        i2 ( h10 p5 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h10 p5 p6 )_v
 *    i1 ( h10 p3 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( h7 p9 ) * t ( p3 p9 h1 h7 )_t * i2 ( h7 h10 h2 p9 )_v
 *        i2 ( h7 h10 h1 p9 )_v + = 1 * v ( h7 h10 h1 p9 )_v
 *        i2 ( h7 h10 h1 p9 )_vt + = 1 * Sum ( p5 ) * t ( p5 h1 )_t * v ( h7 h10 p5 p9 )_v
 *    i1 ( h10 p3 h1 h2 )_vt + = 1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( h10 p3 p5 p6 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i1 ( p3 p4 h2 p5 )_v
 *    i1 ( p3 p4 h1 p5 )_v + = 1 * v ( p3 p4 h1 p5 )_v
 *    i1 ( p3 p4 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( p3 p4 p5 p6 )_v
 *i0 ( p3 p4 h1 h2 )_tf + = -1 * P( 2 ) * Sum ( h9 ) * t ( p3 p4 h1 h9 )_t * i1 ( h9 h2 )_f
 *    i1 ( h9 h1 )_f + = 1 * f ( h9 h1 )_f
 *    i1 ( h9 h1 )_ft + = 1 * Sum ( p8 ) * t ( p8 h1 )_t * i2 ( h9 p8 )_f
 *        i2 ( h9 p8 )_f + = 1 * f ( h9 p8 )_f
 *        i2 ( h9 p8 )_vt + = 1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h9 p6 p8 )_v
 *    i1 ( h9 h1 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h9 h1 p6 )_v
 *    i1 ( h9 h1 )_vt + = -1/2 * Sum ( h8 p6 p7 ) * t ( p6 p7 h1 h8 )_t * v ( h8 h9 p6 p7 )_v
 *i0 ( p3 p4 h1 h2 )_tf + = 1 * P( 2 ) * Sum ( p5 ) * t ( p3 p5 h1 h2 )_t * i1 ( p4 p5 )_f
 *    i1 ( p3 p5 )_f + = 1 * f ( p3 p5 )_f
 *    i1 ( p3 p5 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 p3 p5 p6 )_v
 *    i1 ( p3 p5 )_vt + = -1/2 * Sum ( h7 h8 p6 ) * t ( p3 p6 h7 h8 )_t * v ( h7 h8 p5 p6 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = -1/2 * Sum ( h11 h9 ) * t ( p3 p4 h9 h11 )_t * i1 ( h9 h11 h1 h2 )_v
 *    i1 ( h9 h11 h1 h2 )_v + = -1 * v ( h9 h11 h1 h2 )_v
 *    i1 ( h9 h11 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( p8 ) * t ( p8 h1 )_t * i2 ( h9 h11 h2 p8 )_v
 *        i2 ( h9 h11 h1 p8 )_v + = 1 * v ( h9 h11 h1 p8 )_v
 *        i2 ( h9 h11 h1 p8 )_vt + = 1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h9 h11 p6 p8 )_v
 *    i1 ( h9 h11 h1 h2 )_vt + = -1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( h9 h11 p5 p6 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = -1 * P( 4 ) * Sum ( h6 p5 ) * t ( p3 p5 h1 h6 )_t * i1 ( h6 p4 h2 p5 )_v
 *    i1 ( h6 p3 h1 p5 )_v + = 1 * v ( h6 p3 h1 p5 )_v
 *    i1 ( h6 p3 h1 p5 )_vt + = -1 * Sum ( p7 ) * t ( p7 h1 )_t * v ( h6 p3 p5 p7 )_v
 *    i1 ( h6 p3 h1 p5 )_vt + = -1/2 * Sum ( h8 p7 ) * t ( p3 p7 h1 h8 )_t * v ( h6 h8 p5 p7 )_v
 *i0 ( p3 p4 h1 h2 )_vt + = 1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( p3 p4 p5 p6 )_v
 */

/*
 * t2_1:  i0 ( p3 p4 h1 h2 ) += 1 * v ( p3 p4 h1 h2 )
 * t2_2_1_createfile:     i1 ( h10 p3 h1 h2 )
 * t2_2_1:      t2_2_1 ( h10 p3 h1 h2 ) += 1 * v ( h10 p3 h1 h2 )
 * t2_2_2_1_createfile:     i2 ( h10 h11 h1 h2 )
 * t2_2_2_1:     t2_2_2_1 ( h10 h11 h1 h2 ) += -1 * v ( h10 h11 h1 h2 )
 * t2_2_2_2_1_createfile:     i3 ( h10 h11 h1 p5 )
 * t2_2_2_2_1:     t2_2_2_2_1 ( h10 h11 h1 p5 ) += 1 * v ( h10 h11 h1 p5 )
 * t2_2_2_2_2:     t2_2_2_2_1 ( h10 h11 h1 p5 ) += -0.5 * t ( p6 h1 ) * v ( h10 h11 p5 p6 )
 * t2_2_2_2:     t2_2_2_1 ( h10 h11 h1 h2 ) += 1 * t ( p5 h1 ) * t2_2_2_2_1 ( h10 h11 h2 p5 )
 * t2_2_2_2_1_deletefile
 * t2_2_2_3:     t2_2_2_1 ( h10 h11 h1 h2 ) += -0.5 * t ( p7 p8 h1 h2 ) * v ( h10 h11 p7 p8 )
 * t2_2_2:     t2_2_1 ( h10 p3 h1 h2 ) += 0.5 * t ( p3 h11 ) * t2_2_2_1 ( h10 h11 h1 h2 )
 * t2_2_2_1_deletefile
 * t2_2_4_1_createfile:     i2 ( h10 p5 )
 * t2_2_4_1:     t2_2_4_1 ( h10 p5 ) += 1 * f ( h10 p5 )
 * t2_2_4_2:     t2_2_4_1 ( h10 p5 ) += -1 * t ( p6 h7 ) * v ( h7 h10 p5 p6 )
 * t2_2_4:     t2_2_1 ( h10 p3 h1 h2 ) += -1 * t ( p3 p5 h1 h2 ) * t2_2_4_1 ( h10 p5 )
 * t2_2_4_1_deletefile
 * t2_2_5_1_createfile:     i2 ( h7 h10 h1 p9 )
 * t2_2_5_1:     t2_2_5_1 ( h7 h10 h1 p9 ) += 1 * v ( h7 h10 h1 p9 )
 * t2_2_5_2:     t2_2_5_1 ( h7 h10 h1 p9 ) += 1 * t ( p5 h1 ) * v ( h7 h10 p5 p9 )
 * t2_2_5:     t2_2_1 ( h10 p3 h1 h2 ) += 1 * t ( p3 p9 h1 h7 ) * t2_2_5_1 ( h7 h10 h2 p9 )
 * t2_2_5_1_deletefile
 * c2f_t2_t12: t2 ( p1 p2 h3 h4) += 0.5 * t ( p1 h3 ) * t (p2 h4)
 * t2_2_6:     t2_2_1 ( h10 p3 h1 h2 ) += 0.5 * t ( p5 p6 h1 h2 ) * v ( h10 p3 p5 p6 )
 * c2d_t2_t12: t2 ( p1 p2 h3 h4) += -0.5 * t ( p1 h3 ) * t (p2 h4)
 * t2_2:     i0 ( p3 p4 h1 h2 ) += -1 * t ( p3 h10 ) * t2_2_1 ( h10 p4 h1 h2 )
 * t2_2_1_deletefile
 * lt2_3x:     i0 ( p3 p4 h1 h2 ) += -1 * t ( p5 h1 ) * v ( p3 p4 h2 p5 )
 * OFFSET_ccsd_t2_4_1:     i1 ( h9 h1 )
 * t2_4_1:     t2_4_1 ( h9 h1 ) += 1 * f ( h9 h1 )
 * OFFSET_ccsd_t2_4_2_1: i2 ( h9 p8 )
 * t2_4_2_1:     t2_4_2_1 ( h9 p8 ) += 1 * f ( h9 p8 )
 * t2_4_2_2:     t2_4_2_1 ( h9 p8 ) += 1 * t ( p6 h7 ) * v ( h7 h9 p6 p8 )
 * t2_4_2:     t2_4_1 ( h9 h1 ) += 1 * t ( p8 h1 ) * t2_4_2_1 ( h9 p8 )
 * t2_4_3:     t2_4_1 ( h9 h1 ) += -1 * t ( p6 h7 ) * v ( h7 h9 h1 p6 )
 * t2_4_4:     t2_4_1 ( h9 h1 ) += -0.5 * t ( p6 p7 h1 h8 ) * v ( h8 h9 p6 p7 )
 * create i1_local
c    copy d_t1 ==> l_t1_local
 * t2_4:     i0 ( p3 p4 h1 h2 ) += -1 * t ( p3 p4 h1 h9 ) * t2_4_1 ( h9 h2 )
 * delete i1_local
 * DELETEFILE t2_4_1
 * t2_5_1 create:     i1 ( p3 p5 )
 * t2_5_1:     t2_5_1 ( p3 p5 ) += 1 * f ( p3 p5 )
 * t2_5_2:     t2_5_1 ( p3 p5 ) += -1 * t ( p6 h7 ) * v ( h7 p3 p5 p6 )
 * t2_5_3:     t2_5_1 ( p3 p5 ) += -0.5 * t ( p3 p6 h7 h8 ) * v ( h7 h8 p5 p6 )
 * create i1_local
c    copy d_t1 ==> l_t1_local
 * t2_5:     i0 ( p3 p4 h1 h2 ) += 1 * t ( p3 p5 h1 h2 ) * t2_5_1 ( p4 p5 )
 * delete i1_local
 * DELETEFILE t2_5_1
 * OFFSET_t2_6_1:     i1 ( h9 h11 h1 h2 )
 * t2_6_1:     t2_6_1 ( h9 h11 h1 h2 ) += -1 * v ( h9 h11 h1 h2 )
 * OFFSET_t2_6_2_1:     i2 ( h9 h11 h1 p8 )
 * t2_6_2_1:     t2_6_2_1 ( h9 h11 h1 p8 ) += 1 * v ( h9 h11 h1 p8 )
 * t2_6_2_2:     t2_6_2_1 ( h9 h11 h1 p8 ) += 0.5 * t ( p6 h1 ) * v ( h9 h11 p6 p8 )
 * t2_6_2:     t2_6_1 ( h9 h11 h1 h2 ) += 1 * t ( p8 h1 ) * t2_6_2_1 ( h9 h11 h2 p8 )
 * DELETEFILE t2_6_2_1
 * t2_6_3:     t2_6_1 ( h9 h11 h1 h2 ) += -0.5 * t ( p5 p6 h1 h2 ) * v ( h9 h11 p5 p6 )
 * t2_6:     i0 ( p3 p4 h1 h2 ) += -0.5 * t ( p3 p4 h9 h11 ) * t2_6_1 ( h9 h11 h1 h2 )
 * DELETEFILE t2_6_1
 * OFFSET_t2_7_1:     i1 ( h6 p3 h1 p5 )
 * t2_7_1:     t2_7_1 ( h6 p3 h1 p5 ) += 1 * v ( h6 p3 h1 p5 )
 * t2_7_2:     t2_7_1 ( h6 p3 h1 p5 ) += -1 * t ( p7 h1 ) * v ( h6 p3 p5 p7 )
 * t2_7_3:     t2_7_1 ( h6 p3 h1 p5 ) += -0.5 * t ( p3 p7 h1 h8 ) * v ( h6 h8 p5 p7 )
 * t2_7:     i0 ( p3 p4 h1 h2 ) += -1 * t ( p3 p5 h1 h6 ) * t2_7_1 ( h6 p4 h2 p5 )
 * DELETEFILE t2_7_1
 * vt1t1_1_createfile: C     i1 ( h5 p3 h1 h2 )
 * vt1t1_1_2:     vt1t1_1 ( h5 p3 h1 h2 ) += -2 * t ( p6 h1 ) * v ( h5 p3 h2 p6 ) 
 * vt1t1_1:     i0 ( p3 p4 h1 h2 )t += -0.5 * t ( p3 h5 ) * vt1t1_1 ( h5 p4 h1 h2 )
 * vt1t1_1_deletefile
 * c2f_t2_t12: t2 ( p1 p2 h3 h4) += 0.5 * t ( p1 h3 ) * t (p2 h4)
 * t2_8:   i0 ( p3 p4 h1 h2 )_vt + = 0.5 * t ( p5 p6 h1 h2 ) * v ( p3 p4 p5 p6 )
 * c2d_t2_t12: t2 ( p1 p2 h3 h4) += -0.5 * t ( p1 h3 ) * t (p2 h4)
 */

namespace ctce {

  void ccsd_t2_equations(ctce::Equations &eqs) {
    ::Equations peqs;
    ctce_parser(CTCE_EQ_PATH"/ccsd_t2_hand.eq", &peqs);
    parser_eqs_to_ctce_eqs(&peqs, eqs);
  }

}; /*ctce*/

