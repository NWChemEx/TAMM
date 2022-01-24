#pragma once

#include "diis.hpp"
#include "ccsd_util.hpp"
#include "ga/macdecls.h"
#include "ga/ga-mpi.h"

using namespace tamm;

bool debug = false;

TiledIndexSpace o_alpha,v_alpha,o_beta,v_beta;
Tensor<double> f1_aa_oo, f1_aa_ov, f1_aa_vo, f1_aa_vv;
Tensor<double> f1_bb_oo, f1_bb_ov, f1_bb_vo, f1_bb_vv;
Tensor<double> t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb; //t2_baba
Tensor<double> chol3d_aa_oo, chol3d_aa_ov, chol3d_aa_vo, chol3d_aa_vv;
Tensor<double> chol3d_bb_oo, chol3d_bb_ov, chol3d_bb_vo, chol3d_bb_vv;
Tensor<double> r1_aa, r1_bb, r2_aaaa, r2_bbbb; //r2_abab, r2_baba, r2_abba, r2_baab;
Tensor<double> _a004_aaaa, _a004_abab, _a004_bbbb;

template<typename T>
void ccsd_e(/* ExecutionContext &ec, */
            Scheduler& sch,
            const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de, const Tensor<T>& t1,
            const Tensor<T>& t2, const Tensor<T>& f1, Tensor<T>& chol3d) { 

    // const TiledIndexSpace& O = MO("occ");
    // const TiledIndexSpace& V = MO("virt");

    Tensor<T> _a01{CI};
    // Tensor<T> _a02{{O,O,CI},{1,1}};
    // Tensor<T> _a03{{O,V,CI},{1,1}};

    auto [cind] = CI.labels<1>("all");
    // auto [p1, p2, p3, p4, p5] = MO.labels<5>("virt");
    // auto [h3, h4, h5, h6]     = MO.labels<4>("occ");

    auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
    auto [p1_vb, p2_vb, p3_vb] = v_beta.labels<3>("all");
    auto [h3_oa, h4_oa, h6_oa] = o_alpha.labels<3>("all");
    auto [h3_ob, h4_ob, h6_ob] = o_beta.labels<3>("all");

    // Scheduler sch{ec};
    // sch.allocate(_a01,_a02,_a03);

    // sch 
    //     (_a01(cind) = t1(p3, h4) * chol3d(h4, p3,cind))
    //     (_a02(h4, h6, cind) = t1(p3, h4) * chol3d(h6, p3, cind))
    //     (de() =  0.5 * _a01() * _a01())
    //     (de() += -0.5 * _a02(h4, h6, cind) * _a02(h6, h4, cind))
    //     (_a03(h4, p2, cind) = t2(p1, p2, h3, h4) * chol3d(h3, p1, cind))
    //     (de() += 0.5 * _a03(h4, p1, cind) * chol3d(h4, p1, cind));
    //     ;
    // sch.deallocate(_a01,_a02,_a03);


    // Tensor<T> _a01_a{CI},_a01_b{CI};
    Tensor<T> _a02_aa{{o_alpha,o_alpha,CI},{1,1}}, _a02_bb{{o_beta,o_beta,CI},{1,1}};
    Tensor<T> _a03_aa{{o_alpha,v_alpha,CI},{1,1}}, _a03_bb{{o_beta,v_beta,CI},{1,1}};

    sch.allocate(_a01,_a02_aa,_a02_bb,_a03_aa,_a03_bb);

    // sch (_a01()=0)
    //     (_a02_aa()=0)
    //     (_a02_bb()=0)
    //     (_a03_aa()=0)
    //     (_a03_bb()=0);

    sch 
    (_a01(cind) = t1_aa(p3_va, h4_oa) * chol3d_aa_ov(h4_oa, p3_va, cind))
    (_a02_aa(h4_oa, h6_oa, cind) = t1_aa(p3_va, h4_oa) * chol3d_aa_ov(h6_oa, p3_va, cind))
    (_a03_aa(h4_oa, p2_va, cind) = t2_aaaa(p1_va, p2_va, h3_oa, h4_oa) * chol3d_aa_ov(h3_oa, p1_va, cind))
    (_a03_aa(h4_oa, p2_va, cind) += t2_abab(p2_va, p1_vb, h4_oa, h3_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind))
    (_a01(cind) += t1_bb(p3_vb, h4_ob) * chol3d_bb_ov(h4_ob, p3_vb, cind))
    (_a02_bb(h4_ob, h6_ob, cind) = t1_bb(p3_vb, h4_ob) * chol3d_bb_ov(h6_ob, p3_vb, cind))
    (_a03_bb(h4_ob, p2_vb, cind) = t2_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind))
    (_a03_bb(h4_ob, p2_vb, cind) += t2_abab(p1_va, p2_vb, h3_oa, h4_ob) * chol3d_aa_ov(h3_oa, p1_va, cind))
    (de() =  0.5 * _a01() * _a01())
    //(de() +=  0.5 * _a01_b() * _a01_b ())
    (de() += -0.5 * _a02_aa(h4_oa, h6_oa, cind) * _a02_aa(h6_oa, h4_oa, cind))
    (de() += -0.5 * _a02_bb(h4_ob, h6_ob, cind) * _a02_bb(h6_ob, h4_ob, cind))
    (de() += 0.5 * _a03_aa(h4_oa, p1_va, cind) * chol3d_aa_ov(h4_oa, p1_va, cind))
    (de() += 0.5 * _a03_bb(h4_ob, p1_vb, cind) * chol3d_bb_ov(h4_ob, p1_vb, cind));

    sch.deallocate(_a01,_a02_aa,_a02_bb,_a03_aa,_a03_bb);
}

template<typename T>
void ccsd_t1(/* ExecutionContext& ec,  */
             Scheduler& sch,
             const TiledIndexSpace& MO,const TiledIndexSpace& CI, 
             Tensor<T>& i0, const Tensor<T>& t1, const Tensor<T>& t2, 
             const Tensor<T>& f1, Tensor<T>& chol3d) {
    // const TiledIndexSpace& O = MO("occ");
    // const TiledIndexSpace& V = MO("virt");
    
    // Tensor<T> _a01{{O,O,CI},{1,1}};
    Tensor<T> _a02{CI};
    // Tensor<T> _a03{{V,O,CI},{1,1}};
    // Tensor<T> _a04{{O,O},{1,1}};
    // Tensor<T> _a05{{O,V},{1,1}};
    // Tensor<T> _a06{{O,O,CI},{1,1}};
    
    auto [cind] = CI.labels<1>("all");
    auto [p2] = MO.labels<1>("virt");
    auto [h1] = MO.labels<1>("occ");

    auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
    auto [p1_vb, p2_vb, p3_vb] = v_beta.labels<3>("all");
    auto [h1_oa, h2_oa, h3_oa, h7_oa] = o_alpha.labels<4>("all");
    auto [h1_ob, h2_ob, h3_ob, h7_ob] = o_beta.labels<4>("all");

    // sch
    //     .allocate(_a01, _a02, _a03, _a04, _a05, _a06)
    //     (_a01() = 0)
    //     (_a02() = 0)
    //     (_a03() = 0)
    //     (_a04() = 0)
    //     (_a05() = 0)
    //     (_a06() = 0)
    //     (i0(p2, h1) = f1(p2, h1))
    //     (_a01(h2, h1, cind) +=  1.0 * t1(p1, h1) * chol3d(h2, p1, cind))         // ovm
    //     (_a02(cind)         +=  1.0 * t1(p3, h3) * chol3d(h3, p3, cind))         // ovm
    //     (_a03(p1, h1, cind) +=  1.0 * t2(p1, p3, h2, h1) * chol3d(h2, p3, cind)) // o2v2m
    //     (_a04(h2, h1)       +=  1.0 * chol3d(h2, p1, cind) * _a03(p1, h1, cind)) // o2vm
    //     (i0(p2, h1)         +=  1.0 * t1(p2, h2) * _a04(h2, h1))                 // o2v
    //     (i0(p1, h2)         +=  1.0 * chol3d(p1, h2, cind) * _a02(cind))         // ovm
    //     (_a05(h2, p1)       += -1.0 * chol3d(h3, p1, cind) * _a01(h2, h3, cind)) // o2vm
    //     (i0(p2, h1)         +=  1.0 * t2(p1, p2, h2, h1) * _a05(h2, p1))         // o2v
    //     (i0(p2, h1)         += -1.0 * chol3d(p2, p1, cind) * _a03(p1, h1, cind)) // ov2m
    //     (_a03(p2, h2, cind) += -1.0 * t1(p1, h2) * chol3d(p2, p1, cind))         // ov2m
    //     (i0(p1, h2)         += -1.0 * _a03(p1, h2, cind) * _a02(cind))           // ovm
    //     (_a03(p2, h3, cind) += -1.0 * t1(p2, h3) * _a02(cind))                   // ovm
    //     (_a03(p2, h3, cind) +=  1.0 * t1(p2, h2) * _a01(h2, h3, cind))           // o2vm
    //     (_a01(h3, h1, cind) +=  1.0 * chol3d(h3, h1, cind))                      // o2m
    //     (i0(p2, h1)         +=  1.0 * _a01(h3, h1, cind) * _a03(p2, h3, cind))   // o2vm
    //     (i0(p2, h1)         += -1.0 * t1(p2, h7) * f1(h7, h1))                 // o2v
    //     (i0(p2, h1)         +=  1.0 * t1(p3, h1) * f1(p2, p3))                   // ov2
    //     .deallocate(_a01, _a02, _a03, _a04, _a05, _a06);

    // Tensor<T> _a02_aa{CI},_a02_bb{CI}; 
    Tensor<T> _a01_aa{{o_alpha,o_alpha,CI},{1,1}}, _a01_bb{{o_beta,o_beta,CI},{1,1}};
    Tensor<T> _a03_aa{{v_alpha,o_alpha,CI},{1,1}}, _a03_bb{{v_beta,o_beta,CI},{1,1}};

    Tensor<T> _a04_aa{{o_alpha,o_alpha},{1,1}}, _a04_bb{{o_beta,o_beta},{1,1}};
    Tensor<T> _a05_aa{{o_alpha,v_alpha},{1,1}}, _a05_bb{{o_beta,v_beta},{1,1}};


    Tensor<T> i0_aa = r1_aa;
    Tensor<T> i0_bb = r1_bb;

    sch.allocate(_a02, _a01_aa,_a01_bb,_a03_aa,_a03_bb,_a04_aa,_a04_bb,_a05_aa,_a05_bb);
    // sch (_a01_aa() = 0)
    //     (_a01_bb() = 0)
    //     (_a02() = 0)
    //     (_a03_aa() = 0)
    //     (_a03_bb() = 0)
    //     (_a04_aa() = 0)
    //     (_a04_bb() = 0)
    //     (_a05_aa() = 0)
    //     (_a05_bb() = 0);
    
     sch
        (i0(p2, h1) = 0)
        (i0_aa(p2_va, h1_oa) = f1_aa_vo(p2_va, h1_oa))
        (i0_bb(p2_vb, h1_ob) = f1_bb_vo(p2_vb, h1_ob))
        (_a01_aa(h2_oa, h1_oa, cind) =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_aa_ov(h2_oa, p1_va, cind))  // ovm
        (_a01_bb(h2_ob, h1_ob, cind) =  1.0 * t1_bb(p1_vb, h1_ob) * chol3d_bb_ov(h2_ob, p1_vb, cind))         // ovm
        (_a02(cind)         =  1.0 * t1_aa(p3_va, h3_oa) * chol3d_aa_ov(h3_oa, p3_va, cind))         // ovm
        (_a02(cind)         +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_bb_ov(h3_ob, p3_vb, cind))         // ovm

        (_a03_aa(p1_va, h1_oa, cind) =  1.0 * t2_aaaa(p1_va, p3_va, h2_oa, h1_oa) * chol3d_aa_ov(h2_oa, p3_va, cind)) // o2v2m
        (_a03_aa(p1_va, h1_oa, cind) += -1.0 * t2_abab(p1_va, p3_vb, h1_oa, h2_ob) * chol3d_bb_ov(h2_ob, p3_vb, cind)) // o2v2m
        (_a03_bb(p1_vb, h1_ob, cind) = -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h1_ob) * chol3d_aa_ov(h2_oa, p3_va, cind)) // o2v2m
        (_a03_bb(p1_vb, h1_ob, cind) +=  1.0 * t2_bbbb(p1_vb, p3_vb, h2_ob, h1_ob) * chol3d_bb_ov(h2_ob, p3_vb, cind)) // o2v2m
        (_a04_aa(h2_oa, h1_oa)       =  1.0 * chol3d_aa_ov(h2_oa, p1_va, cind) * _a03_aa(p1_va, h1_oa, cind)) // o2vm
        (_a04_bb(h2_ob, h1_ob)       +=  1.0 * chol3d_bb_ov(h2_ob, p1_vb, cind) * _a03_bb(p1_vb, h1_ob, cind)) // o2vm
        (i0_aa(p2_va, h1_oa)         +=  1.0 * t1_aa(p2_va, h2_oa) * _a04_aa(h2_oa, h1_oa))                 // o2v
        (i0_bb(p2_vb, h1_ob)         +=  1.0 * t1_bb(p2_vb, h2_ob) * _a04_bb(h2_ob, h1_ob))                 // o2v
        (i0_aa(p1_va, h2_oa)         +=  1.0 * chol3d_aa_vo(p1_va, h2_oa, cind) * _a02(cind))         // ovm
        (i0_bb(p1_vb, h2_ob)         +=  1.0 * chol3d_bb_vo(p1_vb, h2_ob, cind) * _a02(cind))         // ovm
        (_a05_aa(h2_oa, p1_va)       = -1.0 * chol3d_aa_ov(h3_oa, p1_va, cind) * _a01_aa(h2_oa, h3_oa, cind)) // o2vm
        (_a05_bb(h2_ob, p1_vb)       = -1.0 * chol3d_bb_ov(h3_ob, p1_vb, cind) * _a01_bb(h2_ob, h3_ob, cind)) // o2vm
        (i0_aa(p2_va, h1_oa)         +=  1.0 * t2_aaaa(p1_va, p2_va, h2_oa, h1_oa) * _a05_aa(h2_oa, p1_va))         // o2v
        (i0_bb(p2_vb, h1_ob)         +=  1.0 * t2_abab(p1_va, p2_vb, h2_oa, h1_ob) * _a05_aa(h2_oa, p1_va))         // o2v
        (i0_aa(p2_va, h1_oa)         +=  1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a05_bb(h2_ob, p1_vb))         // o2v
        (i0_bb(p2_vb, h1_ob)         +=  1.0 * t2_bbbb(p1_vb, p2_vb, h2_ob, h1_ob) * _a05_bb(h2_ob, p1_vb))         // o2v
        (i0_aa(p2_va, h1_oa)         += -1.0 * chol3d_aa_vv(p2_va, p1_va, cind) * _a03_aa(p1_va, h1_oa, cind)) // ov2m
        (i0_bb(p2_vb, h1_ob)         += -1.0 * chol3d_bb_vv(p2_vb, p1_vb, cind) * _a03_bb(p1_vb, h1_ob, cind)) // ov2m
        (_a03_aa(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_aa_vv(p2_va, p1_va, cind))         // ov2m
        (_a03_bb(p2_vb, h2_ob, cind) += -1.0 * t1_bb(p1_vb, h2_ob) * chol3d_bb_vv(p2_vb, p1_vb, cind))         // ov2m
        (i0_aa(p1_va, h2_oa)         += -1.0 * _a03_aa(p1_va, h2_oa, cind) * _a02(cind))           // ovm
        (i0_bb(p1_vb, h2_ob)         += -1.0 * _a03_bb(p1_vb, h2_ob, cind) * _a02(cind))           // ovm
        (_a03_aa(p2_va, h3_oa, cind) += -1.0 * t1_aa(p2_va, h3_oa) * _a02(cind))                   // ovm
        (_a03_bb(p2_vb, h3_ob, cind) += -1.0 * t1_bb(p2_vb, h3_ob) * _a02(cind))                   // ovm
        (_a03_aa(p2_va, h3_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01_aa(h2_oa, h3_oa, cind))           // o2vm
        (_a03_bb(p2_vb, h3_ob, cind) +=  1.0 * t1_bb(p2_vb, h2_ob) * _a01_bb(h2_ob, h3_ob, cind))           // o2vm
        (_a01_aa(h3_oa, h1_oa, cind) +=  1.0 * chol3d_aa_oo(h3_oa, h1_oa, cind))                      // o2m
        (_a01_bb(h3_ob, h1_ob, cind) +=  1.0 * chol3d_bb_oo(h3_ob, h1_ob, cind))                      // o2m        
        (i0_aa(p2_va, h1_oa)         +=  1.0 * _a01_aa(h3_oa, h1_oa, cind) * _a03_aa(p2_va, h3_oa, cind))   // o2vm
        (i0_aa(p2_va, h1_oa)         += -1.0 * t1_aa(p2_va, h7_oa) * f1_aa_oo(h7_oa, h1_oa))                 // o2v
        (i0_aa(p2_va, h1_oa)         +=  1.0 * t1_aa(p3_va, h1_oa) * f1_aa_vv(p2_va, p3_va))                   // ov2
        (i0_bb(p2_vb, h1_ob)         +=  1.0 * _a01_bb(h3_ob, h1_ob, cind) * _a03_bb(p2_vb, h3_ob, cind))   // o2vm
        (i0_bb(p2_vb, h1_ob)         += -1.0 * t1_bb(p2_vb, h7_ob) * f1_bb_oo(h7_ob, h1_ob))                 // o2v
        (i0_bb(p2_vb, h1_ob)         +=  1.0 * t1_bb(p3_vb, h1_ob) * f1_bb_vv(p2_vb, p3_vb))                   // ov2
        (i0(p2_va, h1_oa) += i0_aa(p2_va, h1_oa))
        (i0(p2_vb, h1_ob) += i0_bb(p2_vb, h1_ob))
        ;
        sch.deallocate(_a02, _a01_aa,_a01_bb,_a03_aa,_a03_bb,_a04_aa,_a04_bb,_a05_aa,_a05_bb);
        // .execute();
}

template<typename T>
void ccsd_t2(/* ExecutionContext& ec, */
             Scheduler& sch,
             const TiledIndexSpace& MO,const TiledIndexSpace& CI, 
             Tensor<T>& i0, const Tensor<T>& t1, Tensor<T>& t2, 
             const Tensor<T>& f1, Tensor<T>& chol3d) {
                 
    // const TiledIndexSpace &O = MO("occ");
    // const TiledIndexSpace &V = MO("virt");
    // const TiledIndexSpace &N = MO("all");

    auto [cind] = CI.labels<1>("all");
    auto [p3, p4] = MO.labels<2>("virt");
    auto [h1, h2] = MO.labels<2>("occ");

    auto [p1_va, p2_va, p3_va, p4_va, p5_va, p8_va] = v_alpha.labels<6>("all");
    auto [p1_vb, p2_vb, p3_vb, p4_vb, p6_vb, p8_vb] = v_beta.labels<6>("all");
    auto [h1_oa, h2_oa, h3_oa, h4_oa, h7_oa, h9_oa] = o_alpha.labels<6>("all");
    auto [h1_ob, h2_ob, h3_ob, h4_ob, h8_ob, h9_ob] = o_beta.labels<6>("all");

    
    // Tensor<T> _a001{{V,V}, {1,1}};
    // Tensor<T> _a002{{V,V}, {1,1}};
    // // Tensor<T> _a004{{V,V,O,O}, {2,2}};
    // Tensor<T> _a006{{O,O}, {1,1}};
    // Tensor<T> _a008{{O,O,CI}, {1,1}};
    // Tensor<T> _a009{{O,O,CI}, {1,1}};
    // Tensor<T> _a017{{V,O,CI}, {1,1}};
    // Tensor<T> _a019{{O,O,O,O}, {2,2}};
    // Tensor<T> _a020{{V,O,V,O}, {2,2}};
    // Tensor<T> _a021{{V,V,CI}, {1,1}};
    // Tensor<T> _a022{{V,V,V,V}, {2,2}};

    Tensor<T> _a007{CI};
    Tensor<T> _a017_aa{{v_alpha,o_alpha,CI},{1,1}};
    Tensor<T> _a017_bb{{v_beta,o_beta,CI},{1,1}};
    Tensor<T> _a006_aa{{o_alpha,o_alpha},{1,1}};
    Tensor<T> _a006_bb{{o_beta,o_beta},{1,1}};
    Tensor<T> _a009_aa{{o_alpha,o_alpha,CI},{1,1}};
    Tensor<T> _a009_bb{{o_beta,o_beta,CI},{1,1}};
    Tensor<T> _a021_aa{{v_alpha,v_alpha,CI},{1,1}};
    Tensor<T> _a021_bb{{v_beta,v_beta,CI},{1,1}};
    Tensor<T> _a008_aa{{o_alpha,o_alpha,CI},{1,1}};
    Tensor<T> _a008_bb{{o_beta,o_beta,CI},{1,1}};
    Tensor<T> _a001_aa{{v_alpha,v_alpha},{1,1}};
    Tensor<T> _a001_bb{{v_beta,v_beta},{1,1}};

    Tensor<T> _a019_aaaa{{o_alpha,o_alpha,o_alpha,o_alpha},{2,2}};
    Tensor<T> _a019_abab{{o_alpha,o_beta,o_alpha,o_beta},{2,2}};
    Tensor<T> _a019_bbbb{{o_beta,o_beta,o_beta,o_beta},{2,2}};
    Tensor<T> _a019_abba{{o_alpha,o_beta,o_beta,o_alpha},{2,2}};
    // Tensor<T> _a019_baab{o_beta,o_alpha,o_alpha,o_beta},{2,2}};
    Tensor<T> _a019_baba{{o_beta,o_alpha,o_beta,o_alpha},{2,2}};
    
    Tensor<T> _a020_aaaa{{v_alpha,o_alpha,v_alpha,o_alpha},{2,2}};
    Tensor<T> _a020_abab{{v_alpha,o_beta,v_alpha,o_beta},{2,2}};
    Tensor<T> _a020_baab{{v_beta,o_alpha,v_alpha,o_beta},{2,2}};
    Tensor<T> _a020_abba{{v_alpha,o_beta,v_beta,o_alpha},{2,2}};

    Tensor<T> _a020_baba{{v_beta,o_alpha,v_beta,o_alpha},{2,2}};
    Tensor<T> _a020_bbbb{{v_beta,o_beta,v_beta,o_beta},{2,2}};
    Tensor<T> _a022_aaaa{{v_alpha,v_alpha,v_alpha,v_alpha},{2,2}};
    Tensor<T> _a022_abab{{v_alpha,v_beta,v_alpha,v_beta},{2,2}};
    // Tensor<T> _a022_baba{v_beta,v_alpha,v_beta,v_alpha},{2,2}};
    Tensor<T> _a022_bbbb{{v_beta,v_beta,v_beta,v_beta},{2,2}};

    Tensor<T> i0_aaaa = r2_aaaa;
    Tensor<T> i0_bbbb = r2_bbbb;

 //------------------------------CD------------------------------
    #if 1
    // sch.allocate(_a001, _a002, _a006, _a007,
    //              _a008, _a009, _a017, _a019, _a020, _a021,
    //              _a022); 

    sch 
        (i0(p3, p4, h1, h2) = 0)
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa) =  0)
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob) =  0)
        ;
    
    #endif   
        //_a007
    sch.allocate(_a007,_a001_aa,_a001_bb,_a017_aa,_a017_bb,
        _a006_aa,_a006_bb,_a009_aa,_a009_bb,_a021_aa,_a021_bb,_a008_aa,
        _a008_bb,_a019_aaaa,_a019_abab,_a019_abba,_a019_baba,_a019_bbbb,_a020_aaaa,_a020_baba,
        _a020_abab,_a020_baab,_a020_bbbb,_a020_abba,_a022_aaaa,_a022_abab,_a022_bbbb
        );

    // sch (_a007() = 0)
    //     (_a017_aa() = 0)
    //     (_a017_bb() = 0)
    //     (_a006_aa() = 0)
    //     (_a006_bb() = 0)
    //     (_a009_aa() = 0)
    //     (_a009_bb() = 0)
    //     (_a021_aa() = 0)
    //     (_a021_bb() = 0)     
    //     (_a008_aa() = 0)
    //     (_a008_bb() = 0)
    //     (_a001_aa() = 0)
    //     (_a001_bb() = 0)

    //     (_a019_aaaa() = 0)
    //     (_a019_abab() = 0)
    //     (_a019_bbbb() = 0)
    //     (_a019_abba() = 0)
    //     (_a019_baba() = 0)

    //     (_a020_aaaa() = 0)
    //     (_a020_abab() = 0)
    //     (_a020_baab() = 0)
    //     (_a020_abba() = 0)
    //     (_a020_baba() = 0)
    //     (_a020_bbbb() = 0)

    //     (_a022_aaaa() = 0)
    //     (_a022_abab() = 0)
    //     (_a022_bbbb() = 0);

    sch 
        (_a017_aa(p3_va, h2_oa, cind) = -1.0 * t2_aaaa(p1_va, p3_va, h3_oa, h2_oa) * chol3d_aa_ov(h3_oa, p1_va, cind))
        (_a017_bb(p3_vb, h2_ob, cind) = -1.0 * t2_bbbb(p1_vb, p3_vb, h3_ob, h2_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind))

        (_a017_bb(p3_vb, h2_ob, cind) += -1.0 * t2_abab(p1_va, p3_vb, h3_oa, h2_ob) * chol3d_aa_ov(h3_oa, p1_va, cind))
        (_a017_aa(p3_va, h2_oa, cind) += -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h3_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind))
        (_a006_aa(h4_oa, h1_oa) = -1.0 * chol3d_aa_ov(h4_oa, p2_va, cind) * _a017_aa(p2_va, h1_oa, cind))
        (_a006_bb(h4_ob, h1_ob) = -1.0 * chol3d_bb_ov(h4_ob, p2_vb, cind) * _a017_bb(p2_vb, h1_ob, cind))
        (_a007(cind)      =  1.0 * chol3d_aa_ov(h4_oa, p1_va, cind) * t1_aa(p1_va, h4_oa))
        (_a007(cind)     +=  1.0 * chol3d_bb_ov(h4_ob, p1_vb, cind) * t1_bb(p1_vb, h4_ob))
        (_a009_aa(h3_oa, h2_oa, cind)  =  1.0 * chol3d_aa_ov(h3_oa, p1_va, cind) * t1_aa(p1_va, h2_oa))
        (_a009_bb(h3_ob, h2_ob, cind)  =  1.0 * chol3d_bb_ov(h3_ob, p1_vb, cind) * t1_bb(p1_vb, h2_ob))
        (_a021_aa(p3_va, p1_va, cind)  = -0.5 * chol3d_aa_ov(h3_oa, p1_va, cind) * t1_aa(p3_va, h3_oa))
        (_a021_bb(p3_vb, p1_vb, cind)  = -0.5 * chol3d_bb_ov(h3_ob, p1_vb, cind) * t1_bb(p3_vb, h3_ob))
        (_a021_aa(p3_va, p1_va, cind) +=  0.5 * chol3d_aa_vv(p3_va, p1_va, cind))
        (_a021_bb(p3_vb, p1_vb, cind) +=  0.5 * chol3d_bb_vv(p3_vb, p1_vb, cind))
        (_a017_aa(p3_va, h2_oa, cind) += -2.0 * t1_aa(p2_va, h2_oa) * _a021_aa(p3_va, p2_va, cind))
        (_a017_bb(p3_vb, h2_ob, cind) += -2.0 * t1_bb(p2_vb, h2_ob) * _a021_bb(p3_vb, p2_vb, cind))
        (_a008_aa(h3_oa, h1_oa, cind)  =  1.0 * _a009_aa(h3_oa, h1_oa, cind))
        (_a008_bb(h3_ob, h1_ob, cind)  =  1.0 * _a009_bb(h3_ob, h1_ob, cind))
        (_a009_aa(h3_oa, h1_oa, cind) +=  1.0 * chol3d_aa_oo(h3_oa, h1_oa, cind))
        (_a009_bb(h3_ob, h1_ob, cind) +=  1.0 * chol3d_bb_oo(h3_ob, h1_ob, cind));
        
    sch
        (_a001_aa(p4_va, p2_va)  = -2.0 * _a021_aa(p4_va, p2_va, cind) * _a007(cind))
        (_a001_bb(p4_vb, p2_vb)  = -2.0 * _a021_bb(p4_vb, p2_vb, cind) * _a007(cind))
        (_a001_aa(p4_va, p2_va) += -1.0 * _a017_aa(p4_va, h2_oa, cind) * chol3d_aa_ov(h2_oa, p2_va, cind))
        (_a001_bb(p4_vb, p2_vb) += -1.0 * _a017_bb(p4_vb, h2_ob, cind) * chol3d_bb_ov(h2_ob, p2_vb, cind))
        (_a006_aa(h4_oa, h1_oa) +=  1.0 * _a009_aa(h4_oa, h1_oa, cind) * _a007(cind))
        (_a006_bb(h4_ob, h1_ob) +=  1.0 * _a009_bb(h4_ob, h1_ob, cind) * _a007(cind))
        (_a006_aa(h4_oa, h1_oa) += -1.0 * _a009_aa(h3_oa, h1_oa, cind) * _a008_aa(h4_oa, h3_oa, cind))
        (_a006_bb(h4_ob, h1_ob) += -1.0 * _a009_bb(h3_ob, h1_ob, cind) * _a008_bb(h4_ob, h3_ob, cind))
        (_a019_aaaa(h4_oa, h3_oa, h1_oa, h2_oa) =  0.25 * _a009_aa(h4_oa, h1_oa, cind) * _a009_aa(h3_oa, h2_oa, cind)) 
        (_a019_abab(h4_oa, h3_ob, h1_oa, h2_ob) =  0.25 * _a009_aa(h4_oa, h1_oa, cind) * _a009_bb(h3_ob, h2_ob, cind))
        (_a019_bbbb(h4_ob, h3_ob, h1_ob, h2_ob) =  0.25 * _a009_bb(h4_ob, h1_ob, cind) * _a009_bb(h3_ob, h2_ob, cind)) 
        (_a020_aaaa(p4_va, h4_oa, p1_va, h1_oa) = -2.0  * _a009_aa(h4_oa, h1_oa, cind) * _a021_aa(p4_va, p1_va, cind))
        (_a020_abab(p4_va, h4_ob, p1_va, h1_ob) = -2.0  * _a009_bb(h4_ob, h1_ob, cind) * _a021_aa(p4_va, p1_va, cind))
        (_a020_baba(p4_vb, h4_oa, p1_vb, h1_oa) = -2.0  * _a009_aa(h4_oa, h1_oa, cind) * _a021_bb(p4_vb, p1_vb, cind))
        (_a020_bbbb(p4_vb, h4_ob, p1_vb, h1_ob) = -2.0  * _a009_bb(h4_ob, h1_ob, cind) * _a021_bb(p4_vb, p1_vb, cind))
        ;

    sch
        (_a017_aa(p3_va, h2_oa, cind) +=  1.0 * t1_aa(p3_va, h3_oa) * chol3d_aa_oo(h3_oa, h2_oa, cind))
        (_a017_bb(p3_vb, h2_ob, cind) +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_bb_oo(h3_ob, h2_ob, cind))
        (_a017_aa(p3_va, h2_oa, cind) += -1.0 * chol3d_aa_vo(p3_va, h2_oa, cind))
        (_a017_bb(p3_vb, h2_ob, cind) += -1.0 * chol3d_bb_vo(p3_vb, h2_ob, cind))

        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa) =  0.5 * _a017_aa(p3_va, h1_oa, cind) * _a017_aa(p4_va, h2_oa, cind))
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob) =  0.5 * _a017_bb(p3_vb, h1_ob, cind) * _a017_bb(p4_vb, h2_ob, cind))
        (i0(p3_va, p4_vb, h1_oa, h2_ob) +=  1.0 * _a017_aa(p3_va, h1_oa, cind) * _a017_bb(p4_vb, h2_ob, cind))
        ;

    sch 
        (_a022_aaaa(p3_va,p4_va,p2_va,p1_va) = _a021_aa(p3_va,p2_va,cind) * _a021_aa(p4_va,p1_va,cind))
        (_a022_abab(p3_va,p4_vb,p2_va,p1_vb) = _a021_aa(p3_va,p2_va,cind) * _a021_bb(p4_vb,p1_vb,cind))
        (_a022_bbbb(p3_vb,p4_vb,p2_vb,p1_vb) = _a021_bb(p3_vb,p2_vb,cind) * _a021_bb(p4_vb,p1_vb,cind))

        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)  +=  1.0 * _a022_aaaa(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa))
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)  +=  1.0 * _a022_bbbb(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob))
        
        (i0(p3_va, p4_vb, h1_oa, h2_ob)  +=  4.0 * _a022_abab(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob))
        
        (_a019_aaaa(h4_oa, h3_oa, h1_oa, h2_oa) += -0.125 * _a004_aaaa(p1_va, p2_va, h3_oa, h4_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa))
        (_a019_abab(h4_oa, h3_ob, h1_oa, h2_ob) +=  0.25  * _a004_abab(p1_va, p2_vb, h4_oa, h3_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)) 
        (_a019_bbbb(h4_ob, h3_ob, h1_ob, h2_ob) += -0.125 * _a004_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob))

        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa) +=  1.0 * _a019_aaaa(h4_oa, h3_oa, h1_oa, h2_oa) * t2_aaaa(p3_va, p4_va, h4_oa, h3_oa))
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob) +=  1.0 * _a019_bbbb(h4_ob, h3_ob, h1_ob, h2_ob) * t2_bbbb(p3_vb, p4_vb, h4_ob, h3_ob))
        
        (i0(p3_va, p4_vb, h1_oa, h2_ob) +=  4.0 * _a019_abab(h4_oa, h3_ob, h1_oa, h2_ob) * t2_abab(p3_va, p4_vb, h4_oa, h3_ob))
        

        (_a020_aaaa(p1_va, h3_oa, p4_va, h2_oa) +=  0.5   * _a004_aaaa(p2_va, p4_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)) 
        (_a020_baab(p1_vb, h3_oa, p4_va, h2_ob) =  -0.5   * _a004_aaaa(p2_va, p4_va, h3_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)) 
        (_a020_abba(p1_va, h3_ob, p4_vb, h2_oa) =  -0.5   * _a004_bbbb(p2_vb, p4_vb, h3_ob, h1_ob) * t2_abab(p1_va,p2_vb,h2_oa,h1_ob))
        (_a020_bbbb(p1_vb, h3_ob, p4_vb, h2_ob) +=  0.5   * _a004_bbbb(p2_vb, p4_vb, h3_ob, h1_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob))
        
        (_a020_baba(p1_vb, h7_oa, p6_vb, h2_oa) +=  1.0   * _a004_abab(p5_va, p6_vb, h7_oa, h8_ob) * t2_abab(p5_va,p1_vb,h2_oa,h8_ob))
        
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa) +=  1.0 * _a020_aaaa(p4_va, h4_oa, p1_va, h1_oa) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa))
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa) += -1.0 * _a020_abba(p4_va, h4_ob, p1_vb, h1_oa) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob))
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob) +=  1.0 * _a020_bbbb(p4_vb, h4_ob, p1_vb, h1_ob) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob))
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0 * _a020_baab(p4_vb, h4_oa, p1_va, h1_ob) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob))
        
        (i0(p3_va, p1_vb, h2_oa, h4_ob) +=  1.0 * _a020_baba(p1_vb, h7_oa, p6_vb, h2_oa) * t2_abab(p3_va, p6_vb, h7_oa, h4_ob))
        (i0(p3_va, p1_vb, h2_oa, h4_ob) +=  1.0 * _a020_abab(p3_va, h8_ob, p5_va, h4_ob) * t2_abab(p5_va, p1_vb, h2_oa, h8_ob))
        
        (i0(p3_va, p4_vb, h2_oa, h1_ob) +=  1.0 * _a020_bbbb(p4_vb, h4_ob, p1_vb, h1_ob) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob))
        (i0(p3_va, p4_vb, h2_oa, h1_ob) += -1.0 * _a020_baab(p4_vb, h4_oa, p1_va, h1_ob) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa))
        (i0(p4_va, p3_vb, h1_oa, h2_ob) +=  1.0 * _a020_aaaa(p4_va, h4_oa, p1_va, h1_oa) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob))
        (i0(p4_va, p3_vb, h1_oa, h2_ob) += -1.0 * _a020_abba(p4_va, h4_ob, p1_vb, h1_oa) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob))
        ;

    sch
        (_a001_aa(p4_va, p1_va) += -1 * f1_aa_vv(p4_va, p1_va))
        (_a001_bb(p4_vb, p1_vb) += -1 * f1_bb_vv(p4_vb, p1_vb))
        (_a006_aa(h9_oa, h1_oa) += f1_aa_oo(h9_oa, h1_oa))
        (_a006_bb(h9_ob, h1_ob) += f1_bb_oo(h9_ob, h1_ob))
        (_a006_aa(h9_oa, h1_oa) += t1_aa(p8_va, h1_oa) * f1_aa_ov(h9_oa, p8_va))
        (_a006_bb(h9_ob, h1_ob) += t1_bb(p8_vb, h1_ob) * f1_bb_ov(h9_ob, p8_vb))
        
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa) += -0.5 * t2_aaaa(p3_va, p2_va, h1_oa, h2_oa) * _a001_aa(p4_va, p2_va))
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob) += -0.5 * t2_bbbb(p3_vb, p2_vb, h1_ob, h2_ob) * _a001_bb(p4_vb, p2_vb))
        (i0(p3_va, p4_vb, h1_oa, h2_ob) += -1.0 * t2_abab(p3_va, p2_vb, h1_oa, h2_ob) * _a001_bb(p4_vb, p2_vb))
        (i0(p4_va, p3_vb, h1_oa, h2_ob) += -1.0 * t2_abab(p2_va, p3_vb, h1_oa, h2_ob) * _a001_aa(p4_va, p2_va))

        (i0_aaaa(p3_va, p4_va, h2_oa, h1_oa) += -0.5 * t2_aaaa(p3_va, p4_va, h3_oa, h1_oa) * _a006_aa(h3_oa, h2_oa))
        (i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob) += -0.5 * t2_bbbb(p3_vb, p4_vb, h3_ob, h1_ob) * _a006_bb(h3_ob, h2_ob))
        (i0(p3_va, p4_vb, h2_oa, h1_ob) += -1.0 * t2_abab(p3_va, p4_vb, h3_oa, h1_ob) * _a006_aa(h3_oa, h2_oa))
        (i0(p3_va, p4_vb, h1_oa, h2_ob) += -1.0 * t2_abab(p3_va, p4_vb, h1_oa, h3_ob) * _a006_bb(h3_ob, h2_ob))

        (i0(p3_va, p4_va, h1_oa, h2_oa) +=  1.0 * i0_aaaa(p3_va, p4_va, h1_oa, h2_oa))        
        (i0(p3_va, p4_va, h1_oa, h2_oa) +=  1.0 * i0_aaaa(p4_va, p3_va, h2_oa, h1_oa))        
        (i0(p3_va, p4_va, h1_oa, h2_oa) += -1.0 * i0_aaaa(p3_va, p4_va, h2_oa, h1_oa))        
        (i0(p3_va, p4_va, h1_oa, h2_oa) += -1.0 * i0_aaaa(p4_va, p3_va, h1_oa, h2_oa))
        (i0(p3_vb, p4_vb, h1_ob, h2_ob) +=  1.0 * i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob))
        (i0(p3_vb, p4_vb, h1_ob, h2_ob) +=  1.0 * i0_bbbb(p4_vb, p3_vb, h2_ob, h1_ob))
        (i0(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0 * i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)) 
        (i0(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0 * i0_bbbb(p4_vb, p3_vb, h1_ob, h2_ob))
        ;

        
    sch.deallocate(_a007,_a001_aa,_a001_bb,_a017_aa,_a017_bb,
        _a006_aa,_a006_bb,_a009_aa,_a009_bb,_a021_aa,_a021_bb,_a008_aa,
        _a008_bb,_a019_aaaa,_a019_abab,_a019_abba,_a019_baba,_a019_bbbb,_a020_aaaa,_a020_baba,
        _a020_abab,_a020_baab,_a020_bbbb,_a020_abba,_a022_aaaa,_a022_abab,_a022_bbbb //_a022_baba,_a019_baab
        );

    // sch.deallocate(_a001, _a006, 
    //     _a008, _a009, _a017, _a019, _a020, _a021,
    //     _a022); //_a007
    //-----------------------------CD----------------------------------
    
    // sch.execute();

}


template<typename T>
std::tuple<double,double> cd_ccsd_driver_NK(ExecutionContext& ec, const TiledIndexSpace& MO,
                    const TiledIndexSpace& CI,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, 
                   Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s, 
                   std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s, 
                   std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted,
                   int maxiter, double thresh,
                   double zshiftl, int ndiis, 
                   const TAMM_SIZE& noab,
                   Tensor<T>& cv3d, bool writet=false, bool ccsd_restart=false, std::string out_fp="") {

    std::string t1file = out_fp+".t1amp";
    std::string t2file = out_fp+".t2amp";                       

    std::cout.precision(15);

    double residual = 0.0;
    double energy = 0.0;

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");
    auto [cind] = CI.labels<1>("all");
    // auto [p1, p2] = MO.labels<2>("virt");
    // auto [h3, h4] = MO.labels<2>("occ");


    const int otiles = O.num_tiles();
    const int vtiles = V.num_tiles();
    const int oabtiles = otiles/2;
    const int vabtiles = vtiles/2;

    o_alpha = {MO("occ"), range(oabtiles)};
    v_alpha = {MO("virt"), range(vabtiles)};
    o_beta = {MO("occ"), range(oabtiles,otiles)};
    v_beta = {MO("virt"), range(vabtiles,vtiles)};

    auto [p1_va, p2_va] = v_alpha.labels<2>("all");
    auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
    auto [h3_oa, h4_oa] = o_alpha.labels<2>("all");
    auto [h3_ob, h4_ob] = o_beta.labels<2>("all");


    t1_aa = {{v_alpha,o_alpha},{1,1}}; 
    t1_bb = {{v_beta,o_beta},{1,1}};
    t2_aaaa = {{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    t2_abab = {{v_alpha,v_beta,o_alpha,o_beta},{2,2}};
    t2_bbbb = {{v_beta,v_beta,o_beta,o_beta},{2,2}};

    _a004_aaaa = {{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    _a004_abab = {{v_alpha,v_beta,o_alpha,o_beta},{2,2}};
    _a004_bbbb = {{v_beta,v_beta,o_beta,o_beta},{2,2}};

    f1_aa_oo = {{o_alpha,o_alpha},{1,1}};
    f1_aa_ov = {{o_alpha,v_alpha},{1,1}};
    f1_aa_vo = {{v_alpha,o_alpha},{1,1}};
    f1_aa_vv = {{v_alpha,v_alpha},{1,1}};

    f1_bb_oo = {{o_beta,o_beta},{1,1}};
    f1_bb_ov = {{o_beta,v_beta},{1,1}};
    f1_bb_vo = {{v_beta,o_beta},{1,1}};
    f1_bb_vv = {{v_beta,v_beta},{1,1}};

    chol3d_aa_oo = {{o_alpha,o_alpha,CI},{1,1}};
    chol3d_aa_ov = {{o_alpha,v_alpha,CI},{1,1}};
    chol3d_aa_vo = {{v_alpha,o_alpha,CI},{1,1}};
    chol3d_aa_vv = {{v_alpha,v_alpha,CI},{1,1}};

    chol3d_bb_oo = {{o_beta,o_beta,CI},{1,1}};
    chol3d_bb_ov = {{o_beta,v_beta,CI},{1,1}};
    chol3d_bb_vo = {{v_beta,o_beta,CI},{1,1}};
    chol3d_bb_vv = {{v_beta,v_beta,CI},{1,1}};

    r1_aa = {{v_alpha,o_alpha},{1,1}}; 
    r1_bb = {{v_beta,o_beta},{1,1}};

    r2_aaaa = {{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    //r2_abab = {{v_alpha,v_beta,o_alpha,o_beta},{2,2}};
    r2_bbbb = {{v_beta,v_beta,o_beta,o_beta},{2,2}};

    // r2_baba = {v_beta,v_alpha,o_beta,o_alpha};
    // r2_abba = {v_alpha,v_beta,o_beta,o_alpha};
    // r2_baab = {v_beta,v_alpha,o_alpha,o_beta};
    
    Scheduler sch{ec};
    //t2_baba
    sch.allocate(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, r1_aa, r1_bb, 
                r2_aaaa, r2_bbbb,   //r2_abab, r2_baba, r2_abba, r2_baab,
                f1_aa_oo, f1_aa_ov, f1_aa_vo, f1_aa_vv, f1_bb_oo, f1_bb_ov, f1_bb_vo, f1_bb_vv,
                chol3d_aa_oo, chol3d_aa_ov, chol3d_aa_vo, chol3d_aa_vv,
                chol3d_bb_oo, chol3d_bb_ov, chol3d_bb_vo, chol3d_bb_vv);
    sch.allocate(_a004_aaaa,_a004_abab,_a004_bbbb);

    sch
        // (t1_aa(p1_va,h3_oa) = d_t1(p1,h3))
        // (t1_bb(p1_vb,h3_ob) = d_t1(p1,h3))
        // (t2_aaaa(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1,p2,h3,h4))
        // (t2_abab(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1,p2,h3,h4))
        // (t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1,p2,h3,h4))

        // (chol3d_aa_oo() = 0)
        // (chol3d_aa_ov() = 0)
        // (chol3d_aa_vo() = 0)
        // (chol3d_aa_vv() = 0)
        // (chol3d_bb_oo() = 0)
        // (chol3d_bb_ov() = 0)
        // (chol3d_bb_vo() = 0)
        // (chol3d_bb_vv() = 0)

        (chol3d_aa_oo(h3_oa,h4_oa,cind) = cv3d(h3_oa,h4_oa,cind))
        (chol3d_aa_ov(h3_oa,p2_va,cind) = cv3d(h3_oa,p2_va,cind))
        (chol3d_aa_vo(p1_va,h4_oa,cind) = cv3d(p1_va,h4_oa,cind))
        (chol3d_aa_vv(p1_va,p2_va,cind) = cv3d(p1_va,p2_va,cind))
        (chol3d_bb_oo(h3_ob,h4_ob,cind) = cv3d(h3_ob,h4_ob,cind))
        (chol3d_bb_ov(h3_ob,p1_vb,cind) = cv3d(h3_ob,p1_vb,cind))
        (chol3d_bb_vo(p1_vb,h3_ob,cind) = cv3d(p1_vb,h3_ob,cind))
        (chol3d_bb_vv(p1_vb,p2_vb,cind) = cv3d(p1_vb,p2_vb,cind))

        // (f1_aa_oo() = 0)
        // (f1_aa_ov() = 0)
        // (f1_aa_vo() = 0)
        // (f1_aa_vv() = 0)
        // (f1_bb_oo() = 0)
        // (f1_bb_ov() = 0)
        // (f1_bb_vo() = 0)
        // (f1_bb_vv() = 0)

        (f1_aa_oo(h3_oa,h4_oa) = d_f1(h3_oa,h4_oa))
        (f1_aa_ov(h3_oa,p2_va) = d_f1(h3_oa,p2_va))
        (f1_aa_vo(p1_va,h4_oa) = d_f1(p1_va,h4_oa))
        (f1_aa_vv(p1_va,p2_va) = d_f1(p1_va,p2_va))
        (f1_bb_oo(h3_ob,h4_ob) = d_f1(h3_ob,h4_ob))
        (f1_bb_ov(h3_ob,p1_vb) = d_f1(h3_ob,p1_vb))
        (f1_bb_vo(p1_vb,h3_ob) = d_f1(p1_vb,h3_ob))
        (f1_bb_vv(p1_vb,p2_vb) = d_f1(p1_vb,p2_vb));

        // (r1_aa(p1_va,h3_oa) = d_r1(p1,h3))
        // (r1_bb(p1_vb,h3_ob) = d_r1(p1,h3))
        // (r2_aaaa(p1_va,p2_va,h3_oa,h4_oa) = d_r2(p1,p2,h3,h4))
        // (r2_abab(p1_va,p2_vb,h3_oa,h4_ob) = d_r2(p1,p2,h3,h4))
        // (r2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob) = d_r2(p1,p2,h3,h4))

    Tensor<T> d_e{};
    Tensor<T>::allocate(&ec, d_e);

    if(!ccsd_restart) {
        sch
            (d_r1() = 0)
            (d_r2() = 0)

            // (_a004_aaaa(p1_va, p2_va, h4_oa, h3_oa) = 0)
            // (_a004_abab(p1_va, p2_vb, h4_oa, h3_ob) = 0)
            // (_a004_bbbb(p1_vb, p2_vb, h4_ob, h3_ob) = 0)
            (_a004_aaaa(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_aa_vo(p1_va, h4_oa, cind) * chol3d_aa_vo(p2_va, h3_oa, cind))
            (_a004_abab(p1_va, p2_vb, h4_oa, h3_ob) = 1.0 * chol3d_aa_vo(p1_va, h4_oa, cind) * chol3d_bb_vo(p2_vb, h3_ob, cind))
            (_a004_bbbb(p1_vb, p2_vb, h4_ob, h3_ob) = 1.0 * chol3d_bb_vo(p1_vb, h4_ob, cind) * chol3d_bb_vo(p2_vb, h3_ob, cind))
            ;//.execute();

        // Chao: GMRES uses space allocated for DIIS. So the max
        // GMRES iteration is set to be ndiis-1
        // In principle, this is not necessary, but then we
        // need to allocate separate space for GMRES
        int maxgmiter = ndiis-1; 
        double  del = 1.0e-3; // Chao a small constant for finite difference
        Tensor<T> d_r1_residual{};
        Tensor<T> d_r2_residual{};

        Tensor<T>::allocate(&ec, d_r1_residual, d_r2_residual);

        sch(d_e() = 0)(d_r1_residual() = 0)(d_r2_residual() = 0)
          .execute();

        //Chao: allocate space for GMRES projected problem
        T *rhs  = new double[maxgmiter+1]; 
        T *Hmat = new double[(maxgmiter+1)*maxgmiter];
        int nrows, ncols;
        int ione = 1;
        T beta = 1.0, normr1, normr2, rnorm12=1.0;
        const auto timer_start = std::chrono::high_resolution_clock::now();

        maxiter = 100; // Chao: make take this out later
        for(int titer = 0; titer < maxiter; titer++) {
             
           // Chao: save a copy of t1 and t2, this is 
           // a kludge to get around the issue that 
           // rest updates t1 and t2 
           sch
              (d_t1s[1]() = d_t1())
              (d_t2s[1]() = d_t2())
           .execute();

           sch
             (t1_aa(p1_va,h3_oa) = d_t1(p1_va,h3_oa))
             (t1_bb(p1_vb,h3_ob) = d_t1(p1_vb,h3_ob))
             (t2_aaaa(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1_va,p2_va,h3_oa,h4_oa))
             (t2_abab(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1_va,p2_vb,h3_oa,h4_ob))
             // (t2_baba(p1_vb,p2_va,h3_ob,h4_oa) = d_t2(p1_vb,p2_va,h3_ob,h4_oa))
             (t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1_vb,p2_vb,h3_ob,h4_ob))
            .execute();
            // evaluated energy and residual
            ccsd_e(/* ec,  */sch, MO, CI, d_e, d_t1, d_t2, d_f1, cv3d);
            ccsd_t1(/* ec,  */sch, MO, CI, d_r1, d_t1, d_t2, d_f1, cv3d);
            ccsd_t2(/* ec,  */sch, MO, CI, d_r2, d_t1, d_t2, d_f1, cv3d);
            sch.execute();

            std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_t1, d_t2,
                                              d_e, p_evl_sorted, zshiftl, noab);
            update_r2(ec, d_r2());

            const auto timer_end = std::chrono::high_resolution_clock::now();
            auto iter_time = std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

            iteration_print(sys_data, ec.pg(), titer, residual, energy, iter_time);
           
            if(residual < thresh || titer >= maxiter) { break; }

            // Chao: subtract the saved (t1,t2) to recover inv(D)*(r1,r2)
            // again, this is a kludge to get around the issue that rest()
            // updates t1 and t2
            sch
              (d_r1() =  d_t1())
              (d_r1() -= d_t1s[1]())
              (d_r2() =  d_t2())
              (d_r2() -= d_t2s[1]())
            .execute();

            // Chao: copy t1 and t2 back. 
            // again, not necessary, but need to change rest to not to 
            // update t1 t2
            sch
              (d_t1() = d_t1s[1]())
              (d_t2() = d_t2s[1]())
            .execute();
            
            // Chao: shouldn't need to calculate the norm of (d_r1, d_r2) again
            // But what is calculated below seems to be different from the 
            // residual calculated above, not sure why
            normr1 = norm(d_r1);
            normr2 = norm(d_r2);
            beta = sqrt(normr1*normr1 + normr2*normr2);
            // beta = residual;
            // if(ec.pg().rank()==0) cout << "beta =" << beta << endl;

            // normalized residual used for the first column of the 
            // Arnoldi basis
            sch 
               (d_r1s[0]() = d_r1() )
               (d_r2s[0]() = d_r2() )
            .execute();

            scale_ip(d_r1s[0], 1.0/beta);
            scale_ip(d_r2s[0], 1.0/beta);  
          
            // flush the temp array just in case
            for (int j = 0; j < (maxgmiter+1)*maxgmiter; j++) Hmat[j]=0.0;

            // GMRES
            for(int iter = 0; iter < maxgmiter; iter++) {

               // perform Jacobin vector multiplication
               // use d_t1s d_t2s for finite difference calculations                   
               sch
                  (d_t1s[0]()  = del * d_r1s[iter]())
                  (d_t1s[0]() += d_t1())
                  (d_t2s[0]()  = del * d_r2s[iter]())
                  (d_t2s[0]() += d_t2())
               .execute();

               // save a copy of t1 and t2 
               sch
                 (d_t1s[1]() = d_t1s[0]())
                 (d_t2s[1]() = d_t2s[0]())
               .execute();

               // need to do the following before contraction
               sch
               (t1_aa(p1_va,h3_oa) = d_t1s[0](p1_va,h3_oa))
               (t1_bb(p1_vb,h3_ob) = d_t1s[0](p1_vb,h3_ob))
               (t2_aaaa(p1_va,p2_va,h3_oa,h4_oa) = d_t2s[0](p1_va,p2_va,h3_oa,h4_oa))
               (t2_abab(p1_va,p2_vb,h3_oa,h4_ob) = d_t2s[0](p1_va,p2_vb,h3_oa,h4_ob))
               // (t2_baba(p1_vb,p2_va,h3_ob,h4_oa) = d_t2(p1_vb,p2_va,h3_ob,h4_oa))
               (t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2s[0](p1_vb,p2_vb,h3_ob,h4_ob))
               .execute();

               ccsd_t1(/* ec,  */sch, MO, CI, d_r1s[iter+1], d_t1s[0], d_t2s[0], d_f1, cv3d);
               ccsd_t2(/* ec,  */sch, MO, CI, d_r2s[iter+1], d_t1s[0], d_t2s[0], d_f1, cv3d);
               sch.execute();

               std::tie(residual, energy) = rest(ec, MO, d_r1s[iter+1], d_r2s[iter+1], d_t1s[0], d_t2s[0], d_e, p_evl_sorted, zshiftl, noab);
               update_r2(ec, d_r2s[iter+1]());

               // subtract the saved (t1,t2) to recover inv(D)*(r1,r2)
               sch
                 ( d_r1s[iter+1]()  = d_t1s[0]() )
                 ( d_r1s[iter+1]() -= d_t1s[1]() )
                 ( d_r2s[iter+1]()  = d_t2s[0]() )
                 ( d_r2s[iter+1]() -= d_t2s[1]() )
               .execute();

               // take the difference
               sch
                 (d_r1s[iter+1]() -= d_r1())
                 (d_r2s[iter+1]() -= d_r2())
               .execute();

               scale_ip(d_r1s[iter+1], 1.0/del);
               scale_ip(d_r2s[iter+1], 1.0/del);

               // update the Hessenberg matrix, calculate Arnold residual
               for(int j=0; j <= iter; j++) {
                  sch 
                    (d_r1_residual()  = d_r1s[j]()*d_r1s[iter+1]())
                    (d_r1_residual() += d_r2s[j]()*d_r2s[iter+1]())
                    (d_r1s[iter+1]() -= d_r1_residual() * d_r1s[j]() )
                    (d_r2s[iter+1]() -= d_r1_residual() * d_r2s[j]() )
                  .execute();
                  Hmat[j+iter*(maxgmiter+1)] = get_scalar(d_r1_residual);
               }

               normr1 = norm(d_r1s[iter+1]);
               normr2 = norm(d_r2s[iter+1]);
               rnorm12 = sqrt(normr1*normr1+normr2*normr2);
               Hmat[iter+1+iter*(maxgmiter+1)] = rnorm12;

               // normalize the residual to make it a new Arnoldi vector
               scale_ip(d_r1s[iter+1], 1.0/rnorm12);
               scale_ip(d_r2s[iter+1], 1.0/rnorm12);
//Chao check ortho
#if 0
               T orth = 0.0;
               if(ec.pg().rank()==0) cout << "iter = " << iter << endl;
               for(int j=0; j <= iter; j++) {
                  sch 
                     (d_r1_residual()  = d_r1s[j]()*d_r1s[iter+1]())
                  .execute();
                  sch 
                     (d_r1_residual() += d_r2s[j]()*d_r2s[iter+1]())
                  .execute();
                  orth = get_scalar(d_r1_residual);
                  if(ec.pg().rank()==0) cout << "orth = " << orth << endl;
               }
               if(ec.pg().rank()==0) cout << "----" << endl;
#endif
            } //end for iter GMRES

            // now need to solve a linear least squares problem
            // rhs  is a length maxgmiter+1 vector of zeros execpt in the first entry
            for (int j = 0; j < maxgmiter+1;j++)
               rhs[j] = 0.0;
            rhs[0] = beta;
            nrows = maxgmiter + 1;
            ncols = maxgmiter;
            ione = 1;
            if (maxgmiter > 0) {
               LAPACKE_dgels(LAPACK_COL_MAJOR,'N', nrows, ncols, ione, Hmat, nrows,
                             rhs, nrows);
            }
           
            // update t1 and t2
            for(int j=0; j<maxgmiter; j++) {
               sch
                  (d_t1() -= rhs[j] * d_r1s[j]() )
                  (d_t2() -= rhs[j] * d_r2s[j]() )
               .execute();
            }

#if 0
           if(residual < thresh || titer + ndiis >= maxiter) { break; }
           if(ec.pg().rank() == 0) {
               std::cout << " MICROCYCLE DIIS UPDATE:";
               std::cout.width(21);
               std::cout << std::right << std::min(titer + ndiis, maxiter) + 1;
               std::cout.width(21);
               std::cout << std::right << "5" << std::endl;
           }


           std::vector<std::vector<Tensor<T>>> rs{d_r1s, d_r2s};
           std::vector<std::vector<Tensor<T>>> ts{d_t1s, d_t2s};
           std::vector<Tensor<T>> next_t{d_t1, d_t2};
           diis<T>(ec, rs, ts, next_t);
           if(writet) {
               write_to_disk(d_t1,t1file);
               write_to_disk(d_t2,t2file);
           }
#endif
       } // end for outer iteration
       Tensor<T>::deallocate(d_r1_residual, d_r2_residual);
       //Chao deallocate temp storage
       delete [] rhs;
       delete [] Hmat;

    } //no restart
    else {
            sch
            (d_e()=0)
            (t1_aa(p1_va,h3_oa) = d_t1(p1_va,h3_oa))
            (t1_bb(p1_vb,h3_ob) = d_t1(p1_vb,h3_ob))
            (t2_aaaa(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1_va,p2_va,h3_oa,h4_oa))
            (t2_abab(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1_va,p2_vb,h3_oa,h4_ob))
            // (t2_baba(p1_vb,p2_va,h3_ob,h4_oa) = d_t2(p1_vb,p2_va,h3_ob,h4_oa))
            (t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1_vb,p2_vb,h3_ob,h4_ob));

            ccsd_e(/* ec,  */sch, MO, CI, d_e, d_t1, d_t2, d_f1, cv3d);
            sch.execute();
            energy = get_scalar(d_e);
            residual = 0.0;
    }

  sch.deallocate(d_e,_a004_aaaa,_a004_abab,_a004_bbbb);
  //t2_baba
  sch.deallocate(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, r1_aa, r1_bb, 
                r2_aaaa, r2_bbbb,   //r2_abab, r2_baba, r2_abba, r2_baab,
                f1_aa_oo, f1_aa_ov, f1_aa_vo, f1_aa_vv, f1_bb_oo, f1_bb_ov, f1_bb_vo, f1_bb_vv,
                chol3d_aa_oo, chol3d_aa_ov, chol3d_aa_vo, chol3d_aa_vv,
                chol3d_bb_oo, chol3d_bb_ov, chol3d_bb_vo, chol3d_bb_vv).execute();

  return std::make_tuple(residual,energy);


}
