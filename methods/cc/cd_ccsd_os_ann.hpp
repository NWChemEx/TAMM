

#include "cd_ccsd_cs_ann.hpp"


template<typename T>
void ccsd_e_os(/* ExecutionContext &ec, */
            Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de) { 

    auto [cind]                =      CI.labels<1>("all");
    auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
    auto [p1_vb, p2_vb, p3_vb] =  v_beta.labels<3>("all");
    auto [h3_oa, h4_oa, h6_oa] = o_alpha.labels<3>("all");
    auto [h3_ob, h4_ob, h6_ob] =  o_beta.labels<3>("all");
    sch 
    (_a01(cind)                   = t1_aa(p3_va, h4_oa) * chol3d_aa_ov(h4_oa, p3_va, cind), 
		"_a01(cind)                   = t1_aa(p3_va, h4_oa) * chol3d_aa_ov(h4_oa, p3_va, cind)")
    (_a02_aa(h4_oa, h6_oa, cind)  = t1_aa(p3_va, h4_oa) * chol3d_aa_ov(h6_oa, p3_va, cind), 
		"_a02_aa(h4_oa, h6_oa, cind)  = t1_aa(p3_va, h4_oa) * chol3d_aa_ov(h6_oa, p3_va, cind)")
    (_a03_aa(h4_oa, p2_va, cind)  = t2_aaaa(p1_va, p2_va, h3_oa, h4_oa) * chol3d_aa_ov(h3_oa, p1_va, cind), 
		"_a03_aa(h4_oa, p2_va, cind)  = t2_aaaa(p1_va, p2_va, h3_oa, h4_oa) * chol3d_aa_ov(h3_oa, p1_va, cind)")
    (_a03_aa(h4_oa, p2_va, cind) += t2_abab(p2_va, p1_vb, h4_oa, h3_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind), 
		"_a03_aa(h4_oa, p2_va, cind) += t2_abab(p2_va, p1_vb, h4_oa, h3_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind)")
    (_a01(cind)                  += t1_bb(p3_vb, h4_ob) * chol3d_bb_ov(h4_ob, p3_vb, cind), 
		"_a01(cind)                  += t1_bb(p3_vb, h4_ob) * chol3d_bb_ov(h4_ob, p3_vb, cind)")
    (_a02_bb(h4_ob, h6_ob, cind)  = t1_bb(p3_vb, h4_ob) * chol3d_bb_ov(h6_ob, p3_vb, cind), 
		"_a02_bb(h4_ob, h6_ob, cind)  = t1_bb(p3_vb, h4_ob) * chol3d_bb_ov(h6_ob, p3_vb, cind)")
    (_a03_bb(h4_ob, p2_vb, cind)  = t2_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind), 
		"_a03_bb(h4_ob, p2_vb, cind)  = t2_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind)")
    (_a03_bb(h4_ob, p2_vb, cind) += t2_abab(p1_va, p2_vb, h3_oa, h4_ob) * chol3d_aa_ov(h3_oa, p1_va, cind), 
		"_a03_bb(h4_ob, p2_vb, cind) += t2_abab(p1_va, p2_vb, h3_oa, h4_ob) * chol3d_aa_ov(h3_oa, p1_va, cind)")
    (de()                         =  0.5 * _a01() * _a01(), 
		"de()                         =  0.5 * _a01() * _a01()")
    (de()                        += -0.5 * _a02_aa(h4_oa, h6_oa, cind) * _a02_aa(h6_oa, h4_oa, cind), 
		"de()                        += -0.5 * _a02_aa(h4_oa, h6_oa, cind) * _a02_aa(h6_oa, h4_oa, cind)")
    (de()                        += -0.5 * _a02_bb(h4_ob, h6_ob, cind) * _a02_bb(h6_ob, h4_ob, cind), 
		"de()                        += -0.5 * _a02_bb(h4_ob, h6_ob, cind) * _a02_bb(h6_ob, h4_ob, cind)")
    (de()                        +=  0.5 * _a03_aa(h4_oa, p1_va, cind) * chol3d_aa_ov(h4_oa, p1_va, cind), 
		"de()                        +=  0.5 * _a03_aa(h4_oa, p1_va, cind) * chol3d_aa_ov(h4_oa, p1_va, cind)")
    (de()                        +=  0.5 * _a03_bb(h4_ob, p1_vb, cind) * chol3d_bb_ov(h4_ob, p1_vb, cind), 
		"de()                        +=  0.5 * _a03_bb(h4_ob, p1_vb, cind) * chol3d_bb_ov(h4_ob, p1_vb, cind)")
    ;

}

template<typename T>
void ccsd_t1_os(/* ExecutionContext& ec,  */
             Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& i0) {
    
    auto [cind]                       =      CI.labels<1>("all");
    auto [p2]                         =      MO.labels<1>("virt");
    auto [h1]                         =      MO.labels<1>("occ");
    auto [p1_va, p2_va, p3_va]        = v_alpha.labels<3>("all");
    auto [p1_vb, p2_vb, p3_vb]        =  v_beta.labels<3>("all");
    auto [h1_oa, h2_oa, h3_oa, h7_oa] = o_alpha.labels<4>("all");
    auto [h1_ob, h2_ob, h3_ob, h7_ob] =  o_beta.labels<4>("all");
    
    Tensor<T> i0_aa = r1_aa;
    Tensor<T> i0_bb = r1_bb;

    sch
       (i0_aa(p2_va, h1_oa)             =  1.0 * f1_aa_vo(p2_va, h1_oa), 
			 "i0_aa(p2_va, h1_oa)             =  1.0 * f1_aa_vo(p2_va, h1_oa)")
       (i0_bb(p2_vb, h1_ob)             =  1.0 * f1_bb_vo(p2_vb, h1_ob), 
			 "i0_bb(p2_vb, h1_ob)             =  1.0 * f1_bb_vo(p2_vb, h1_ob)")
       (_a01_aa(h2_oa, h1_oa, cind)     =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_aa_ov(h2_oa, p1_va, cind), 
			 "_a01_aa(h2_oa, h1_oa, cind)     =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_aa_ov(h2_oa, p1_va, cind)")                 // ovm
       (_a01_bb(h2_ob, h1_ob, cind)     =  1.0 * t1_bb(p1_vb, h1_ob) * chol3d_bb_ov(h2_ob, p1_vb, cind), 
			 "_a01_bb(h2_ob, h1_ob, cind)     =  1.0 * t1_bb(p1_vb, h1_ob) * chol3d_bb_ov(h2_ob, p1_vb, cind)")                 // ovm
       (_a02(cind)                      =  1.0 * t1_aa(p3_va, h3_oa) * chol3d_aa_ov(h3_oa, p3_va, cind), 
			 "_a02(cind)                      =  1.0 * t1_aa(p3_va, h3_oa) * chol3d_aa_ov(h3_oa, p3_va, cind)")                 // ovm
       (_a02(cind)                     +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_bb_ov(h3_ob, p3_vb, cind), 
			 "_a02(cind)                     +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_bb_ov(h3_ob, p3_vb, cind)")                 // ovm
       (_a03_aa_vo(p1_va, h1_oa, cind)  =  1.0 * t2_aaaa(p1_va, p3_va, h2_oa, h1_oa) * chol3d_aa_ov(h2_oa, p3_va, cind), 
			 "_a03_aa_vo(p1_va, h1_oa, cind)  =  1.0 * t2_aaaa(p1_va, p3_va, h2_oa, h1_oa) * chol3d_aa_ov(h2_oa, p3_va, cind)") // o2v2m
       (_a03_aa_vo(p1_va, h1_oa, cind) += -1.0 * t2_abab(p1_va, p3_vb, h1_oa, h2_ob) * chol3d_bb_ov(h2_ob, p3_vb, cind), 
			 "_a03_aa_vo(p1_va, h1_oa, cind) += -1.0 * t2_abab(p1_va, p3_vb, h1_oa, h2_ob) * chol3d_bb_ov(h2_ob, p3_vb, cind)") // o2v2m
       (_a03_bb_vo(p1_vb, h1_ob, cind)  = -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h1_ob) * chol3d_aa_ov(h2_oa, p3_va, cind), 
			 "_a03_bb_vo(p1_vb, h1_ob, cind)  = -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h1_ob) * chol3d_aa_ov(h2_oa, p3_va, cind)") // o2v2m
       (_a03_bb_vo(p1_vb, h1_ob, cind) +=  1.0 * t2_bbbb(p1_vb, p3_vb, h2_ob, h1_ob) * chol3d_bb_ov(h2_ob, p3_vb, cind), 
			 "_a03_bb_vo(p1_vb, h1_ob, cind) +=  1.0 * t2_bbbb(p1_vb, p3_vb, h2_ob, h1_ob) * chol3d_bb_ov(h2_ob, p3_vb, cind)") // o2v2m
       (_a04_aa(h2_oa, h1_oa)           =  1.0 * chol3d_aa_ov(h2_oa, p1_va, cind) * _a03_aa_vo(p1_va, h1_oa, cind), 
			 "_a04_aa(h2_oa, h1_oa)           =  1.0 * chol3d_aa_ov(h2_oa, p1_va, cind) * _a03_aa_vo(p1_va, h1_oa, cind)")      // o2vm
       (_a04_bb(h2_ob, h1_ob)           =  1.0 * chol3d_bb_ov(h2_ob, p1_vb, cind) * _a03_bb_vo(p1_vb, h1_ob, cind), 
			 "_a04_bb(h2_ob, h1_ob)           =  1.0 * chol3d_bb_ov(h2_ob, p1_vb, cind) * _a03_bb_vo(p1_vb, h1_ob, cind)")      // o2vm
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04_aa(h2_oa, h1_oa), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04_aa(h2_oa, h1_oa)")                            // o2v
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p2_vb, h2_ob) * _a04_bb(h2_ob, h1_ob), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p2_vb, h2_ob) * _a04_bb(h2_ob, h1_ob)")                            // o2v
       (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_aa_vo(p1_va, h2_oa, cind) * _a02(cind), 
			 "i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_aa_vo(p1_va, h2_oa, cind) * _a02(cind)")                          // ovm
       (i0_bb(p1_vb, h2_ob)            +=  1.0 * chol3d_bb_vo(p1_vb, h2_ob, cind) * _a02(cind), 
			 "i0_bb(p1_vb, h2_ob)            +=  1.0 * chol3d_bb_vo(p1_vb, h2_ob, cind) * _a02(cind)")                          // ovm
       (_a05_aa(h2_oa, p1_va)           = -1.0 * chol3d_aa_ov(h3_oa, p1_va, cind) * _a01_aa(h2_oa, h3_oa, cind), 
			 "_a05_aa(h2_oa, p1_va)           = -1.0 * chol3d_aa_ov(h3_oa, p1_va, cind) * _a01_aa(h2_oa, h3_oa, cind)")         // o2vm
       (_a05_bb(h2_ob, p1_vb)           = -1.0 * chol3d_bb_ov(h3_ob, p1_vb, cind) * _a01_bb(h2_ob, h3_ob, cind), 
			 "_a05_bb(h2_ob, p1_vb)           = -1.0 * chol3d_bb_ov(h3_ob, p1_vb, cind) * _a01_bb(h2_ob, h3_ob, cind)")         // o2vm
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t2_aaaa(p1_va, p2_va, h2_oa, h1_oa) * _a05_aa(h2_oa, p1_va), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * t2_aaaa(p1_va, p2_va, h2_oa, h1_oa) * _a05_aa(h2_oa, p1_va)")            // o2v
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_abab(p1_va, p2_vb, h2_oa, h1_ob) * _a05_aa(h2_oa, p1_va), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_abab(p1_va, p2_vb, h2_oa, h1_ob) * _a05_aa(h2_oa, p1_va)")            // o2v
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a05_bb(h2_ob, p1_vb), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a05_bb(h2_ob, p1_vb)")            // o2v
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_bbbb(p1_vb, p2_vb, h2_ob, h1_ob) * _a05_bb(h2_ob, p1_vb), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_bbbb(p1_vb, p2_vb, h2_ob, h1_ob) * _a05_bb(h2_ob, p1_vb)")            // o2v
       (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_aa_vv(p2_va, p1_va, cind) * _a03_aa_vo(p1_va, h1_oa, cind), 
			 "i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_aa_vv(p2_va, p1_va, cind) * _a03_aa_vo(p1_va, h1_oa, cind)")      // ov2m
       (i0_bb(p2_vb, h1_ob)            += -1.0 * chol3d_bb_vv(p2_vb, p1_vb, cind) * _a03_bb_vo(p1_vb, h1_ob, cind), 
			 "i0_bb(p2_vb, h1_ob)            += -1.0 * chol3d_bb_vv(p2_vb, p1_vb, cind) * _a03_bb_vo(p1_vb, h1_ob, cind)")      // ov2m
       (_a03_aa_vo(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_aa_vv(p2_va, p1_va, cind), 
			 "_a03_aa_vo(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_aa_vv(p2_va, p1_va, cind)")                 // ov2m
       (_a03_bb_vo(p2_vb, h2_ob, cind) += -1.0 * t1_bb(p1_vb, h2_ob) * chol3d_bb_vv(p2_vb, p1_vb, cind), 
			 "_a03_bb_vo(p2_vb, h2_ob, cind) += -1.0 * t1_bb(p1_vb, h2_ob) * chol3d_bb_vv(p2_vb, p1_vb, cind)")                 // ov2m
       (i0_aa(p1_va, h2_oa)            += -1.0 * _a03_aa_vo(p1_va, h2_oa, cind) * _a02(cind), 
			 "i0_aa(p1_va, h2_oa)            += -1.0 * _a03_aa_vo(p1_va, h2_oa, cind) * _a02(cind)")                            // ovm
       (i0_bb(p1_vb, h2_ob)            += -1.0 * _a03_bb_vo(p1_vb, h2_ob, cind) * _a02(cind), 
			 "i0_bb(p1_vb, h2_ob)            += -1.0 * _a03_bb_vo(p1_vb, h2_ob, cind) * _a02(cind)")                            // ovm
       (_a03_aa_vo(p2_va, h3_oa, cind) += -1.0 * t1_aa(p2_va, h3_oa) * _a02(cind), 
			 "_a03_aa_vo(p2_va, h3_oa, cind) += -1.0 * t1_aa(p2_va, h3_oa) * _a02(cind)")                                       // ovm
       (_a03_bb_vo(p2_vb, h3_ob, cind) += -1.0 * t1_bb(p2_vb, h3_ob) * _a02(cind), 
			 "_a03_bb_vo(p2_vb, h3_ob, cind) += -1.0 * t1_bb(p2_vb, h3_ob) * _a02(cind)")                                       // ovm
       (_a03_aa_vo(p2_va, h3_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01_aa(h2_oa, h3_oa, cind), 
			 "_a03_aa_vo(p2_va, h3_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01_aa(h2_oa, h3_oa, cind)")                      // o2vm
       (_a03_bb_vo(p2_vb, h3_ob, cind) +=  1.0 * t1_bb(p2_vb, h2_ob) * _a01_bb(h2_ob, h3_ob, cind), 
			 "_a03_bb_vo(p2_vb, h3_ob, cind) +=  1.0 * t1_bb(p2_vb, h2_ob) * _a01_bb(h2_ob, h3_ob, cind)")                      // o2vm
       (_a01_aa(h3_oa, h1_oa, cind)    +=  1.0 * chol3d_aa_oo(h3_oa, h1_oa, cind), 
			 "_a01_aa(h3_oa, h1_oa, cind)    +=  1.0 * chol3d_aa_oo(h3_oa, h1_oa, cind)")                                       // o2m
       (_a01_bb(h3_ob, h1_ob, cind)    +=  1.0 * chol3d_bb_oo(h3_ob, h1_ob, cind), 
			 "_a01_bb(h3_ob, h1_ob, cind)    +=  1.0 * chol3d_bb_oo(h3_ob, h1_ob, cind)")                                       // o2m        
       (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01_aa(h3_oa, h1_oa, cind) * _a03_aa_vo(p2_va, h3_oa, cind), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * _a01_aa(h3_oa, h1_oa, cind) * _a03_aa_vo(p2_va, h3_oa, cind)")           // o2vm
       (i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h7_oa) * f1_aa_oo(h7_oa, h1_oa), 
			 "i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h7_oa) * f1_aa_oo(h7_oa, h1_oa)")                           // o2v
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p3_va, h1_oa) * f1_aa_vv(p2_va, p3_va), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p3_va, h1_oa) * f1_aa_vv(p2_va, p3_va)")                           // ov2
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * _a01_bb(h3_ob, h1_ob, cind) * _a03_bb_vo(p2_vb, h3_ob, cind), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * _a01_bb(h3_ob, h1_ob, cind) * _a03_bb_vo(p2_vb, h3_ob, cind)")           // o2vm
       (i0_bb(p2_vb, h1_ob)            += -1.0 * t1_bb(p2_vb, h7_ob) * f1_bb_oo(h7_ob, h1_ob), 
			 "i0_bb(p2_vb, h1_ob)            += -1.0 * t1_bb(p2_vb, h7_ob) * f1_bb_oo(h7_ob, h1_ob)")                           // o2v
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p3_vb, h1_ob) * f1_bb_vv(p2_vb, p3_vb), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p3_vb, h1_ob) * f1_bb_vv(p2_vb, p3_vb)")                           // ov2
       ;

}

template<typename T>
void ccsd_t2_os(/* ExecutionContext& ec, */
             Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& i0) {
                
    auto [cind]                                     =      CI.labels<1>("all");
    auto [p3, p4]                                   =      MO.labels<2>("virt");
    auto [h1, h2]                                   =      MO.labels<2>("occ");
    auto [p1_va, p2_va, p3_va, p4_va, p5_va, p8_va] = v_alpha.labels<6>("all");
    auto [p1_vb, p2_vb, p3_vb, p4_vb, p6_vb, p8_vb] =  v_beta.labels<6>("all");
    auto [h1_oa, h2_oa, h3_oa, h4_oa, h7_oa, h9_oa] = o_alpha.labels<6>("all");
    auto [h1_ob, h2_ob, h3_ob, h4_ob, h8_ob, h9_ob] =  o_beta.labels<6>("all");                

    Tensor<T> i0_aaaa = r2_aaaa;
    Tensor<T> i0_abab = r2_abab;
    Tensor<T> i0_bbbb = r2_bbbb;

    sch 
        (_a017_aa(p3_va, h2_oa, cind)            = -1.0   * t2_aaaa(p1_va, p3_va, h3_oa, h2_oa) * chol3d_aa_ov(h3_oa, p1_va, cind), 
			  "_a017_aa(p3_va, h2_oa, cind)            = -1.0   * t2_aaaa(p1_va, p3_va, h3_oa, h2_oa) * chol3d_aa_ov(h3_oa, p1_va, cind)")
        (_a017_bb(p3_vb, h2_ob, cind)            = -1.0   * t2_bbbb(p1_vb, p3_vb, h3_ob, h2_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind), 
			  "_a017_bb(p3_vb, h2_ob, cind)            = -1.0   * t2_bbbb(p1_vb, p3_vb, h3_ob, h2_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind)")
        (_a017_bb(p3_vb, h2_ob, cind)           += -1.0   * t2_abab(p1_va, p3_vb, h3_oa, h2_ob) * chol3d_aa_ov(h3_oa, p1_va, cind), 
			  "_a017_bb(p3_vb, h2_ob, cind)           += -1.0   * t2_abab(p1_va, p3_vb, h3_oa, h2_ob) * chol3d_aa_ov(h3_oa, p1_va, cind)")
        (_a017_aa(p3_va, h2_oa, cind)           += -1.0   * t2_abab(p3_va, p1_vb, h2_oa, h3_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind), 
			  "_a017_aa(p3_va, h2_oa, cind)           += -1.0   * t2_abab(p3_va, p1_vb, h2_oa, h3_ob) * chol3d_bb_ov(h3_ob, p1_vb, cind)")
        (_a006_aa(h4_oa, h1_oa)                  = -1.0   * chol3d_aa_ov(h4_oa, p2_va, cind) * _a017_aa(p2_va, h1_oa, cind), 
			  "_a006_aa(h4_oa, h1_oa)                  = -1.0   * chol3d_aa_ov(h4_oa, p2_va, cind) * _a017_aa(p2_va, h1_oa, cind)")
        (_a006_bb(h4_ob, h1_ob)                  = -1.0   * chol3d_bb_ov(h4_ob, p2_vb, cind) * _a017_bb(p2_vb, h1_ob, cind), 
			  "_a006_bb(h4_ob, h1_ob)                  = -1.0   * chol3d_bb_ov(h4_ob, p2_vb, cind) * _a017_bb(p2_vb, h1_ob, cind)")
        (_a007(cind)                             =  1.0   * chol3d_aa_ov(h4_oa, p1_va, cind) * t1_aa(p1_va, h4_oa), 
			  "_a007(cind)                             =  1.0   * chol3d_aa_ov(h4_oa, p1_va, cind) * t1_aa(p1_va, h4_oa)")
        (_a007(cind)                            +=  1.0   * chol3d_bb_ov(h4_ob, p1_vb, cind) * t1_bb(p1_vb, h4_ob), 
			  "_a007(cind)                            +=  1.0   * chol3d_bb_ov(h4_ob, p1_vb, cind) * t1_bb(p1_vb, h4_ob)")
        (_a009_aa(h3_oa, h2_oa, cind)            =  1.0   * chol3d_aa_ov(h3_oa, p1_va, cind) * t1_aa(p1_va, h2_oa), 
			  "_a009_aa(h3_oa, h2_oa, cind)            =  1.0   * chol3d_aa_ov(h3_oa, p1_va, cind) * t1_aa(p1_va, h2_oa)")
        (_a009_bb(h3_ob, h2_ob, cind)            =  1.0   * chol3d_bb_ov(h3_ob, p1_vb, cind) * t1_bb(p1_vb, h2_ob), 
			  "_a009_bb(h3_ob, h2_ob, cind)            =  1.0   * chol3d_bb_ov(h3_ob, p1_vb, cind) * t1_bb(p1_vb, h2_ob)")
        (_a021_aa(p3_va, p1_va, cind)            = -0.5   * chol3d_aa_ov(h3_oa, p1_va, cind) * t1_aa(p3_va, h3_oa), 
			  "_a021_aa(p3_va, p1_va, cind)            = -0.5   * chol3d_aa_ov(h3_oa, p1_va, cind) * t1_aa(p3_va, h3_oa)")
        (_a021_bb(p3_vb, p1_vb, cind)            = -0.5   * chol3d_bb_ov(h3_ob, p1_vb, cind) * t1_bb(p3_vb, h3_ob), 
			  "_a021_bb(p3_vb, p1_vb, cind)            = -0.5   * chol3d_bb_ov(h3_ob, p1_vb, cind) * t1_bb(p3_vb, h3_ob)")
        (_a021_aa(p3_va, p1_va, cind)           +=  0.5   * chol3d_aa_vv(p3_va, p1_va, cind), 
			  "_a021_aa(p3_va, p1_va, cind)           +=  0.5   * chol3d_aa_vv(p3_va, p1_va, cind)")
        (_a021_bb(p3_vb, p1_vb, cind)           +=  0.5   * chol3d_bb_vv(p3_vb, p1_vb, cind), 
			  "_a021_bb(p3_vb, p1_vb, cind)           +=  0.5   * chol3d_bb_vv(p3_vb, p1_vb, cind)")
        (_a017_aa(p3_va, h2_oa, cind)           += -2.0   * t1_aa(p2_va, h2_oa) * _a021_aa(p3_va, p2_va, cind), 
			  "_a017_aa(p3_va, h2_oa, cind)           += -2.0   * t1_aa(p2_va, h2_oa) * _a021_aa(p3_va, p2_va, cind)")
        (_a017_bb(p3_vb, h2_ob, cind)           += -2.0   * t1_bb(p2_vb, h2_ob) * _a021_bb(p3_vb, p2_vb, cind), 
			  "_a017_bb(p3_vb, h2_ob, cind)           += -2.0   * t1_bb(p2_vb, h2_ob) * _a021_bb(p3_vb, p2_vb, cind)")
        (_a008_aa(h3_oa, h1_oa, cind)            =  1.0   * _a009_aa(h3_oa, h1_oa, cind), 
			  "_a008_aa(h3_oa, h1_oa, cind)            =  1.0   * _a009_aa(h3_oa, h1_oa, cind)")
        (_a008_bb(h3_ob, h1_ob, cind)            =  1.0   * _a009_bb(h3_ob, h1_ob, cind), 
			  "_a008_bb(h3_ob, h1_ob, cind)            =  1.0   * _a009_bb(h3_ob, h1_ob, cind)")
        (_a009_aa(h3_oa, h1_oa, cind)           +=  1.0   * chol3d_aa_oo(h3_oa, h1_oa, cind), 
			  "_a009_aa(h3_oa, h1_oa, cind)           +=  1.0   * chol3d_aa_oo(h3_oa, h1_oa, cind)")
        (_a009_bb(h3_ob, h1_ob, cind)           +=  1.0   * chol3d_bb_oo(h3_ob, h1_ob, cind), 
			  "_a009_bb(h3_ob, h1_ob, cind)           +=  1.0   * chol3d_bb_oo(h3_ob, h1_ob, cind)")

        (_a001_aa(p4_va, p2_va)                  = -2.0   * _a021_aa(p4_va, p2_va, cind) * _a007(cind), 
			  "_a001_aa(p4_va, p2_va)                  = -2.0   * _a021_aa(p4_va, p2_va, cind) * _a007(cind)")
        (_a001_bb(p4_vb, p2_vb)                  = -2.0   * _a021_bb(p4_vb, p2_vb, cind) * _a007(cind), 
			  "_a001_bb(p4_vb, p2_vb)                  = -2.0   * _a021_bb(p4_vb, p2_vb, cind) * _a007(cind)")
        (_a001_aa(p4_va, p2_va)                 += -1.0   * _a017_aa(p4_va, h2_oa, cind) * chol3d_aa_ov(h2_oa, p2_va, cind), 
			  "_a001_aa(p4_va, p2_va)                 += -1.0   * _a017_aa(p4_va, h2_oa, cind) * chol3d_aa_ov(h2_oa, p2_va, cind)")
        (_a001_bb(p4_vb, p2_vb)                 += -1.0   * _a017_bb(p4_vb, h2_ob, cind) * chol3d_bb_ov(h2_ob, p2_vb, cind), 
			  "_a001_bb(p4_vb, p2_vb)                 += -1.0   * _a017_bb(p4_vb, h2_ob, cind) * chol3d_bb_ov(h2_ob, p2_vb, cind)")
        (_a006_aa(h4_oa, h1_oa)                 +=  1.0   * _a009_aa(h4_oa, h1_oa, cind) * _a007(cind), 
			  "_a006_aa(h4_oa, h1_oa)                 +=  1.0   * _a009_aa(h4_oa, h1_oa, cind) * _a007(cind)")
        (_a006_bb(h4_ob, h1_ob)                 +=  1.0   * _a009_bb(h4_ob, h1_ob, cind) * _a007(cind), 
			  "_a006_bb(h4_ob, h1_ob)                 +=  1.0   * _a009_bb(h4_ob, h1_ob, cind) * _a007(cind)")
        (_a006_aa(h4_oa, h1_oa)                 += -1.0   * _a009_aa(h3_oa, h1_oa, cind) * _a008_aa(h4_oa, h3_oa, cind), 
			  "_a006_aa(h4_oa, h1_oa)                 += -1.0   * _a009_aa(h3_oa, h1_oa, cind) * _a008_aa(h4_oa, h3_oa, cind)")
        (_a006_bb(h4_ob, h1_ob)                 += -1.0   * _a009_bb(h3_ob, h1_ob, cind) * _a008_bb(h4_ob, h3_ob, cind), 
			  "_a006_bb(h4_ob, h1_ob)                 += -1.0   * _a009_bb(h3_ob, h1_ob, cind) * _a008_bb(h4_ob, h3_ob, cind)")
        (_a019_aaaa(h4_oa, h3_oa, h1_oa, h2_oa)  =  0.25  * _a009_aa(h4_oa, h1_oa, cind) * _a009_aa(h3_oa, h2_oa, cind), 
			  "_a019_aaaa(h4_oa, h3_oa, h1_oa, h2_oa)  =  0.25  * _a009_aa(h4_oa, h1_oa, cind) * _a009_aa(h3_oa, h2_oa, cind)") 
        (_a019_abab(h4_oa, h3_ob, h1_oa, h2_ob)  =  0.25  * _a009_aa(h4_oa, h1_oa, cind) * _a009_bb(h3_ob, h2_ob, cind), 
			  "_a019_abab(h4_oa, h3_ob, h1_oa, h2_ob)  =  0.25  * _a009_aa(h4_oa, h1_oa, cind) * _a009_bb(h3_ob, h2_ob, cind)")
        (_a019_bbbb(h4_ob, h3_ob, h1_ob, h2_ob)  =  0.25  * _a009_bb(h4_ob, h1_ob, cind) * _a009_bb(h3_ob, h2_ob, cind), 
			  "_a019_bbbb(h4_ob, h3_ob, h1_ob, h2_ob)  =  0.25  * _a009_bb(h4_ob, h1_ob, cind) * _a009_bb(h3_ob, h2_ob, cind)") 
        (_a020_aaaa(p4_va, h4_oa, p1_va, h1_oa)  = -2.0   * _a009_aa(h4_oa, h1_oa, cind) * _a021_aa(p4_va, p1_va, cind), 
			  "_a020_aaaa(p4_va, h4_oa, p1_va, h1_oa)  = -2.0   * _a009_aa(h4_oa, h1_oa, cind) * _a021_aa(p4_va, p1_va, cind)")
        (_a020_abab(p4_va, h4_ob, p1_va, h1_ob)  = -2.0   * _a009_bb(h4_ob, h1_ob, cind) * _a021_aa(p4_va, p1_va, cind), 
			  "_a020_abab(p4_va, h4_ob, p1_va, h1_ob)  = -2.0   * _a009_bb(h4_ob, h1_ob, cind) * _a021_aa(p4_va, p1_va, cind)")
        (_a020_baba(p4_vb, h4_oa, p1_vb, h1_oa)  = -2.0   * _a009_aa(h4_oa, h1_oa, cind) * _a021_bb(p4_vb, p1_vb, cind), 
			  "_a020_baba(p4_vb, h4_oa, p1_vb, h1_oa)  = -2.0   * _a009_aa(h4_oa, h1_oa, cind) * _a021_bb(p4_vb, p1_vb, cind)")
        (_a020_bbbb(p4_vb, h4_ob, p1_vb, h1_ob)  = -2.0   * _a009_bb(h4_ob, h1_ob, cind) * _a021_bb(p4_vb, p1_vb, cind), 
			  "_a020_bbbb(p4_vb, h4_ob, p1_vb, h1_ob)  = -2.0   * _a009_bb(h4_ob, h1_ob, cind) * _a021_bb(p4_vb, p1_vb, cind)")

        (_a017_aa(p3_va, h2_oa, cind)           +=  1.0   * t1_aa(p3_va, h3_oa) * chol3d_aa_oo(h3_oa, h2_oa, cind), 
			  "_a017_aa(p3_va, h2_oa, cind)           +=  1.0   * t1_aa(p3_va, h3_oa) * chol3d_aa_oo(h3_oa, h2_oa, cind)")
        (_a017_bb(p3_vb, h2_ob, cind)           +=  1.0   * t1_bb(p3_vb, h3_ob) * chol3d_bb_oo(h3_ob, h2_ob, cind), 
			  "_a017_bb(p3_vb, h2_ob, cind)           +=  1.0   * t1_bb(p3_vb, h3_ob) * chol3d_bb_oo(h3_ob, h2_ob, cind)")
        (_a017_aa(p3_va, h2_oa, cind)           += -1.0   * chol3d_aa_vo(p3_va, h2_oa, cind), 
			  "_a017_aa(p3_va, h2_oa, cind)           += -1.0   * chol3d_aa_vo(p3_va, h2_oa, cind)")
        (_a017_bb(p3_vb, h2_ob, cind)           += -1.0   * chol3d_bb_vo(p3_vb, h2_ob, cind), 
			  "_a017_bb(p3_vb, h2_ob, cind)           += -1.0   * chol3d_bb_vo(p3_vb, h2_ob, cind)")

        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)     =  0.5   * _a017_aa(p3_va, h1_oa, cind) * _a017_aa(p4_va, h2_oa, cind), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)     =  0.5   * _a017_aa(p3_va, h1_oa, cind) * _a017_aa(p4_va, h2_oa, cind)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)     =  0.5   * _a017_bb(p3_vb, h1_ob, cind) * _a017_bb(p4_vb, h2_ob, cind), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)     =  0.5   * _a017_bb(p3_vb, h1_ob, cind) * _a017_bb(p4_vb, h2_ob, cind)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)     =  1.0   * _a017_aa(p3_va, h1_oa, cind) * _a017_bb(p4_vb, h2_ob, cind), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)     =  1.0   * _a017_aa(p3_va, h1_oa, cind) * _a017_bb(p4_vb, h2_ob, cind)")

        (_a022_aaaa(p3_va,p4_va,p2_va,p1_va)     =  1.0   * _a021_aa(p3_va,p2_va,cind) * _a021_aa(p4_va,p1_va,cind), 
			  "_a022_aaaa(p3_va,p4_va,p2_va,p1_va)     =  1.0   * _a021_aa(p3_va,p2_va,cind) * _a021_aa(p4_va,p1_va,cind)")
        (_a022_abab(p3_va,p4_vb,p2_va,p1_vb)     =  1.0   * _a021_aa(p3_va,p2_va,cind) * _a021_bb(p4_vb,p1_vb,cind), 
			  "_a022_abab(p3_va,p4_vb,p2_va,p1_vb)     =  1.0   * _a021_aa(p3_va,p2_va,cind) * _a021_bb(p4_vb,p1_vb,cind)")
        (_a022_bbbb(p3_vb,p4_vb,p2_vb,p1_vb)     =  1.0   * _a021_bb(p3_vb,p2_vb,cind) * _a021_bb(p4_vb,p1_vb,cind), 
			  "_a022_bbbb(p3_vb,p4_vb,p2_vb,p1_vb)     =  1.0   * _a021_bb(p3_vb,p2_vb,cind) * _a021_bb(p4_vb,p1_vb,cind)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    +=  1.0   * _a022_aaaa(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    +=  1.0   * _a022_aaaa(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    +=  1.0   * _a022_bbbb(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    +=  1.0   * _a022_bbbb(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)    +=  4.0   * _a022_abab(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)    +=  4.0   * _a022_abab(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
        (_a019_aaaa(h4_oa, h3_oa, h1_oa, h2_oa) += -0.125 * _a004_aaaa(p1_va, p2_va, h3_oa, h4_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa), 
			  "_a019_aaaa(h4_oa, h3_oa, h1_oa, h2_oa) += -0.125 * _a004_aaaa(p1_va, p2_va, h3_oa, h4_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
        (_a019_abab(h4_oa, h3_ob, h1_oa, h2_ob) +=  0.25  * _a004_abab(p1_va, p2_vb, h4_oa, h3_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob), 
			  "_a019_abab(h4_oa, h3_ob, h1_oa, h2_ob) +=  0.25  * _a004_abab(p1_va, p2_vb, h4_oa, h3_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)") 
        (_a019_bbbb(h4_ob, h3_ob, h1_ob, h2_ob) += -0.125 * _a004_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob), 
			  "_a019_bbbb(h4_ob, h3_ob, h1_ob, h2_ob) += -0.125 * _a004_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    +=  1.0   * _a019_aaaa(h4_oa, h3_oa, h1_oa, h2_oa) * t2_aaaa(p3_va, p4_va, h4_oa, h3_oa), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    +=  1.0   * _a019_aaaa(h4_oa, h3_oa, h1_oa, h2_oa) * t2_aaaa(p3_va, p4_va, h4_oa, h3_oa)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    +=  1.0   * _a019_bbbb(h4_ob, h3_ob, h1_ob, h2_ob) * t2_bbbb(p3_vb, p4_vb, h4_ob, h3_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    +=  1.0   * _a019_bbbb(h4_ob, h3_ob, h1_ob, h2_ob) * t2_bbbb(p3_vb, p4_vb, h4_ob, h3_ob)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)    +=  4.0   * _a019_abab(h4_oa, h3_ob, h1_oa, h2_ob) * t2_abab(p3_va, p4_vb, h4_oa, h3_ob), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)    +=  4.0   * _a019_abab(h4_oa, h3_ob, h1_oa, h2_ob) * t2_abab(p3_va, p4_vb, h4_oa, h3_ob)")
        (_a020_aaaa(p1_va, h3_oa, p4_va, h2_oa) +=  0.5   * _a004_aaaa(p2_va, p4_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa), 
			  "_a020_aaaa(p1_va, h3_oa, p4_va, h2_oa) +=  0.5   * _a004_aaaa(p2_va, p4_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)") 
        (_a020_baab(p1_vb, h3_oa, p4_va, h2_ob)  = -0.5   * _a004_aaaa(p2_va, p4_va, h3_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob), 
			  "_a020_baab(p1_vb, h3_oa, p4_va, h2_ob)  = -0.5   * _a004_aaaa(p2_va, p4_va, h3_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)") 
        (_a020_abba(p1_va, h3_ob, p4_vb, h2_oa)  = -0.5   * _a004_bbbb(p2_vb, p4_vb, h3_ob, h1_ob) * t2_abab(p1_va,p2_vb,h2_oa,h1_ob), 
			  "_a020_abba(p1_va, h3_ob, p4_vb, h2_oa)  = -0.5   * _a004_bbbb(p2_vb, p4_vb, h3_ob, h1_ob) * t2_abab(p1_va,p2_vb,h2_oa,h1_ob)")
        (_a020_bbbb(p1_vb, h3_ob, p4_vb, h2_ob) +=  0.5   * _a004_bbbb(p2_vb, p4_vb, h3_ob, h1_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob), 
			  "_a020_bbbb(p1_vb, h3_ob, p4_vb, h2_ob) +=  0.5   * _a004_bbbb(p2_vb, p4_vb, h3_ob, h1_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob)")
        (_a020_baba(p1_vb, h7_oa, p6_vb, h2_oa) +=  1.0   * _a004_abab(p5_va, p6_vb, h7_oa, h8_ob) * t2_abab(p5_va,p1_vb,h2_oa,h8_ob), 
			  "_a020_baba(p1_vb, h7_oa, p6_vb, h2_oa) +=  1.0   * _a004_abab(p5_va, p6_vb, h7_oa, h8_ob) * t2_abab(p5_va,p1_vb,h2_oa,h8_ob)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    +=  1.0   * _a020_aaaa(p4_va, h4_oa, p1_va, h1_oa) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    +=  1.0   * _a020_aaaa(p4_va, h4_oa, p1_va, h1_oa) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    += -1.0   * _a020_abba(p4_va, h4_ob, p1_vb, h1_oa) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    += -1.0   * _a020_abba(p4_va, h4_ob, p1_vb, h1_oa) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    +=  1.0   * _a020_bbbb(p4_vb, h4_ob, p1_vb, h1_ob) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    +=  1.0   * _a020_bbbb(p4_vb, h4_ob, p1_vb, h1_ob) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    += -1.0   * _a020_baab(p4_vb, h4_oa, p1_va, h1_ob) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    += -1.0   * _a020_baab(p4_vb, h4_oa, p1_va, h1_ob) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob)")
        (i0_abab(p3_va, p1_vb, h2_oa, h4_ob)    +=  1.0   * _a020_baba(p1_vb, h7_oa, p6_vb, h2_oa) * t2_abab(p3_va, p6_vb, h7_oa, h4_ob), 
			  "i0_abab(p3_va, p1_vb, h2_oa, h4_ob)    +=  1.0   * _a020_baba(p1_vb, h7_oa, p6_vb, h2_oa) * t2_abab(p3_va, p6_vb, h7_oa, h4_ob)")
        (i0_abab(p3_va, p1_vb, h2_oa, h4_ob)    +=  1.0   * _a020_abab(p3_va, h8_ob, p5_va, h4_ob) * t2_abab(p5_va, p1_vb, h2_oa, h8_ob), 
			  "i0_abab(p3_va, p1_vb, h2_oa, h4_ob)    +=  1.0   * _a020_abab(p3_va, h8_ob, p5_va, h4_ob) * t2_abab(p5_va, p1_vb, h2_oa, h8_ob)")
        (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)    +=  1.0   * _a020_bbbb(p4_vb, h4_ob, p1_vb, h1_ob) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob), 
			  "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)    +=  1.0   * _a020_bbbb(p4_vb, h4_ob, p1_vb, h1_ob) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob)")
        (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)    += -1.0   * _a020_baab(p4_vb, h4_oa, p1_va, h1_ob) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa), 
			  "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)    += -1.0   * _a020_baab(p4_vb, h4_oa, p1_va, h1_ob) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa)")
        (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)    +=  1.0   * _a020_aaaa(p4_va, h4_oa, p1_va, h1_oa) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob), 
			  "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)    +=  1.0   * _a020_aaaa(p4_va, h4_oa, p1_va, h1_oa) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob)")
        (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)    += -1.0   * _a020_abba(p4_va, h4_ob, p1_vb, h1_oa) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob), 
			  "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)    += -1.0   * _a020_abba(p4_va, h4_ob, p1_vb, h1_oa) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob)")

        (_a001_aa(p4_va, p1_va)                 += -1.0   * f1_aa_vv(p4_va, p1_va), 
			  "_a001_aa(p4_va, p1_va)                 += -1.0   * f1_aa_vv(p4_va, p1_va)")
        (_a001_bb(p4_vb, p1_vb)                 += -1.0   * f1_bb_vv(p4_vb, p1_vb), 
			  "_a001_bb(p4_vb, p1_vb)                 += -1.0   * f1_bb_vv(p4_vb, p1_vb)")
        (_a006_aa(h9_oa, h1_oa)                 +=  1.0   * f1_aa_oo(h9_oa, h1_oa), 
			  "_a006_aa(h9_oa, h1_oa)                 +=  1.0   * f1_aa_oo(h9_oa, h1_oa)")
        (_a006_bb(h9_ob, h1_ob)                 +=  1.0   * f1_bb_oo(h9_ob, h1_ob), 
			  "_a006_bb(h9_ob, h1_ob)                 +=  1.0   * f1_bb_oo(h9_ob, h1_ob)")
        (_a006_aa(h9_oa, h1_oa)                 +=  1.0   * t1_aa(p8_va, h1_oa) * f1_aa_ov(h9_oa, p8_va), 
			  "_a006_aa(h9_oa, h1_oa)                 +=  1.0   * t1_aa(p8_va, h1_oa) * f1_aa_ov(h9_oa, p8_va)")
        (_a006_bb(h9_ob, h1_ob)                 +=  1.0   * t1_bb(p8_vb, h1_ob) * f1_bb_ov(h9_ob, p8_vb), 
			  "_a006_bb(h9_ob, h1_ob)                 +=  1.0   * t1_bb(p8_vb, h1_ob) * f1_bb_ov(h9_ob, p8_vb)")

        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    += -0.5   * t2_aaaa(p3_va, p2_va, h1_oa, h2_oa) * _a001_aa(p4_va, p2_va), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)    += -0.5   * t2_aaaa(p3_va, p2_va, h1_oa, h2_oa) * _a001_aa(p4_va, p2_va)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    += -0.5   * t2_bbbb(p3_vb, p2_vb, h1_ob, h2_ob) * _a001_bb(p4_vb, p2_vb), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)    += -0.5   * t2_bbbb(p3_vb, p2_vb, h1_ob, h2_ob) * _a001_bb(p4_vb, p2_vb)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)    += -1.0   * t2_abab(p3_va, p2_vb, h1_oa, h2_ob) * _a001_bb(p4_vb, p2_vb), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)    += -1.0   * t2_abab(p3_va, p2_vb, h1_oa, h2_ob) * _a001_bb(p4_vb, p2_vb)")
        (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)    += -1.0   * t2_abab(p2_va, p3_vb, h1_oa, h2_ob) * _a001_aa(p4_va, p2_va), 
			  "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)    += -1.0   * t2_abab(p2_va, p3_vb, h1_oa, h2_ob) * _a001_aa(p4_va, p2_va)")

        (i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)    += -0.5   * t2_aaaa(p3_va, p4_va, h3_oa, h1_oa) * _a006_aa(h3_oa, h2_oa), 
			  "i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)    += -0.5   * t2_aaaa(p3_va, p4_va, h3_oa, h1_oa) * _a006_aa(h3_oa, h2_oa)")
        (i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)    += -0.5   * t2_bbbb(p3_vb, p4_vb, h3_ob, h1_ob) * _a006_bb(h3_ob, h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)    += -0.5   * t2_bbbb(p3_vb, p4_vb, h3_ob, h1_ob) * _a006_bb(h3_ob, h2_ob)")
        (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)    += -1.0   * t2_abab(p3_va, p4_vb, h3_oa, h1_ob) * _a006_aa(h3_oa, h2_oa), 
			  "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)    += -1.0   * t2_abab(p3_va, p4_vb, h3_oa, h1_ob) * _a006_aa(h3_oa, h2_oa)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)    += -1.0   * t2_abab(p3_va, p4_vb, h1_oa, h3_ob) * _a006_bb(h3_ob, h2_ob), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)    += -1.0   * t2_abab(p3_va, p4_vb, h1_oa, h3_ob) * _a006_bb(h3_ob, h2_ob)")

        (i0_atmp(p3_va, p4_va, h1_oa, h2_oa)     =  1.0   * i0_aaaa(p3_va, p4_va, h1_oa, h2_oa), 
			  "i0_atmp(p3_va, p4_va, h1_oa, h2_oa)     =  1.0   * i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)")        
        (i0_atmp(p3_va, p4_va, h1_oa, h2_oa)    +=  1.0   * i0_aaaa(p4_va, p3_va, h2_oa, h1_oa), 
			  "i0_atmp(p3_va, p4_va, h1_oa, h2_oa)    +=  1.0   * i0_aaaa(p4_va, p3_va, h2_oa, h1_oa)")        
        (i0_atmp(p3_va, p4_va, h1_oa, h2_oa)    += -1.0   * i0_aaaa(p3_va, p4_va, h2_oa, h1_oa), 
			  "i0_atmp(p3_va, p4_va, h1_oa, h2_oa)    += -1.0   * i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)")        
        (i0_atmp(p3_va, p4_va, h1_oa, h2_oa)    += -1.0   * i0_aaaa(p4_va, p3_va, h1_oa, h2_oa), 
			  "i0_atmp(p3_va, p4_va, h1_oa, h2_oa)    += -1.0   * i0_aaaa(p4_va, p3_va, h1_oa, h2_oa)")
        (i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob)     =  1.0   * i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob), 
			  "i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob)     =  1.0   * i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)")
        (i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob)    +=  1.0   * i0_bbbb(p4_vb, p3_vb, h2_ob, h1_ob), 
			  "i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob)    +=  1.0   * i0_bbbb(p4_vb, p3_vb, h2_ob, h1_ob)")
        (i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob)    += -1.0   * i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob), 
			  "i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob)    += -1.0   * i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)") 
        (i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob)    += -1.0   * i0_bbbb(p4_vb, p3_vb, h1_ob, h2_ob), 
			  "i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob)    += -1.0   * i0_bbbb(p4_vb, p3_vb, h1_ob, h2_ob)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)     =  1.0   * i0_atmp(p3_va, p4_va, h1_oa, h2_oa), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)     =  1.0   * i0_atmp(p3_va, p4_va, h1_oa, h2_oa)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)     =  1.0   * i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)     =  1.0   * i0_btmp(p3_vb, p4_vb, h1_ob, h2_ob)")
        ;
        
}


template<typename T>
std::tuple<double,double> cd_ccsd_os_driver(SystemData sys_data, ExecutionContext& ec, 
                   const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, 
                   Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s, 
                   std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s, 
                   std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted,
                   Tensor<T>& cv3d, bool ccsd_restart=false, std::string out_fp="", bool computeTData=false) {

    int    maxiter     = sys_data.options_map.ccsd_options.ccsd_maxiter;
    int    ndiis       = sys_data.options_map.ccsd_options.ndiis;
    double thresh      = sys_data.options_map.ccsd_options.threshold;
    bool   writet      = sys_data.options_map.ccsd_options.writet;
    int    writet_iter = sys_data.options_map.ccsd_options.writet_iter;
    double zshiftl     = sys_data.options_map.ccsd_options.lshift;
    bool   profile     = sys_data.options_map.ccsd_options.profile_ccsd;    
    double residual    = 0.0;
    double energy      = 0.0;
    int    niter       = 0;

    const TAMM_SIZE n_occ_alpha = static_cast<TAMM_SIZE>(sys_data.n_occ_alpha);
    const TAMM_SIZE n_occ_beta  = static_cast<TAMM_SIZE>(sys_data.n_occ_beta);
    
    std::string t1file = out_fp+".t1amp";
    std::string t2file = out_fp+".t2amp";                       

    std::cout.precision(15);

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");  
    const int otiles         = O.num_tiles();
    const int vtiles         = V.num_tiles();
    const int oatiles        = MO("occ_alpha").num_tiles();
    const int obtiles        = MO("occ_beta").num_tiles();
    const int vatiles        = MO("virt_alpha").num_tiles();
    const int vbtiles        = MO("virt_beta").num_tiles();

    o_alpha = {MO("occ") ,range(oatiles)};
    v_alpha = {MO("virt"),range(vatiles)};
    o_beta  = {MO("occ") ,range(obtiles,otiles)};
    v_beta  = {MO("virt"),range(vbtiles,vtiles)};

    auto [cind]                       =      CI.labels<1>("all");
    auto [p1_va, p2_va, p3_va, p4_va] = v_alpha.labels<4>("all");
    auto [p1_vb, p2_vb, p3_vb, p4_vb] =  v_beta.labels<4>("all");
    auto [h1_oa, h2_oa, h3_oa, h4_oa] = o_alpha.labels<4>("all");
    auto [h1_ob, h2_ob, h3_ob, h4_ob] =  o_beta.labels<4>("all");           

    Tensor<T> d_e{};

    _a004_aaaa   = {{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    _a004_abab   = {{v_alpha,v_beta ,o_alpha,o_beta} ,{2,2}};
    _a004_bbbb   = {{v_beta ,v_beta ,o_beta ,o_beta} ,{2,2}};
    f1_aa_oo     = {{o_alpha,o_alpha}                ,{1,1}};
    f1_aa_ov     = {{o_alpha,v_alpha}                ,{1,1}};
    f1_aa_vo     = {{v_alpha,o_alpha}                ,{1,1}};
    f1_aa_vv     = {{v_alpha,v_alpha}                ,{1,1}};
    f1_bb_oo     = {{o_beta ,o_beta}                 ,{1,1}};
    f1_bb_ov     = {{o_beta ,v_beta}                 ,{1,1}};
    f1_bb_vo     = {{v_beta ,o_beta}                 ,{1,1}};
    f1_bb_vv     = {{v_beta ,v_beta}                 ,{1,1}};
    chol3d_aa_oo = {{o_alpha,o_alpha,CI}             ,{1,1}};
    chol3d_aa_ov = {{o_alpha,v_alpha,CI}             ,{1,1}};
    chol3d_aa_vo = {{v_alpha,o_alpha,CI}             ,{1,1}};
    chol3d_aa_vv = {{v_alpha,v_alpha,CI}             ,{1,1}};
    chol3d_bb_oo = {{o_beta ,o_beta ,CI}             ,{1,1}};
    chol3d_bb_ov = {{o_beta ,v_beta ,CI}             ,{1,1}};
    chol3d_bb_vo = {{v_beta ,o_beta ,CI}             ,{1,1}};
    chol3d_bb_vv = {{v_beta ,v_beta ,CI}             ,{1,1}};
    t1_aa        = {{v_alpha,o_alpha}                ,{1,1}}; 
    t1_bb        = {{v_beta ,o_beta}                 ,{1,1}};
    t2_aaaa      = {{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    t2_abab      = {{v_alpha,v_beta ,o_alpha,o_beta} ,{2,2}};
    t2_bbbb      = {{v_beta ,v_beta ,o_beta ,o_beta} ,{2,2}};
    r1_aa        = {{v_alpha,o_alpha}                ,{1,1}}; 
    r1_bb        = {{v_beta ,o_beta}                 ,{1,1}};
    r2_aaaa      = {{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    r2_abab      = {{v_alpha,v_beta ,o_alpha,o_beta} ,{2,2}};
    r2_bbbb      = {{v_beta ,v_beta ,o_beta ,o_beta} ,{2,2}};

    //Energy intermediates
    _a01    = {CI};
    _a02_aa = {{o_alpha,o_alpha,CI},{1,1}}; 
    _a02_bb = {{o_beta,o_beta,CI}  ,{1,1}};
    _a03_aa = {{o_alpha,v_alpha,CI},{1,1}}; 
    _a03_bb = {{o_beta,v_beta,CI}  ,{1,1}};
    //T1
    _a02       = {CI};
    _a01_aa    = {{o_alpha,o_alpha,CI}             ,{1,1}}; 
    _a01_bb    = {{o_beta,o_beta,CI}               ,{1,1}};
    _a03_aa_vo = {{v_alpha,o_alpha,CI}             ,{1,1}}; 
    _a03_bb_vo = {{v_beta,o_beta,CI}               ,{1,1}};
    _a04_aa    = {{o_alpha,o_alpha}                ,{1,1}};    
    _a04_bb    = {{o_beta,o_beta}                  ,{1,1}};
    _a05_aa    = {{o_alpha,v_alpha}                ,{1,1}};    
    _a05_bb    = {{o_beta,v_beta}                  ,{1,1}};
    //T2
    _a007      = {CI};
    _a017_aa   = {{v_alpha,o_alpha,CI}             ,{1,1}};
    _a017_bb   = {{v_beta,o_beta,CI}               ,{1,1}};
    _a006_aa   = {{o_alpha,o_alpha}                ,{1,1}};
    _a006_bb   = {{o_beta,o_beta}                  ,{1,1}};
    _a009_aa   = {{o_alpha,o_alpha,CI}             ,{1,1}};
    _a009_bb   = {{o_beta,o_beta,CI}               ,{1,1}};
    _a021_aa   = {{v_alpha,v_alpha,CI}             ,{1,1}};
    _a021_bb   = {{v_beta,v_beta,CI}               ,{1,1}};
    _a008_aa   = {{o_alpha,o_alpha,CI}             ,{1,1}};
    _a008_bb   = {{o_beta,o_beta,CI}               ,{1,1}};
    _a001_aa   = {{v_alpha,v_alpha}                ,{1,1}};
    _a001_bb   = {{v_beta,v_beta}                  ,{1,1}};
    _a019_aaaa = {{o_alpha,o_alpha,o_alpha,o_alpha},{2,2}};
    _a019_abab = {{o_alpha,o_beta,o_alpha,o_beta}  ,{2,2}};
    _a019_bbbb = {{o_beta,o_beta,o_beta,o_beta}    ,{2,2}};
    _a020_aaaa = {{v_alpha,o_alpha,v_alpha,o_alpha},{2,2}};
    _a020_abab = {{v_alpha,o_beta,v_alpha,o_beta}  ,{2,2}};
    _a020_baab = {{v_beta,o_alpha,v_alpha,o_beta}  ,{2,2}};
    _a020_abba = {{v_alpha,o_beta,v_beta,o_alpha}  ,{2,2}};
    _a020_baba = {{v_beta,o_alpha,v_beta,o_alpha}  ,{2,2}};
    _a020_bbbb = {{v_beta,o_beta,v_beta,o_beta}    ,{2,2}};
    _a022_aaaa = {{v_alpha,v_alpha,v_alpha,v_alpha},{2,2}};
    _a022_abab = {{v_alpha,v_beta,v_alpha,v_beta}  ,{2,2}};
    _a022_bbbb = {{v_beta,v_beta,v_beta,v_beta}    ,{2,2}};
    i0_atmp    = {{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    i0_btmp    = {{v_beta ,v_beta ,o_beta ,o_beta} ,{2,2}};

    double total_ccsd_mem = sum_tensor_sizes(d_t1,d_t2,d_f1,d_r1,d_r2,cv3d,
                d_e,_a01,_a02_aa,_a02_bb,_a03_aa,_a03_bb,
                t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, 
                r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb,
                f1_aa_oo, f1_aa_ov, f1_aa_vo, f1_aa_vv, 
                f1_bb_oo, f1_bb_ov, f1_bb_vo, f1_bb_vv,
                chol3d_aa_oo, chol3d_aa_ov, chol3d_aa_vo, chol3d_aa_vv,
                chol3d_bb_oo, chol3d_bb_ov, chol3d_bb_vo, chol3d_bb_vv,
                _a004_aaaa, _a004_abab, _a004_bbbb);

    for(size_t ri=0;ri<d_r1s.size();ri++)
        total_ccsd_mem += sum_tensor_sizes(d_r1s[ri],d_r2s[ri],d_t1s[ri],d_t2s[ri]);

    //Intermediates
    double total_ccsd_mem_tmp = sum_tensor_sizes
        (_a02,_a01_aa,_a01_bb,_a03_aa_vo,_a03_bb_vo,_a04_aa,_a04_bb,_a05_aa,_a05_bb,
              _a007,_a001_aa,_a001_bb,_a017_aa,_a017_bb,
              _a006_aa,_a006_bb,_a009_aa,_a009_bb,_a021_aa,_a021_bb,_a008_aa,
              _a008_bb,_a019_aaaa,_a019_abab,_a019_bbbb,_a020_aaaa,_a020_baba,
              _a020_abab,_a020_baab,_a020_bbbb,_a020_abba,_a022_aaaa,_a022_abab,
              _a022_bbbb,i0_atmp,i0_btmp);

    if(!ccsd_restart) total_ccsd_mem += total_ccsd_mem_tmp;

    if(ec.pg().rank()==0) {
        std::cout << "Total CPU memory required for Open Shell Cholesky CCSD calculation: " 
            << std::setprecision(5) << total_ccsd_mem << " GiB" << std::endl << std::endl;
    }

    Scheduler sch{ec};
    sch.allocate(d_e,_a01,_a02_aa,_a02_bb,_a03_aa,_a03_bb);
    sch.allocate(t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb, 
                 r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb,
                 f1_aa_oo, f1_aa_ov, f1_aa_vo, f1_aa_vv, 
                 f1_bb_oo, f1_bb_ov, f1_bb_vo, f1_bb_vv,
                 chol3d_aa_oo, chol3d_aa_ov, chol3d_aa_vo, chol3d_aa_vv,
                 chol3d_bb_oo, chol3d_bb_ov, chol3d_bb_vo, chol3d_bb_vv,
                 _a004_aaaa, _a004_abab, _a004_bbbb);

    sch
       (f1_aa_oo(h3_oa,h4_oa)           =  d_f1(h3_oa,h4_oa))
       (f1_aa_ov(h3_oa,p2_va)           =  d_f1(h3_oa,p2_va))
       (f1_aa_vo(p1_va,h4_oa)           =  d_f1(p1_va,h4_oa))
       (f1_aa_vv(p1_va,p2_va)           =  d_f1(p1_va,p2_va))
       (f1_bb_oo(h3_ob,h4_ob)           =  d_f1(h3_ob,h4_ob))
       (f1_bb_ov(h3_ob,p1_vb)           =  d_f1(h3_ob,p1_vb))
       (f1_bb_vo(p1_vb,h3_ob)           =  d_f1(p1_vb,h3_ob))
       (f1_bb_vv(p1_vb,p2_vb)           =  d_f1(p1_vb,p2_vb))
       (chol3d_aa_oo(h3_oa,h4_oa,cind)  =  cv3d(h3_oa,h4_oa,cind))
       (chol3d_aa_ov(h3_oa,p2_va,cind)  =  cv3d(h3_oa,p2_va,cind))
       (chol3d_aa_vo(p1_va,h4_oa,cind)  =  cv3d(p1_va,h4_oa,cind))
       (chol3d_aa_vv(p1_va,p2_va,cind)  =  cv3d(p1_va,p2_va,cind))
       (chol3d_bb_oo(h3_ob,h4_ob,cind)  =  cv3d(h3_ob,h4_ob,cind))
       (chol3d_bb_ov(h3_ob,p1_vb,cind)  =  cv3d(h3_ob,p1_vb,cind))
       (chol3d_bb_vo(p1_vb,h3_ob,cind)  =  cv3d(p1_vb,h3_ob,cind))
       (chol3d_bb_vv(p1_vb,p2_vb,cind)  =  cv3d(p1_vb,p2_vb,cind))
       ;

    if(!ccsd_restart) {

        //allocate all intermediates 
        sch.allocate(_a02, _a01_aa,_a01_bb,_a03_aa_vo,_a03_bb_vo,_a04_aa,_a04_bb,_a05_aa,_a05_bb,
                     _a007,_a001_aa,_a001_bb,_a017_aa,_a017_bb,
                     _a006_aa,_a006_bb,_a009_aa,_a009_bb,_a021_aa,_a021_bb,_a008_aa,
                     _a008_bb,_a019_aaaa,_a019_abab,_a019_bbbb,_a020_aaaa,_a020_baba,
                     _a020_abab,_a020_baab,_a020_bbbb,_a020_abba,_a022_aaaa,_a022_abab,
                     _a022_bbbb,i0_atmp,i0_btmp).execute();

        sch
           (_a004_aaaa(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_aa_vo(p1_va, h4_oa, cind) * chol3d_aa_vo(p2_va, h3_oa, cind))
           (_a004_abab(p1_va, p2_vb, h4_oa, h3_ob) = 1.0 * chol3d_aa_vo(p1_va, h4_oa, cind) * chol3d_bb_vo(p2_vb, h3_ob, cind))
           (_a004_bbbb(p1_vb, p2_vb, h4_ob, h3_ob) = 1.0 * chol3d_bb_vo(p1_vb, h4_ob, cind) * chol3d_bb_vo(p2_vb, h3_ob, cind));

        #ifdef USE_TALSH
          sch.execute(ExecutionHW::GPU);
        #else
          sch.execute();
        #endif

        Tensor<T> d_r1_residual{}, d_r2_residual{};
        Tensor<T>::allocate(&ec,d_r1_residual, d_r2_residual);

        for(int titer = 0; titer < maxiter; titer += ndiis) {
          for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {

            const auto timer_start = std::chrono::high_resolution_clock::now();

            niter   = iter;
            int off = iter - titer;
            
            sch
               ((d_t1s[off])()  = d_t1())
               ((d_t2s[off])()  = d_t2())
               .execute();

            //TODO:UPDATE FOR DIIS
            sch
               (t1_aa(p1_va,h3_oa)               = d_t1(p1_va,h3_oa))
               (t1_bb(p1_vb,h3_ob)               = d_t1(p1_vb,h3_ob))
               (t2_aaaa(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1_va,p2_va,h3_oa,h4_oa))
               (t2_abab(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1_va,p2_vb,h3_oa,h4_ob))
               (t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1_vb,p2_vb,h3_ob,h4_ob))
               .execute();

            ccsd_e_os( /* ec,  */sch, MO, CI, d_e);
            ccsd_t1_os(/* ec,  */sch, MO, CI, d_r1);
            ccsd_t2_os(/* ec,  */sch, MO, CI, d_r2);

            sch
              (d_r1(p2_va, h1_oa)                = r1_aa(p2_va, h1_oa))
              (d_r1(p2_vb, h1_ob)                = r1_bb(p2_vb, h1_ob))
              (d_r2(p3_va, p4_va, h2_oa, h1_oa)  = r2_aaaa(p3_va, p4_va, h2_oa, h1_oa))
              (d_r2(p3_vb, p4_vb, h2_ob, h1_ob)  = r2_bbbb(p3_vb, p4_vb, h2_ob, h1_ob))
              (d_r2(p3_va, p4_vb, h2_oa, h1_ob)  = r2_abab(p3_va, p4_vb, h2_oa, h1_ob))
              ;

            #ifdef USE_TALSH
              sch.execute(ExecutionHW::GPU, profile);
            #else
              sch.execute(ExecutionHW::CPU, profile);
            #endif

            std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_t1, d_t2,
                                              d_e, d_r1_residual, d_r2_residual, 
                                              p_evl_sorted, zshiftl, n_occ_alpha, n_occ_beta);

            update_r2(ec, d_r2());

            sch
               ((d_r1s[off])() = d_r1())
               ((d_r2s[off])() = d_r2())
                .execute();

            const auto timer_end = std::chrono::high_resolution_clock::now();
            auto iter_time = std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

            iteration_print(sys_data, ec.pg(), iter, residual, energy, iter_time);

            if(writet && ( ((iter+1)%writet_iter == 0) /*|| (residual < thresh)*/ ) ) {
                write_to_disk(d_t1,t1file);
                write_to_disk(d_t2,t2file);
            }

            if(residual < thresh) { 
                Tensor<T> t2_copy{{V,V,O,O},{2,2}};
                sch
                   .allocate(t2_copy)
                   (t2_copy()                     =  1.0 * d_t2())
                   (d_t2(p1_va,p2_vb,h4_ob,h3_oa) = -1.0 * t2_copy(p1_va,p2_vb,h3_oa,h4_ob))
                   (d_t2(p2_vb,p1_va,h3_oa,h4_ob) = -1.0 * t2_copy(p1_va,p2_vb,h3_oa,h4_ob))
                   (d_t2(p2_vb,p1_va,h4_ob,h3_oa) =  1.0 * t2_copy(p1_va,p2_vb,h3_oa,h4_ob))
                   .deallocate(t2_copy)
                   .execute();
                if(writet) {
                  write_to_disk(d_t1,t1file);
                  write_to_disk(d_t2,t2file);
                  if(computeTData && sys_data.options_map.ccsd_options.writev) {
                    fs::copy_file(t1file, out_fp+".fullT1amp", fs::copy_options::update_existing);
                    fs::copy_file(t2file, out_fp+".fullT2amp", fs::copy_options::update_existing);
                  }
                }
                break; 
            }
            
          }

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
        }

        if(profile) {
            std::string profile_csv = out_fp + "_profile.csv";
            std::ofstream pds(profile_csv, std::ios::out);
            if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
            std::string header = "ID;Level;OP;total_op_time_min;total_op_time_max;total_op_time_avg;";
            header += "get_time_min;get_time_max;get_time_avg;gemm_time_min;";
            header += "gemm_time_max;gemm_time_avg;acc_time_min;acc_time_max;acc_time_avg";
            pds << header << std::endl;
            pds << ec.get_profile_data().str() << std::endl;
            pds.close();
        }
        
        sch.deallocate(_a02, _a01_aa,_a01_bb,_a03_aa_vo,_a03_bb_vo,_a04_aa,_a04_bb,_a05_aa,_a05_bb); //t1
        sch.deallocate(_a007,_a001_aa,_a001_bb,_a017_aa,_a017_bb,
                       _a006_aa,_a006_bb,_a009_aa,_a009_bb,_a021_aa,_a021_bb,_a008_aa,
                       _a008_bb,_a019_aaaa,_a019_abab,_a019_bbbb,_a020_aaaa,_a020_baba,
                       _a020_abab,_a020_baab,_a020_bbbb,_a020_abba,_a022_aaaa,_a022_abab,_a022_bbbb,
                       i0_atmp,i0_btmp); //t2
        sch.deallocate(d_r1_residual, d_r2_residual);

    } //no restart
    else {
      sch
         (d_e()=0)
         (t1_aa(p1_va,h3_oa) = d_t1(p1_va,h3_oa))
         (t1_bb(p1_vb,h3_ob) = d_t1(p1_vb,h3_ob))
         (t2_aaaa(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1_va,p2_va,h3_oa,h4_oa))
         (t2_abab(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1_va,p2_vb,h3_oa,h4_ob))
         (t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1_vb,p2_vb,h3_ob,h4_ob));

      ccsd_e_os(/* ec,  */sch, MO, CI, d_e);

      #ifdef USE_TALSH
        sch.execute(ExecutionHW::GPU, profile);
      #else
        sch.execute(ExecutionHW::CPU, profile);
      #endif

      energy   = get_scalar(d_e);
      residual = 0.0;
    }

    sys_data.ccsd_corr_energy  = energy;

    if(ec.pg().rank() == 0) {
        sys_data.results["output"]["CCSD"]["n_iterations"] =   niter+1;
        sys_data.results["output"]["CCSD"]["final_energy"]["correlation"] =  energy;
        sys_data.results["output"]["CCSD"]["final_energy"]["total"] =  sys_data.scf_energy+energy;

        write_json_data(sys_data,"CCSD");
    }

    sch.deallocate(d_e,_a01,_a02_aa,_a02_bb,_a03_aa,_a03_bb,
                   _a004_aaaa,_a004_abab,_a004_bbbb,
                   t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb,   
                   r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb,
                   f1_aa_oo, f1_aa_ov, f1_aa_vo, f1_aa_vv, 
                   f1_bb_oo, f1_bb_ov, f1_bb_vo, f1_bb_vv,
                   chol3d_aa_oo, chol3d_aa_ov, chol3d_aa_vo, chol3d_aa_vv,
                   chol3d_bb_oo, chol3d_bb_ov, chol3d_bb_vo, chol3d_bb_vv)
        .execute();

    return std::make_tuple(residual,energy);
}
