#pragma once

#include "cd_ccsd_cs_ann.hpp"


template<typename T>
void ccsd_e_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, 
        Tensor<T>& de, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
        std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {

    auto [cind]                =      CI.labels<1>("all");
    auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
    auto [p1_vb, p2_vb, p3_vb] =  v_beta.labels<3>("all");
    auto [h3_oa, h4_oa, h6_oa] = o_alpha.labels<3>("all");
    auto [h3_ob, h4_ob, h6_ob] =  o_beta.labels<3>("all");

    Tensor<T> t1_aa = t1("aa");
    Tensor<T> t1_bb = t1("bb");
    Tensor<T> t2_aaaa = t2("aaaa");
    Tensor<T> t2_abab = t2("abab");
    Tensor<T> t2_bbbb = t2("bbbb");

    // f1_se{f1_oo,f1_ov,f1_vo,f1_vv}
    // chol3d_se{chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv}
    auto f1_ov = f1_se[1];
    auto chol3d_ov = chol3d_se[1];
  
    sch 
    (_a01V(cind)                     = t1_aa(p3_va, h4_oa) * chol3d_ov("aa")(h4_oa, p3_va, cind), 
		"_a01V(cind)                     = t1_aa(p3_va, h4_oa) * chol3d_ov( aa )(h4_oa, p3_va, cind)")
    (_a02("aa")(h4_oa, h6_oa, cind)  = t1_aa(p3_va, h4_oa) * chol3d_ov("aa")(h6_oa, p3_va, cind), 
		"_a02( aa )(h4_oa, h6_oa, cind)  = t1_aa(p3_va, h4_oa) * chol3d_ov( aa )(h6_oa, p3_va, cind)")
    (_a03("aa")(h4_oa, p2_va, cind)  = t2_aaaa(p1_va, p2_va, h3_oa, h4_oa) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
		"_a03( aa )(h4_oa, p2_va, cind)  = t2_aaaa(p1_va, p2_va, h3_oa, h4_oa) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
    (_a03("aa")(h4_oa, p2_va, cind) += t2_abab(p2_va, p1_vb, h4_oa, h3_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
		"_a03( aa )(h4_oa, p2_va, cind) += t2_abab(p2_va, p1_vb, h4_oa, h3_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
    (_a01V(cind)                    += t1_bb(p3_vb, h4_ob) * chol3d_ov("bb")(h4_ob, p3_vb, cind), 
		"_a01V(cind)                    += t1_bb(p3_vb, h4_ob) * chol3d_ov( bb )(h4_ob, p3_vb, cind)")
    (_a02("bb")(h4_ob, h6_ob, cind)  = t1_bb(p3_vb, h4_ob) * chol3d_ov("bb")(h6_ob, p3_vb, cind), 
		"_a02( bb )(h4_ob, h6_ob, cind)  = t1_bb(p3_vb, h4_ob) * chol3d_ov( bb )(h6_ob, p3_vb, cind)")
    (_a03("bb")(h4_ob, p2_vb, cind)  = t2_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
		"_a03( bb )(h4_ob, p2_vb, cind)  = t2_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
    (_a03("bb")(h4_ob, p2_vb, cind) += t2_abab(p1_va, p2_vb, h3_oa, h4_ob) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
		"_a03( bb )(h4_ob, p2_vb, cind) += t2_abab(p1_va, p2_vb, h3_oa, h4_ob) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
    (de()                            =  0.5 * _a01V() * _a01V(), 
		"de()                            =  0.5 * _a01V() * _a01V()")
    (de()                           += -0.5 * _a02("aa")(h4_oa, h6_oa, cind) * _a02("aa")(h6_oa, h4_oa, cind), 
		"de()                           += -0.5 * _a02( aa )(h4_oa, h6_oa, cind) * _a02( aa )(h6_oa, h4_oa, cind)")
    (de()                           += -0.5 * _a02("bb")(h4_ob, h6_ob, cind) * _a02("bb")(h6_ob, h4_ob, cind), 
		"de()                           += -0.5 * _a02( bb )(h4_ob, h6_ob, cind) * _a02( bb )(h6_ob, h4_ob, cind)")
    (de()                           +=  0.5 * _a03("aa")(h4_oa, p1_va, cind) * chol3d_ov("aa")(h4_oa, p1_va, cind), 
		"de()                           +=  0.5 * _a03( aa )(h4_oa, p1_va, cind) * chol3d_ov( aa )(h4_oa, p1_va, cind)")
    (de()                           +=  0.5 * _a03("bb")(h4_ob, p1_vb, cind) * chol3d_ov("bb")(h4_ob, p1_vb, cind), 
		"de()                           +=  0.5 * _a03( bb )(h4_ob, p1_vb, cind) * chol3d_ov( bb )(h4_ob, p1_vb, cind)")
    (de()                           +=  1.0 * t1_aa(p1_va, h3_oa) * f1_ov("aa")(h3_oa, p1_va),
    "de()                           +=  1.0 * t1_aa(p1_va, h3_oa) * f1_ov( aa )(h3_oa, p1_va)") // NEW TERM
    (de()                           +=  1.0 * t1_bb(p1_vb, h3_ob) * f1_ov("bb")(h3_ob, p1_vb),
    "de()                           +=  1.0 * t1_bb(p1_vb, h3_ob) * f1_ov( bb )(h3_ob, p1_vb)") // NEW TERM
    ;

}

template<typename T>
void ccsd_t1_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, 
        CCSE_Tensors<T>& r1_vo, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
        std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
    
    auto [cind]                       =      CI.labels<1>("all");
    auto [p2]                         =      MO.labels<1>("virt");
    auto [h1]                         =      MO.labels<1>("occ");
    auto [p1_va, p2_va, p3_va]        = v_alpha.labels<3>("all");
    auto [p1_vb, p2_vb, p3_vb]        =  v_beta.labels<3>("all");
    auto [h1_oa, h2_oa, h3_oa, h7_oa] = o_alpha.labels<4>("all");
    auto [h1_ob, h2_ob, h3_ob, h7_ob] =  o_beta.labels<4>("all");
    
    Tensor<T> i0_aa = r1_vo("aa");
    Tensor<T> i0_bb = r1_vo("bb");

    Tensor<T> t1_aa = t1("aa");
    Tensor<T> t1_bb = t1("bb");
    Tensor<T> t2_aaaa = t2("aaaa");
    Tensor<T> t2_abab = t2("abab");
    Tensor<T> t2_bbbb = t2("bbbb");

    // f1_se{f1_oo,f1_ov,f1_vo,f1_vv}
    // chol3d_se{chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv}
    auto f1_oo = f1_se[0];
    auto f1_ov = f1_se[1];
    auto f1_vo = f1_se[2];
    auto f1_vv = f1_se[3];
    auto chol3d_oo = chol3d_se[0];
    auto chol3d_ov = chol3d_se[1];
    auto chol3d_vo = chol3d_se[2];
    auto chol3d_vv = chol3d_se[3];

    sch
       (i0_aa(p2_va, h1_oa)             =  1.0 * f1_vo("aa")(p2_va, h1_oa), 
			 "i0_aa(p2_va, h1_oa)             =  1.0 * f1_vo( aa )(p2_va, h1_oa)")
       (i0_bb(p2_vb, h1_ob)             =  1.0 * f1_vo("bb")(p2_vb, h1_ob), 
			 "i0_bb(p2_vb, h1_ob)             =  1.0 * f1_vo( bb )(p2_vb, h1_ob)")
       (_a01("aa")(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind), 
			 "_a01( aa )(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")                 // ovm
       (_a01("bb")(h2_ob, h1_ob, cind)  =  1.0 * t1_bb(p1_vb, h1_ob) * chol3d_ov("bb")(h2_ob, p1_vb, cind), 
			 "_a01( bb )(h2_ob, h1_ob, cind)  =  1.0 * t1_bb(p1_vb, h1_ob) * chol3d_ov( bb )(h2_ob, p1_vb, cind)")                 // ovm
       (_a02V(cind)                     =  1.0 * t1_aa(p3_va, h3_oa) * chol3d_ov("aa")(h3_oa, p3_va, cind), 
			 "_a02V(cind)                     =  1.0 * t1_aa(p3_va, h3_oa) * chol3d_ov( aa )(h3_oa, p3_va, cind)")                 // ovm
       (_a02V(cind)                    +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_ov("bb")(h3_ob, p3_vb, cind), 
			 "_a02V(cind)                    +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_ov( bb )(h3_ob, p3_vb, cind)")                 // ovm
       (_a06("aa")(p1_va, h1_oa, cind)  =  1.0 * t2_aaaa(p1_va, p3_va, h2_oa, h1_oa) * chol3d_ov("aa")(h2_oa, p3_va, cind), 
			 "_a06( aa )(p1_va, h1_oa, cind)  =  1.0 * t2_aaaa(p1_va, p3_va, h2_oa, h1_oa) * chol3d_ov( aa )(h2_oa, p3_va, cind)") // o2v2m
       (_a06("aa")(p1_va, h1_oa, cind) += -1.0 * t2_abab(p1_va, p3_vb, h1_oa, h2_ob) * chol3d_ov("bb")(h2_ob, p3_vb, cind), 
			 "_a06( aa )(p1_va, h1_oa, cind) += -1.0 * t2_abab(p1_va, p3_vb, h1_oa, h2_ob) * chol3d_ov( bb )(h2_ob, p3_vb, cind)") // o2v2m
       (_a06("bb")(p1_vb, h1_ob, cind)  = -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h1_ob) * chol3d_ov("aa")(h2_oa, p3_va, cind), 
			 "_a06( bb )(p1_vb, h1_ob, cind)  = -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h1_ob) * chol3d_ov( aa )(h2_oa, p3_va, cind)") // o2v2m
       (_a06("bb")(p1_vb, h1_ob, cind) +=  1.0 * t2_bbbb(p1_vb, p3_vb, h2_ob, h1_ob) * chol3d_ov("bb")(h2_ob, p3_vb, cind), 
			 "_a06( bb )(p1_vb, h1_ob, cind) +=  1.0 * t2_bbbb(p1_vb, p3_vb, h2_ob, h1_ob) * chol3d_ov( bb )(h2_ob, p3_vb, cind)") // o2v2m
       (_a04("aa")(h2_oa, h1_oa)        = -1.0 * f1_oo("aa")(h2_oa, h1_oa), 
			 "_a04( aa )(h2_oa, h1_oa)        = -1.0 * f1_oo( aa )(h2_oa, h1_oa)") // MOVED TERM
       (_a04("bb")(h2_ob, h1_ob)        = -1.0 * f1_oo("bb")(h2_ob, h1_ob), 
			 "_a04( bb )(h2_ob, h1_ob)        = -1.0 * f1_oo( bb )(h2_ob, h1_ob)") // MOVED TERM
       (_a04("aa")(h2_oa, h1_oa)       +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind), 
			 "_a04( aa )(h2_oa, h1_oa)       +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // o2vm
       (_a04("bb")(h2_ob, h1_ob)       +=  1.0 * chol3d_ov("bb")(h2_ob, p1_vb, cind) * _a06("bb")(p1_vb, h1_ob, cind), 
			 "_a04( bb )(h2_ob, h1_ob)       +=  1.0 * chol3d_ov( bb )(h2_ob, p1_vb, cind) * _a06( bb )(p1_vb, h1_ob, cind)")   // o2vm
       (_a04("aa")(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va), 
			 "_a04( aa )(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
       (_a04("bb")(h2_ob, h1_ob)       += -1.0 * t1_bb(p1_vb, h1_ob) * f1_ov("bb")(h2_ob, p1_vb), 
			 "_a04( bb )(h2_ob, h1_ob)       += -1.0 * t1_bb(p1_vb, h1_ob) * f1_ov( bb )(h2_ob, p1_vb)") // NEW TERM
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04("aa")(h2_oa, h1_oa), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04( aa )(h2_oa, h1_oa)")                         // o2v
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p2_vb, h2_ob) * _a04("bb")(h2_ob, h1_ob), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p2_vb, h2_ob) * _a04( bb )(h2_ob, h1_ob)")                         // o2v
       (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_vo("aa")(p1_va, h2_oa, cind) * _a02V(cind), 
			 "i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_vo( aa )(p1_va, h2_oa, cind) * _a02V(cind)")                      // ovm
       (i0_bb(p1_vb, h2_ob)            +=  1.0 * chol3d_vo("bb")(p1_vb, h2_ob, cind) * _a02V(cind), 
			 "i0_bb(p1_vb, h2_ob)            +=  1.0 * chol3d_vo( bb )(p1_vb, h2_ob, cind) * _a02V(cind)")                      // ovm
       (_a05("aa")(h2_oa, p1_va)        = -1.0 * chol3d_ov("aa")(h3_oa, p1_va, cind) * _a01("aa")(h2_oa, h3_oa, cind), 
			 "_a05( aa )(h2_oa, p1_va)        = -1.0 * chol3d_ov( aa )(h3_oa, p1_va, cind) * _a01( aa )(h2_oa, h3_oa, cind)")   // o2vm
       (_a05("bb")(h2_ob, p1_vb)        = -1.0 * chol3d_ov("bb")(h3_ob, p1_vb, cind) * _a01("bb")(h2_ob, h3_ob, cind), 
			 "_a05( bb )(h2_ob, p1_vb)        = -1.0 * chol3d_ov( bb )(h3_ob, p1_vb, cind) * _a01( bb )(h2_ob, h3_ob, cind)")   // o2vm
       (_a05("aa")(h2_oa, p1_va)       +=  1.0 * f1_ov("aa")(h2_oa, p1_va), 
			 "_a05( aa )(h2_oa, p1_va)       +=  1.0 * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
       (_a05("bb")(h2_ob, p1_vb)       +=  1.0 * f1_ov("bb")(h2_ob, p1_vb), 
			 "_a05( bb )(h2_ob, p1_vb)       +=  1.0 * f1_ov( bb )(h2_ob, p1_vb)") // NEW TERM
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t2_aaaa(p1_va, p2_va, h2_oa, h1_oa) * _a05("aa")(h2_oa, p1_va), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * t2_aaaa(p1_va, p2_va, h2_oa, h1_oa) * _a05( aa )(h2_oa, p1_va)")         // o2v
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_abab(p1_va, p2_vb, h2_oa, h1_ob) * _a05("aa")(h2_oa, p1_va), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_abab(p1_va, p2_vb, h2_oa, h1_ob) * _a05( aa )(h2_oa, p1_va)")         // o2v
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a05("bb")(h2_ob, p1_vb), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a05( bb )(h2_ob, p1_vb)")         // o2v
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_bbbb(p1_vb, p2_vb, h2_ob, h1_ob) * _a05("bb")(h2_ob, p1_vb), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_bbbb(p1_vb, p2_vb, h2_ob, h1_ob) * _a05( bb )(h2_ob, p1_vb)")         // o2v
       (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv("aa")(p2_va, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind), 
			 "i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv( aa )(p2_va, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // ov2m
       (i0_bb(p2_vb, h1_ob)            += -1.0 * chol3d_vv("bb")(p2_vb, p1_vb, cind) * _a06("bb")(p1_vb, h1_ob, cind), 
			 "i0_bb(p2_vb, h1_ob)            += -1.0 * chol3d_vv( bb )(p2_vb, p1_vb, cind) * _a06( bb )(p1_vb, h1_ob, cind)")   // ov2m
       (_a06("aa")(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv("aa")(p2_va, p1_va, cind), 
			 "_a06( aa )(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv( aa )(p2_va, p1_va, cind)")              // ov2m
       (_a06("bb")(p2_vb, h2_ob, cind) += -1.0 * t1_bb(p1_vb, h2_ob) * chol3d_vv("bb")(p2_vb, p1_vb, cind), 
			 "_a06( bb )(p2_vb, h2_ob, cind) += -1.0 * t1_bb(p1_vb, h2_ob) * chol3d_vv( bb )(p2_vb, p1_vb, cind)")              // ov2m
       (i0_aa(p1_va, h2_oa)            += -1.0 * _a06("aa")(p1_va, h2_oa, cind) * _a02V(cind), 
			 "i0_aa(p1_va, h2_oa)            += -1.0 * _a06( aa )(p1_va, h2_oa, cind) * _a02V(cind)")                           // ovm
       (i0_bb(p1_vb, h2_ob)            += -1.0 * _a06("bb")(p1_vb, h2_ob, cind) * _a02V(cind), 
			 "i0_bb(p1_vb, h2_ob)            += -1.0 * _a06( bb )(p1_vb, h2_ob, cind) * _a02V(cind)")                           // ovm
       (_a06("aa")(p2_va, h3_oa, cind) += -1.0 * t1_aa(p2_va, h3_oa) * _a02V(cind), 
			 "_a06( aa )(p2_va, h3_oa, cind) += -1.0 * t1_aa(p2_va, h3_oa) * _a02V(cind)")                                      // ovm
       (_a06("bb")(p2_vb, h3_ob, cind) += -1.0 * t1_bb(p2_vb, h3_ob) * _a02V(cind), 
			 "_a06( bb )(p2_vb, h3_ob, cind) += -1.0 * t1_bb(p2_vb, h3_ob) * _a02V(cind)")                                      // ovm
       (_a06("aa")(p2_va, h3_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01("aa")(h2_oa, h3_oa, cind), 
			 "_a06( aa )(p2_va, h3_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01( aa )(h2_oa, h3_oa, cind)")                   // o2vm
       (_a06("bb")(p2_vb, h3_ob, cind) +=  1.0 * t1_bb(p2_vb, h2_ob) * _a01("bb")(h2_ob, h3_ob, cind), 
			 "_a06( bb )(p2_vb, h3_ob, cind) +=  1.0 * t1_bb(p2_vb, h2_ob) * _a01( bb )(h2_ob, h3_ob, cind)")                   // o2vm
       (_a01("aa")(h3_oa, h1_oa, cind) +=  1.0 * chol3d_oo("aa")(h3_oa, h1_oa, cind), 
			 "_a01( aa )(h3_oa, h1_oa, cind) +=  1.0 * chol3d_oo( aa )(h3_oa, h1_oa, cind)")                                    // o2m
       (_a01("bb")(h3_ob, h1_ob, cind) +=  1.0 * chol3d_oo("bb")(h3_ob, h1_ob, cind), 
			 "_a01( bb )(h3_ob, h1_ob, cind) +=  1.0 * chol3d_oo( bb )(h3_ob, h1_ob, cind)")                                    // o2m        
       (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01("aa")(h3_oa, h1_oa, cind) * _a06("aa")(p2_va, h3_oa, cind), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * _a01( aa )(h3_oa, h1_oa, cind) * _a06( aa )(p2_va, h3_oa, cind)")        // o2vm
      //  (i0_aa(p2_va, h1_oa)         += -1.0 * t1_aa(p2_va, h7_oa) * f1_oo("aa")(h7_oa, h1_oa), 
			//  "i0_aa(p2_va, h1_oa)         += -1.0 * t1_aa(p2_va, h7_oa) * f1_oo( aa )(h7_oa, h1_oa)") // MOVED ABOVE         // o2v
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p3_va, h1_oa) * f1_vv("aa")(p2_va, p3_va), 
			 "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p3_va, h1_oa) * f1_vv( aa )(p2_va, p3_va)")                        // ov2
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * _a01("bb")(h3_ob, h1_ob, cind) * _a06("bb")(p2_vb, h3_ob, cind), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * _a01( bb )(h3_ob, h1_ob, cind) * _a06( bb )(p2_vb, h3_ob, cind)")        // o2vm
      //  (i0_bb(p2_vb, h1_ob)         += -1.0 * t1_bb(p2_vb, h7_ob) * f1_oo("bb")(h7_ob, h1_ob), 
			//  "i0_bb(p2_vb, h1_ob)         += -1.0 * t1_bb(p2_vb, h7_ob) * f1_oo( bb )(h7_ob, h1_ob)") // MOVED ABOVE         // o2v
       (i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p3_vb, h1_ob) * f1_vv("bb")(p2_vb, p3_vb), 
			 "i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p3_vb, h1_ob) * f1_vv( bb )(p2_vb, p3_vb)")                        // ov2
       ;

}

template<typename T>
void ccsd_t2_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,  
        CCSE_Tensors<T>& r2, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
        std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se,
        CCSE_Tensors<T>& i0tmp) {
                
    auto [cind]                                     =      CI.labels<1>("all");
    auto [p3, p4]                                   =      MO.labels<2>("virt");
    auto [h1, h2]                                   =      MO.labels<2>("occ");
    auto [p1_va, p2_va, p3_va, p4_va, p5_va, p8_va] = v_alpha.labels<6>("all");
    auto [p1_vb, p2_vb, p3_vb, p4_vb, p6_vb, p8_vb] =  v_beta.labels<6>("all");
    auto [h1_oa, h2_oa, h3_oa, h4_oa, h7_oa, h9_oa] = o_alpha.labels<6>("all");
    auto [h1_ob, h2_ob, h3_ob, h4_ob, h8_ob, h9_ob] =  o_beta.labels<6>("all");                

    Tensor<T> i0_aaaa = r2("aaaa");
    Tensor<T> i0_abab = r2("abab");
    Tensor<T> i0_bbbb = r2("bbbb");

    Tensor<T> t1_aa = t1("aa");
    Tensor<T> t1_bb = t1("bb");
    Tensor<T> t2_aaaa = t2("aaaa");
    Tensor<T> t2_abab = t2("abab");
    Tensor<T> t2_bbbb = t2("bbbb");

    // f1_se{f1_oo,f1_ov,f1_vo,f1_vv}
    // chol3d_se{chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv}
    auto f1_oo = f1_se[0];
    auto f1_ov = f1_se[1];
    auto f1_vo = f1_se[2];
    auto f1_vv = f1_se[3];
    auto chol3d_oo = chol3d_se[0];
    auto chol3d_ov = chol3d_se[1];
    auto chol3d_vo = chol3d_se[2];
    auto chol3d_vv = chol3d_se[3];

    sch 
        (_a017("aa")(p3_va, h2_oa, cind)            = -1.0   * t2_aaaa(p1_va, p3_va, h3_oa, h2_oa) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
			  "_a017( aa )(p3_va, h2_oa, cind)            = -1.0   * t2_aaaa(p1_va, p3_va, h3_oa, h2_oa) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
        (_a017("bb")(p3_vb, h2_ob, cind)            = -1.0   * t2_bbbb(p1_vb, p3_vb, h3_ob, h2_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
			  "_a017( bb )(p3_vb, h2_ob, cind)            = -1.0   * t2_bbbb(p1_vb, p3_vb, h3_ob, h2_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
        (_a017("bb")(p3_vb, h2_ob, cind)           += -1.0   * t2_abab(p1_va, p3_vb, h3_oa, h2_ob) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
			  "_a017( bb )(p3_vb, h2_ob, cind)           += -1.0   * t2_abab(p1_va, p3_vb, h3_oa, h2_ob) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
        (_a017("aa")(p3_va, h2_oa, cind)           += -1.0   * t2_abab(p3_va, p1_vb, h2_oa, h3_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
			  "_a017( aa )(p3_va, h2_oa, cind)           += -1.0   * t2_abab(p3_va, p1_vb, h2_oa, h3_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
        (_a006("aa")(h4_oa, h1_oa)                  = -1.0   * chol3d_ov("aa")(h4_oa, p2_va, cind) * _a017("aa")(p2_va, h1_oa, cind), 
			  "_a006( aa )(h4_oa, h1_oa)                  = -1.0   * chol3d_ov( aa )(h4_oa, p2_va, cind) * _a017( aa )(p2_va, h1_oa, cind)")
        (_a006("bb")(h4_ob, h1_ob)                  = -1.0   * chol3d_ov("bb")(h4_ob, p2_vb, cind) * _a017("bb")(p2_vb, h1_ob, cind), 
			  "_a006( bb )(h4_ob, h1_ob)                  = -1.0   * chol3d_ov( bb )(h4_ob, p2_vb, cind) * _a017( bb )(p2_vb, h1_ob, cind)")
        (_a007V(cind)                               =  1.0   * chol3d_ov("aa")(h4_oa, p1_va, cind) * t1_aa(p1_va, h4_oa), 
			  "_a007V(cind)                               =  1.0   * chol3d_ov( aa )(h4_oa, p1_va, cind) * t1_aa(p1_va, h4_oa)")
        (_a007V(cind)                              +=  1.0   * chol3d_ov("bb")(h4_ob, p1_vb, cind) * t1_bb(p1_vb, h4_ob), 
			  "_a007V(cind)                              +=  1.0   * chol3d_ov( bb )(h4_ob, p1_vb, cind) * t1_bb(p1_vb, h4_ob)")
        (_a009("aa")(h3_oa, h2_oa, cind)            =  1.0   * chol3d_ov("aa")(h3_oa, p1_va, cind) * t1_aa(p1_va, h2_oa), 
			  "_a009( aa )(h3_oa, h2_oa, cind)            =  1.0   * chol3d_ov( aa )(h3_oa, p1_va, cind) * t1_aa(p1_va, h2_oa)")
        (_a009("bb")(h3_ob, h2_ob, cind)            =  1.0   * chol3d_ov("bb")(h3_ob, p1_vb, cind) * t1_bb(p1_vb, h2_ob), 
			  "_a009( bb )(h3_ob, h2_ob, cind)            =  1.0   * chol3d_ov( bb )(h3_ob, p1_vb, cind) * t1_bb(p1_vb, h2_ob)")
        (_a021("aa")(p3_va, p1_va, cind)            = -0.5   * chol3d_ov("aa")(h3_oa, p1_va, cind) * t1_aa(p3_va, h3_oa), 
			  "_a021( aa )(p3_va, p1_va, cind)            = -0.5   * chol3d_ov( aa )(h3_oa, p1_va, cind) * t1_aa(p3_va, h3_oa)")
        (_a021("bb")(p3_vb, p1_vb, cind)            = -0.5   * chol3d_ov("bb")(h3_ob, p1_vb, cind) * t1_bb(p3_vb, h3_ob), 
			  "_a021( bb )(p3_vb, p1_vb, cind)            = -0.5   * chol3d_ov( bb )(h3_ob, p1_vb, cind) * t1_bb(p3_vb, h3_ob)")
        (_a021("aa")(p3_va, p1_va, cind)           +=  0.5   * chol3d_vv("aa")(p3_va, p1_va, cind), 
			  "_a021( aa )(p3_va, p1_va, cind)           +=  0.5   * chol3d_vv( aa )(p3_va, p1_va, cind)")
        (_a021("bb")(p3_vb, p1_vb, cind)           +=  0.5   * chol3d_vv("bb")(p3_vb, p1_vb, cind), 
			  "_a021( bb )(p3_vb, p1_vb, cind)           +=  0.5   * chol3d_vv( bb )(p3_vb, p1_vb, cind)")
        (_a017("aa")(p3_va, h2_oa, cind)           += -2.0   * t1_aa(p2_va, h2_oa) * _a021("aa")(p3_va, p2_va, cind), 
			  "_a017( aa )(p3_va, h2_oa, cind)           += -2.0   * t1_aa(p2_va, h2_oa) * _a021( aa )(p3_va, p2_va, cind)")
        (_a017("bb")(p3_vb, h2_ob, cind)           += -2.0   * t1_bb(p2_vb, h2_ob) * _a021("bb")(p3_vb, p2_vb, cind), 
			  "_a017( bb )(p3_vb, h2_ob, cind)           += -2.0   * t1_bb(p2_vb, h2_ob) * _a021( bb )(p3_vb, p2_vb, cind)")
        (_a008("aa")(h3_oa, h1_oa, cind)            =  1.0   * _a009("aa")(h3_oa, h1_oa, cind), 
			  "_a008( aa )(h3_oa, h1_oa, cind)            =  1.0   * _a009( aa )(h3_oa, h1_oa, cind)")
        (_a008("bb")(h3_ob, h1_ob, cind)            =  1.0   * _a009("bb")(h3_ob, h1_ob, cind), 
			  "_a008( bb )(h3_ob, h1_ob, cind)            =  1.0   * _a009( bb )(h3_ob, h1_ob, cind)")
        (_a009("aa")(h3_oa, h1_oa, cind)           +=  1.0   * chol3d_oo("aa")(h3_oa, h1_oa, cind), 
			  "_a009( aa )(h3_oa, h1_oa, cind)           +=  1.0   * chol3d_oo( aa )(h3_oa, h1_oa, cind)")
        (_a009("bb")(h3_ob, h1_ob, cind)           +=  1.0   * chol3d_oo("bb")(h3_ob, h1_ob, cind), 
			  "_a009( bb )(h3_ob, h1_ob, cind)           +=  1.0   * chol3d_oo( bb )(h3_ob, h1_ob, cind)")

        (_a001("aa")(p4_va, p2_va)                  = -2.0   * _a021("aa")(p4_va, p2_va, cind) * _a007V(cind), 
			  "_a001( aa )(p4_va, p2_va)                  = -2.0   * _a021( aa )(p4_va, p2_va, cind) * _a007V(cind)")
        (_a001("bb")(p4_vb, p2_vb)                  = -2.0   * _a021("bb")(p4_vb, p2_vb, cind) * _a007V(cind), 
			  "_a001( bb )(p4_vb, p2_vb)                  = -2.0   * _a021( bb )(p4_vb, p2_vb, cind) * _a007V(cind)")
        (_a001("aa")(p4_va, p2_va)                 += -1.0   * _a017("aa")(p4_va, h2_oa, cind) * chol3d_ov("aa")(h2_oa, p2_va, cind), 
			  "_a001( aa )(p4_va, p2_va)                 += -1.0   * _a017( aa )(p4_va, h2_oa, cind) * chol3d_ov( aa )(h2_oa, p2_va, cind)")
        (_a001("bb")(p4_vb, p2_vb)                 += -1.0   * _a017("bb")(p4_vb, h2_ob, cind) * chol3d_ov("bb")(h2_ob, p2_vb, cind), 
			  "_a001( bb )(p4_vb, p2_vb)                 += -1.0   * _a017( bb )(p4_vb, h2_ob, cind) * chol3d_ov( bb )(h2_ob, p2_vb, cind)")
        (_a006("aa")(h4_oa, h1_oa)                 +=  1.0   * _a009("aa")(h4_oa, h1_oa, cind) * _a007V(cind), 
			  "_a006( aa )(h4_oa, h1_oa)                 +=  1.0   * _a009( aa )(h4_oa, h1_oa, cind) * _a007V(cind)")
        (_a006("bb")(h4_ob, h1_ob)                 +=  1.0   * _a009("bb")(h4_ob, h1_ob, cind) * _a007V(cind), 
			  "_a006( bb )(h4_ob, h1_ob)                 +=  1.0   * _a009( bb )(h4_ob, h1_ob, cind) * _a007V(cind)")
        (_a006("aa")(h4_oa, h1_oa)                 += -1.0   * _a009("aa")(h3_oa, h1_oa, cind) * _a008("aa")(h4_oa, h3_oa, cind), 
			  "_a006( aa )(h4_oa, h1_oa)                 += -1.0   * _a009( aa )(h3_oa, h1_oa, cind) * _a008( aa )(h4_oa, h3_oa, cind)")
        (_a006("bb")(h4_ob, h1_ob)                 += -1.0   * _a009("bb")(h3_ob, h1_ob, cind) * _a008("bb")(h4_ob, h3_ob, cind), 
			  "_a006( bb )(h4_ob, h1_ob)                 += -1.0   * _a009( bb )(h3_ob, h1_ob, cind) * _a008( bb )(h4_ob, h3_ob, cind)")
        (_a019("aaaa")(h4_oa, h3_oa, h1_oa, h2_oa)  =  0.25  * _a009("aa")(h4_oa, h1_oa, cind) * _a009("aa")(h3_oa, h2_oa, cind), 
			  "_a019( aaaa )(h4_oa, h3_oa, h1_oa, h2_oa)  =  0.25  * _a009( aa )(h4_oa, h1_oa, cind) * _a009( aa )(h3_oa, h2_oa, cind)") 
        (_a019("abab")(h4_oa, h3_ob, h1_oa, h2_ob)  =  0.25  * _a009("aa")(h4_oa, h1_oa, cind) * _a009("bb")(h3_ob, h2_ob, cind), 
			  "_a019( abab )(h4_oa, h3_ob, h1_oa, h2_ob)  =  0.25  * _a009( aa )(h4_oa, h1_oa, cind) * _a009( bb )(h3_ob, h2_ob, cind)")
        (_a019("bbbb")(h4_ob, h3_ob, h1_ob, h2_ob)  =  0.25  * _a009("bb")(h4_ob, h1_ob, cind) * _a009("bb")(h3_ob, h2_ob, cind), 
			  "_a019( bbbb )(h4_ob, h3_ob, h1_ob, h2_ob)  =  0.25  * _a009( bb )(h4_ob, h1_ob, cind) * _a009( bb )(h3_ob, h2_ob, cind)") 
        (_a020("aaaa")(p4_va, h4_oa, p1_va, h1_oa)  = -2.0   * _a009("aa")(h4_oa, h1_oa, cind) * _a021("aa")(p4_va, p1_va, cind), 
			  "_a020( aaaa )(p4_va, h4_oa, p1_va, h1_oa)  = -2.0   * _a009( aa )(h4_oa, h1_oa, cind) * _a021( aa )(p4_va, p1_va, cind)")
        (_a020("abab")(p4_va, h4_ob, p1_va, h1_ob)  = -2.0   * _a009("bb")(h4_ob, h1_ob, cind) * _a021("aa")(p4_va, p1_va, cind), 
			  "_a020( abab )(p4_va, h4_ob, p1_va, h1_ob)  = -2.0   * _a009( bb )(h4_ob, h1_ob, cind) * _a021( aa )(p4_va, p1_va, cind)")
        (_a020("baba")(p4_vb, h4_oa, p1_vb, h1_oa)  = -2.0   * _a009("aa")(h4_oa, h1_oa, cind) * _a021("bb")(p4_vb, p1_vb, cind), 
			  "_a020( baba )(p4_vb, h4_oa, p1_vb, h1_oa)  = -2.0   * _a009( aa )(h4_oa, h1_oa, cind) * _a021( bb )(p4_vb, p1_vb, cind)")
        (_a020("bbbb")(p4_vb, h4_ob, p1_vb, h1_ob)  = -2.0   * _a009("bb")(h4_ob, h1_ob, cind) * _a021("bb")(p4_vb, p1_vb, cind), 
			  "_a020( bbbb )(p4_vb, h4_ob, p1_vb, h1_ob)  = -2.0   * _a009( bb )(h4_ob, h1_ob, cind) * _a021( bb )(p4_vb, p1_vb, cind)")

        (_a017("aa")(p3_va, h2_oa, cind)           +=  1.0   * t1_aa(p3_va, h3_oa) * chol3d_oo("aa")(h3_oa, h2_oa, cind), 
			  "_a017( aa )(p3_va, h2_oa, cind)           +=  1.0   * t1_aa(p3_va, h3_oa) * chol3d_oo( aa )(h3_oa, h2_oa, cind)")
        (_a017("bb")(p3_vb, h2_ob, cind)           +=  1.0   * t1_bb(p3_vb, h3_ob) * chol3d_oo("bb")(h3_ob, h2_ob, cind), 
			  "_a017( bb )(p3_vb, h2_ob, cind)           +=  1.0   * t1_bb(p3_vb, h3_ob) * chol3d_oo( bb )(h3_ob, h2_ob, cind)")
        (_a017("aa")(p3_va, h2_oa, cind)           += -1.0   * chol3d_vo("aa")(p3_va, h2_oa, cind), 
			  "_a017( aa )(p3_va, h2_oa, cind)           += -1.0   * chol3d_vo( aa )(p3_va, h2_oa, cind)")
        (_a017("bb")(p3_vb, h2_ob, cind)           += -1.0   * chol3d_vo("bb")(p3_vb, h2_ob, cind), 
			  "_a017( bb )(p3_vb, h2_ob, cind)           += -1.0   * chol3d_vo( bb )(p3_vb, h2_ob, cind)")

        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  0.5   * _a017("aa")(p3_va, h1_oa, cind) * _a017("aa")(p4_va, h2_oa, cind), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  0.5   * _a017( aa )(p3_va, h1_oa, cind) * _a017( aa )(p4_va, h2_oa, cind)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  0.5   * _a017("bb")(p3_vb, h1_ob, cind) * _a017("bb")(p4_vb, h2_ob, cind), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  0.5   * _a017( bb )(p3_vb, h1_ob, cind) * _a017( bb )(p4_vb, h2_ob, cind)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)        =  1.0   * _a017("aa")(p3_va, h1_oa, cind) * _a017("bb")(p4_vb, h2_ob, cind), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)        =  1.0   * _a017( aa )(p3_va, h1_oa, cind) * _a017( bb )(p4_vb, h2_ob, cind)")

        (_a022("aaaa")(p3_va,p4_va,p2_va,p1_va)     =  1.0   * _a021("aa")(p3_va,p2_va,cind) * _a021("aa")(p4_va,p1_va,cind), 
			  "_a022( aaaa )(p3_va,p4_va,p2_va,p1_va)     =  1.0   * _a021( aa )(p3_va,p2_va,cind) * _a021( aa )(p4_va,p1_va,cind)")
        (_a022("abab")(p3_va,p4_vb,p2_va,p1_vb)     =  1.0   * _a021("aa")(p3_va,p2_va,cind) * _a021("bb")(p4_vb,p1_vb,cind), 
			  "_a022( abab )(p3_va,p4_vb,p2_va,p1_vb)     =  1.0   * _a021( aa )(p3_va,p2_va,cind) * _a021( bb )(p4_vb,p1_vb,cind)")
        (_a022("bbbb")(p3_vb,p4_vb,p2_vb,p1_vb)     =  1.0   * _a021("bb")(p3_vb,p2_vb,cind) * _a021("bb")(p4_vb,p1_vb,cind), 
			  "_a022( bbbb )(p3_vb,p4_vb,p2_vb,p1_vb)     =  1.0   * _a021( bb )(p3_vb,p2_vb,cind) * _a021( bb )(p4_vb,p1_vb,cind)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a022("aaaa")(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a022( aaaa )(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a022("bbbb")(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a022( bbbb )(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a022("abab")(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a022( abab )(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
        (_a019("aaaa")(h4_oa, h3_oa, h1_oa, h2_oa) += -0.125 * _a004("aaaa")(p1_va, p2_va, h3_oa, h4_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa), 
			  "_a019( aaaa )(h4_oa, h3_oa, h1_oa, h2_oa) += -0.125 * _a004( aaaa )(p1_va, p2_va, h3_oa, h4_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
        (_a019("abab")(h4_oa, h3_ob, h1_oa, h2_ob) +=  0.25  * _a004("abab")(p1_va, p2_vb, h4_oa, h3_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob), 
			  "_a019( abab )(h4_oa, h3_ob, h1_oa, h2_ob) +=  0.25  * _a004( abab )(p1_va, p2_vb, h4_oa, h3_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)") 
        (_a019("bbbb")(h4_ob, h3_ob, h1_ob, h2_ob) += -0.125 * _a004("bbbb")(p1_vb, p2_vb, h3_ob, h4_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob), 
			  "_a019( bbbb )(h4_ob, h3_ob, h1_ob, h2_ob) += -0.125 * _a004( bbbb )(p1_vb, p2_vb, h3_ob, h4_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a019("aaaa")(h4_oa, h3_oa, h1_oa, h2_oa) * t2_aaaa(p3_va, p4_va, h4_oa, h3_oa), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a019( aaaa )(h4_oa, h3_oa, h1_oa, h2_oa) * t2_aaaa(p3_va, p4_va, h4_oa, h3_oa)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a019("bbbb")(h4_ob, h3_ob, h1_ob, h2_ob) * t2_bbbb(p3_vb, p4_vb, h4_ob, h3_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a019( bbbb )(h4_ob, h3_ob, h1_ob, h2_ob) * t2_bbbb(p3_vb, p4_vb, h4_ob, h3_ob)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a019("abab")(h4_oa, h3_ob, h1_oa, h2_ob) * t2_abab(p3_va, p4_vb, h4_oa, h3_ob), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a019( abab )(h4_oa, h3_ob, h1_oa, h2_ob) * t2_abab(p3_va, p4_vb, h4_oa, h3_ob)")
        (_a020("aaaa")(p1_va, h3_oa, p4_va, h2_oa) +=  0.5   * _a004("aaaa")(p2_va, p4_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa), 
			  "_a020( aaaa )(p1_va, h3_oa, p4_va, h2_oa) +=  0.5   * _a004( aaaa )(p2_va, p4_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)") 
        (_a020("baab")(p1_vb, h3_oa, p4_va, h2_ob)  = -0.5   * _a004("aaaa")(p2_va, p4_va, h3_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob), 
			  "_a020( baab )(p1_vb, h3_oa, p4_va, h2_ob)  = -0.5   * _a004( aaaa )(p2_va, p4_va, h3_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)") 
        (_a020("abba")(p1_va, h3_ob, p4_vb, h2_oa)  = -0.5   * _a004("bbbb")(p2_vb, p4_vb, h3_ob, h1_ob) * t2_abab(p1_va,p2_vb,h2_oa,h1_ob), 
			  "_a020( abba )(p1_va, h3_ob, p4_vb, h2_oa)  = -0.5   * _a004( bbbb )(p2_vb, p4_vb, h3_ob, h1_ob) * t2_abab(p1_va,p2_vb,h2_oa,h1_ob)")
        (_a020("bbbb")(p1_vb, h3_ob, p4_vb, h2_ob) +=  0.5   * _a004("bbbb")(p2_vb, p4_vb, h3_ob, h1_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob), 
			  "_a020( bbbb )(p1_vb, h3_ob, p4_vb, h2_ob) +=  0.5   * _a004( bbbb )(p2_vb, p4_vb, h3_ob, h1_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob)")
        (_a020("baba")(p1_vb, h7_oa, p6_vb, h2_oa) +=  1.0   * _a004("abab")(p5_va, p6_vb, h7_oa, h8_ob) * t2_abab(p5_va,p1_vb,h2_oa,h8_ob), 
			  "_a020( baba )(p1_vb, h7_oa, p6_vb, h2_oa) +=  1.0   * _a004( abab )(p5_va, p6_vb, h7_oa, h8_ob) * t2_abab(p5_va,p1_vb,h2_oa,h8_ob)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a020("aaaa")(p4_va, h4_oa, p1_va, h1_oa) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a020( aaaa )(p4_va, h4_oa, p1_va, h1_oa) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -1.0   * _a020("abba")(p4_va, h4_ob, p1_vb, h1_oa) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -1.0   * _a020( abba )(p4_va, h4_ob, p1_vb, h1_oa) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a020("bbbb")(p4_vb, h4_ob, p1_vb, h1_ob) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a020( bbbb )(p4_vb, h4_ob, p1_vb, h1_ob) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -1.0   * _a020("baab")(p4_vb, h4_oa, p1_va, h1_ob) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -1.0   * _a020( baab )(p4_vb, h4_oa, p1_va, h1_ob) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob)")
        (i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020("baba")(p1_vb, h7_oa, p6_vb, h2_oa) * t2_abab(p3_va, p6_vb, h7_oa, h4_ob), 
			  "i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020( baba )(p1_vb, h7_oa, p6_vb, h2_oa) * t2_abab(p3_va, p6_vb, h7_oa, h4_ob)")
        (i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020("abab")(p3_va, h8_ob, p5_va, h4_ob) * t2_abab(p5_va, p1_vb, h2_oa, h8_ob), 
			  "i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020( abab )(p3_va, h8_ob, p5_va, h4_ob) * t2_abab(p5_va, p1_vb, h2_oa, h8_ob)")
        (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       +=  1.0   * _a020("bbbb")(p4_vb, h4_ob, p1_vb, h1_ob) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob), 
			  "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       +=  1.0   * _a020( bbbb )(p4_vb, h4_ob, p1_vb, h1_ob) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob)")
        (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * _a020("baab")(p4_vb, h4_oa, p1_va, h1_ob) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa), 
			  "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * _a020( baab )(p4_vb, h4_oa, p1_va, h1_ob) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa)")
        (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       +=  1.0   * _a020("aaaa")(p4_va, h4_oa, p1_va, h1_oa) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob), 
			  "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       +=  1.0   * _a020( aaaa )(p4_va, h4_oa, p1_va, h1_oa) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob)")
        (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * _a020("abba")(p4_va, h4_ob, p1_vb, h1_oa) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob), 
			  "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * _a020( abba )(p4_va, h4_ob, p1_vb, h1_oa) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob)")

        (_a001("aa")(p4_va, p1_va)                 += -1.0   * f1_vv("aa")(p4_va, p1_va), 
			  "_a001( aa )(p4_va, p1_va)                 += -1.0   * f1_vv( aa )(p4_va, p1_va)")
        (_a001("bb")(p4_vb, p1_vb)                 += -1.0   * f1_vv("bb")(p4_vb, p1_vb), 
			  "_a001( bb )(p4_vb, p1_vb)                 += -1.0   * f1_vv( bb )(p4_vb, p1_vb)")
        (_a001("aa")(p4_va, p1_va)                 +=  1.0   * t1_aa(p4_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va), 
			  "_a001( aa )(p4_va, p1_va)                 +=  1.0   * t1_aa(p4_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
        (_a001("bb")(p4_vb, p1_vb)                 +=  1.0   * t1_bb(p4_vb, h1_ob) * f1_ov("bb")(h1_ob, p1_vb), 
			  "_a001( bb )(p4_vb, p1_vb)                 +=  1.0   * t1_bb(p4_vb, h1_ob) * f1_ov( bb )(h1_ob, p1_vb)") // NEW TERM
        (_a006("aa")(h9_oa, h1_oa)                 +=  1.0   * f1_oo("aa")(h9_oa, h1_oa), 
			  "_a006( aa )(h9_oa, h1_oa)                 +=  1.0   * f1_oo( aa )(h9_oa, h1_oa)")
        (_a006("bb")(h9_ob, h1_ob)                 +=  1.0   * f1_oo("bb")(h9_ob, h1_ob), 
			  "_a006( bb )(h9_ob, h1_ob)                 +=  1.0   * f1_oo( bb )(h9_ob, h1_ob)")
        (_a006("aa")(h9_oa, h1_oa)                 +=  1.0   * t1_aa(p8_va, h1_oa) * f1_ov("aa")(h9_oa, p8_va), 
			  "_a006( aa )(h9_oa, h1_oa)                 +=  1.0   * t1_aa(p8_va, h1_oa) * f1_ov( aa )(h9_oa, p8_va)")
        (_a006("bb")(h9_ob, h1_ob)                 +=  1.0   * t1_bb(p8_vb, h1_ob) * f1_ov("bb")(h9_ob, p8_vb), 
			  "_a006( bb )(h9_ob, h1_ob)                 +=  1.0   * t1_bb(p8_vb, h1_ob) * f1_ov( bb )(h9_ob, p8_vb)")

        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -0.5   * t2_aaaa(p3_va, p2_va, h1_oa, h2_oa) * _a001("aa")(p4_va, p2_va), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -0.5   * t2_aaaa(p3_va, p2_va, h1_oa, h2_oa) * _a001( aa )(p4_va, p2_va)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -0.5   * t2_bbbb(p3_vb, p2_vb, h1_ob, h2_ob) * _a001("bb")(p4_vb, p2_vb), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -0.5   * t2_bbbb(p3_vb, p2_vb, h1_ob, h2_ob) * _a001( bb )(p4_vb, p2_vb)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p2_vb, h1_oa, h2_ob) * _a001("bb")(p4_vb, p2_vb), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p2_vb, h1_oa, h2_ob) * _a001( bb )(p4_vb, p2_vb)")
        (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p2_va, p3_vb, h1_oa, h2_ob) * _a001("aa")(p4_va, p2_va), 
			  "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p2_va, p3_vb, h1_oa, h2_ob) * _a001( aa )(p4_va, p2_va)")

        (i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)       += -0.5   * t2_aaaa(p3_va, p4_va, h3_oa, h1_oa) * _a006("aa")(h3_oa, h2_oa), 
			  "i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)       += -0.5   * t2_aaaa(p3_va, p4_va, h3_oa, h1_oa) * _a006( aa )(h3_oa, h2_oa)")
        (i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)       += -0.5   * t2_bbbb(p3_vb, p4_vb, h3_ob, h1_ob) * _a006("bb")(h3_ob, h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)       += -0.5   * t2_bbbb(p3_vb, p4_vb, h3_ob, h1_ob) * _a006( bb )(h3_ob, h2_ob)")
        (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h3_oa, h1_ob) * _a006("aa")(h3_oa, h2_oa), 
			  "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h3_oa, h1_ob) * _a006( aa )(h3_oa, h2_oa)")
        (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h1_oa, h3_ob) * _a006("bb")(h3_ob, h2_ob), 
			  "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h1_oa, h3_ob) * _a006( bb )(h3_ob, h2_ob)")

        (i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa)  =  1.0   * i0_aaaa(p3_va, p4_va, h1_oa, h2_oa), 
			  "i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa)  =  1.0   * i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)")        
        (i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa) +=  1.0   * i0_aaaa(p4_va, p3_va, h2_oa, h1_oa), 
			  "i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa) +=  1.0   * i0_aaaa(p4_va, p3_va, h2_oa, h1_oa)")        
        (i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa) += -1.0   * i0_aaaa(p3_va, p4_va, h2_oa, h1_oa), 
			  "i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa) += -1.0   * i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)")        
        (i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa) += -1.0   * i0_aaaa(p4_va, p3_va, h1_oa, h2_oa), 
			  "i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa) += -1.0   * i0_aaaa(p4_va, p3_va, h1_oa, h2_oa)")
        (i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob)  =  1.0   * i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob), 
			  "i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob)  =  1.0   * i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)")
        (i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob) +=  1.0   * i0_bbbb(p4_vb, p3_vb, h2_ob, h1_ob), 
			  "i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob) +=  1.0   * i0_bbbb(p4_vb, p3_vb, h2_ob, h1_ob)")
        (i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0   * i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob), 
			  "i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0   * i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)") 
        (i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0   * i0_bbbb(p4_vb, p3_vb, h1_ob, h2_ob), 
			  "i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0   * i0_bbbb(p4_vb, p3_vb, h1_ob, h2_ob)")
        (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  1.0   * i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa), 
			  "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  1.0   * i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa)")
        (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  1.0   * i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob), 
			  "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  1.0   * i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob)")
        ;
        
}


template<typename T>
std::tuple<double,double> cd_ccsd_os_driver(SystemData& sys_data, ExecutionContext& ec,
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
    CCSE_Tensors<CCEType> r1_vo, r2_vvoo; //r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb;

    CCSE_Tensors<T> f1_oo{MO,{O,O},"f1_oo",{"aa","bb"}};
    CCSE_Tensors<T> f1_ov{MO,{O,V},"f1_ov",{"aa","bb"}};
    CCSE_Tensors<T> f1_vo{MO,{V,O},"f1_vo",{"aa","bb"}};
    CCSE_Tensors<T> f1_vv{MO,{V,V},"f1_vv",{"aa","bb"}};

    CCSE_Tensors<T> chol3d_oo{MO,{O,O,CI},"chol3d_oo",{"aa","bb"}};
    CCSE_Tensors<T> chol3d_ov{MO,{O,V,CI},"chol3d_ov",{"aa","bb"}};
    CCSE_Tensors<T> chol3d_vo{MO,{V,O,CI},"chol3d_vo",{"aa","bb"}};
    CCSE_Tensors<T> chol3d_vv{MO,{V,V,CI},"chol3d_vv",{"aa","bb"}};

    std::vector<CCSE_Tensors<T>> f1_se{f1_oo,f1_ov,f1_vo,f1_vv};
    std::vector<CCSE_Tensors<T>> chol3d_se{chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv};

    CCSE_Tensors<T> t1_vo  {MO,{V,O},"t1",{"aa","bb"}};
    CCSE_Tensors<T> t2_vvoo{MO,{V,V,O,O},"t2",{"aaaa","abab","bbbb"}};

    r1_vo    = CCSE_Tensors<T>{MO,{V,O},"r1",{"aa","bb"}};
    r2_vvoo  = CCSE_Tensors<T>{MO,{V,V,O,O},"r2",{"aaaa","abab","bbbb"}};

    _a004 = CCSE_Tensors<T>{MO,{V,V,O,O},"_a004",{"aaaa","abab","bbbb"}};

    //Energy intermediates
    _a01V = {CI};
    _a02  = CCSE_Tensors<T>{MO,{O,O,CI},"_a02",{"aa","bb"}};
    _a03  = CCSE_Tensors<T>{MO,{O,V,CI},"_a03",{"aa","bb"}};
    
    //T1
    _a02V = {CI};
    _a01  = CCSE_Tensors<T>{MO,{O,O,CI},"_a01",{"aa","bb"}};
    _a04  = CCSE_Tensors<T>{MO,{O,O},"_a04",{"aa","bb"}};
    _a05  = CCSE_Tensors<T>{MO,{O,V},"_a05",{"aa","bb"}};
    _a06  = CCSE_Tensors<T>{MO,{V,O,CI},"_a06",{"aa","bb"}};

    //T2
    _a007V = {CI};
    _a001  = CCSE_Tensors<T>{MO,{V,V},"_a001",{"aa","bb"}};
    _a006  = CCSE_Tensors<T>{MO,{O,O},"_a006",{"aa","bb"}};
    _a008  = CCSE_Tensors<T>{MO,{O,O,CI},"_a008",{"aa","bb"}};
    _a009  = CCSE_Tensors<T>{MO,{O,O,CI},"_a009",{"aa","bb"}};
    _a017  = CCSE_Tensors<T>{MO,{V,O,CI},"_a017",{"aa","bb"}};
    _a021  = CCSE_Tensors<T>{MO,{V,V,CI},"_a021",{"aa","bb"}};

    _a019  = CCSE_Tensors<T>{MO,{O,O,O,O},"_a019",{"aaaa","abab","bbbb"}};
    _a022  = CCSE_Tensors<T>{MO,{V,V,V,V},"_a022",{"aaaa","abab","bbbb"}};
    _a020  = CCSE_Tensors<T>{MO,{V,O,V,O},"_a020",{"aaaa","abab","baab","abba","baba","bbbb"}};

    CCSE_Tensors<CCEType> i0_t2_tmp{MO,{V,V,O,O},"i0_t2_tmp",{"aaaa","bbbb"}};

    double total_ccsd_mem = sum_tensor_sizes(d_t1,d_t2,d_f1,d_r1,d_r2,cv3d,d_e,_a01V)
        + CCSE_Tensors<T>::sum_tensor_sizes_list(r1_vo,r2_vvoo,t1_vo,t2_vvoo)
        + CCSE_Tensors<T>::sum_tensor_sizes_list(f1_oo,f1_ov,f1_vo,f1_vv,
                           chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv)
        + CCSE_Tensors<T>::sum_tensor_sizes_list(_a02,_a03);

    for(size_t ri=0;ri<d_r1s.size();ri++)
        total_ccsd_mem += sum_tensor_sizes(d_r1s[ri],d_r2s[ri],d_t1s[ri],d_t2s[ri]);

    //Intermediates
    double total_ccsd_mem_tmp = sum_tensor_sizes(_a02V,_a007V) 
      + CCSE_Tensors<T>::sum_tensor_sizes_list(i0_t2_tmp,_a01,_a04,_a05,_a06,_a001,
                            _a004,_a006,_a008,_a009,_a017,_a019,_a020,_a021,_a022);

    if(!ccsd_restart) total_ccsd_mem += total_ccsd_mem_tmp;

    if(ec.print()) {
      std::cout << std::endl << "Total CPU memory required for Open Shell Cholesky CCSD calculation: " 
                << std::setprecision(5) << total_ccsd_mem << " GiB" << std::endl;
    }

    Scheduler sch{ec};
    ExecutionHW exhw = ec.exhw();

    sch.allocate(d_e, _a01V);
    CCSE_Tensors<T>::allocate_list(sch,f1_oo,f1_ov,f1_vo,f1_vv,
                        chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv);
    CCSE_Tensors<T>::allocate_list(sch,r1_vo,r2_vvoo,t1_vo,t2_vvoo);
    CCSE_Tensors<T>::allocate_list(sch,_a02,_a03);
    sch.execute();

    const int pcore = sys_data.options_map.ccsd_options.pcore-1; //0-based indexing
    if(pcore >= 0)
    {
      const auto timer_start = std::chrono::high_resolution_clock::now();

      TiledIndexSpace mo_ut{IndexSpace{range(0,MO.max_num_indices())},1};
      TiledIndexSpace cv3d_occ {mo_ut,range(0,O.max_num_indices())};
      TiledIndexSpace cv3d_virt{mo_ut,range(O.max_num_indices(),MO.max_num_indices())};

      auto [h1, h2, h3, h4] = cv3d_occ.labels<4>("all");
      auto [p1, p2, p3, p4] = cv3d_virt.labels<4>("all");

      Tensor<T> d_f1_ut = redistribute_tensor<T>(d_f1,(TiledIndexSpaceVec){mo_ut,mo_ut});
      Tensor<T> cv3d_ut = redistribute_tensor<T>(cv3d,(TiledIndexSpaceVec){mo_ut,mo_ut,CI});

      TiledIndexSpace cv3d_utis{mo_ut,range(sys_data.nmo-sys_data.n_vir_beta,sys_data.nmo-sys_data.n_vir_beta+1)};
      auto [c1,c2] = cv3d_utis.labels<2>("all");


      sch
      (d_f1_ut(h1,h2) += -1.0 * cv3d_ut(h1,h2,cind) * cv3d_ut(c1,c2,cind))
      (d_f1_ut(h1,h2) +=  1.0 * cv3d_ut(h1,c1,cind) * cv3d_ut(h2,c2,cind))
      (d_f1_ut(p1,p2) += -1.0 * cv3d_ut(p1,p2,cind) * cv3d_ut(c1,c2,cind))
      (d_f1_ut(p1,p2) +=  1.0 * cv3d_ut(p1,c1,cind) * cv3d_ut(p2,c2,cind))
      (d_f1_ut(p1,h1) += -1.0 * cv3d_ut(p1,h1,cind) * cv3d_ut(c1,c2,cind))
      (d_f1_ut(p1,h1) +=  1.0 * cv3d_ut(p1,c1,cind) * cv3d_ut(h1,c2,cind))
      (d_f1_ut(h1,p1) += -1.0 * cv3d_ut(h1,p1,cind) * cv3d_ut(c1,c2,cind))
      (d_f1_ut(h1,p1) +=  1.0 * cv3d_ut(h1,c1,cind) * cv3d_ut(p1,c2,cind));

      sch.execute(exhw);

      sch.deallocate(d_f1,cv3d_ut).execute();

      d_f1 = redistribute_tensor<T>(d_f1_ut,(TiledIndexSpaceVec){MO,MO},{1,1});
      sch.deallocate(d_f1_ut).execute();

      p_evl_sorted = tamm::diagonal(d_f1);

      const auto timer_end = std::chrono::high_resolution_clock::now();
      auto f1_rctime = std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();
      if(ec.print()) std::cout << "Time to reconstruct Fock matrix: " << f1_rctime << " secs" << std::endl;
    }

    print_ccsd_header(ec.print());

    sch
       (f1_oo("aa")(h3_oa,h4_oa)           =  d_f1(h3_oa,h4_oa))
       (f1_ov("aa")(h3_oa,p2_va)           =  d_f1(h3_oa,p2_va))
       (f1_vo("aa")(p1_va,h4_oa)           =  d_f1(p1_va,h4_oa))
       (f1_vv("aa")(p1_va,p2_va)           =  d_f1(p1_va,p2_va))
       (f1_oo("bb")(h3_ob,h4_ob)           =  d_f1(h3_ob,h4_ob))
       (f1_ov("bb")(h3_ob,p1_vb)           =  d_f1(h3_ob,p1_vb))
       (f1_vo("bb")(p1_vb,h3_ob)           =  d_f1(p1_vb,h3_ob))
       (f1_vv("bb")(p1_vb,p2_vb)           =  d_f1(p1_vb,p2_vb))
       (chol3d_oo("aa")(h3_oa,h4_oa,cind)  =  cv3d(h3_oa,h4_oa,cind))
       (chol3d_ov("aa")(h3_oa,p2_va,cind)  =  cv3d(h3_oa,p2_va,cind))
       (chol3d_vo("aa")(p1_va,h4_oa,cind)  =  cv3d(p1_va,h4_oa,cind))
       (chol3d_vv("aa")(p1_va,p2_va,cind)  =  cv3d(p1_va,p2_va,cind))
       (chol3d_oo("bb")(h3_ob,h4_ob,cind)  =  cv3d(h3_ob,h4_ob,cind))
       (chol3d_ov("bb")(h3_ob,p1_vb,cind)  =  cv3d(h3_ob,p1_vb,cind))
       (chol3d_vo("bb")(p1_vb,h3_ob,cind)  =  cv3d(p1_vb,h3_ob,cind))
       (chol3d_vv("bb")(p1_vb,p2_vb,cind)  =  cv3d(p1_vb,p2_vb,cind))
       ;

    if(!ccsd_restart) {

        //allocate all intermediates
        sch.allocate(_a02V,_a007V);
        CCSE_Tensors<T>::allocate_list(sch,_a004,i0_t2_tmp,_a01,_a04,_a05,_a06,_a001,
                                  _a006,_a008,_a009,_a017,_a019,_a020,_a021,_a022);
        sch.execute();

        sch
           (_a004("aaaa")(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_vo("aa")(p1_va, h4_oa, cind) * chol3d_vo("aa")(p2_va, h3_oa, cind))
           (_a004("abab")(p1_va, p2_vb, h4_oa, h3_ob) = 1.0 * chol3d_vo("aa")(p1_va, h4_oa, cind) * chol3d_vo("bb")(p2_vb, h3_ob, cind))
           (_a004("bbbb")(p1_vb, p2_vb, h4_ob, h3_ob) = 1.0 * chol3d_vo("bb")(p1_vb, h4_ob, cind) * chol3d_vo("bb")(p2_vb, h3_ob, cind));

        sch.execute(exhw);

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
               (t1_vo("aa")(p1_va,h3_oa)                 = d_t1(p1_va,h3_oa))
               (t1_vo("bb")(p1_vb,h3_ob)                 = d_t1(p1_vb,h3_ob))
               (t2_vvoo("aaaa")(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1_va,p2_va,h3_oa,h4_oa))
               (t2_vvoo("abab")(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1_va,p2_vb,h3_oa,h4_ob))
               (t2_vvoo("bbbb")(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1_vb,p2_vb,h3_ob,h4_ob))
               .execute();

            ccsd_e_os (sch, MO, CI, d_e, t1_vo, t2_vvoo, f1_se, chol3d_se);
            ccsd_t1_os(sch, MO, CI, /*d_r1,*/ r1_vo, t1_vo, t2_vvoo, f1_se, chol3d_se);
            ccsd_t2_os(sch, MO, CI, /*d_r2,*/ r2_vvoo, t1_vo, t2_vvoo, f1_se, chol3d_se, i0_t2_tmp);

            sch
              (d_r1(p2_va, h1_oa)                = r1_vo("aa")(p2_va, h1_oa))
              (d_r1(p2_vb, h1_ob)                = r1_vo("bb")(p2_vb, h1_ob))
              (d_r2(p3_va, p4_va, h2_oa, h1_oa)  = r2_vvoo("aaaa")(p3_va, p4_va, h2_oa, h1_oa))
              (d_r2(p3_vb, p4_vb, h2_ob, h1_ob)  = r2_vvoo("bbbb")(p3_vb, p4_vb, h2_ob, h1_ob))
              (d_r2(p3_va, p4_vb, h2_oa, h1_ob)  = r2_vvoo("abab")(p3_va, p4_vb, h2_oa, h1_ob))
              ;

            sch.execute(exhw, profile);

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

        if(profile && ec.print()) {
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
        
        sch.deallocate(_a02V,_a007V,d_r1_residual, d_r2_residual);
        CCSE_Tensors<T>::deallocate_list(sch,_a004,i0_t2_tmp,_a01,_a04,_a05,_a06,_a001,
                                    _a006,_a008,_a009,_a017,_a019,_a020,_a021,_a022);


    } //no restart
    else {
      sch
         (d_e()=0)
         (t1_vo("aa")(p1_va,h3_oa) = d_t1(p1_va,h3_oa))
         (t1_vo("bb")(p1_vb,h3_ob) = d_t1(p1_vb,h3_ob))
         (t2_vvoo("aaaa")(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1_va,p2_va,h3_oa,h4_oa))
         (t2_vvoo("abab")(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1_va,p2_vb,h3_oa,h4_ob))
         (t2_vvoo("bbbb")(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1_vb,p2_vb,h3_ob,h4_ob));

      ccsd_e_os(sch, MO, CI, d_e, t1_vo, t2_vvoo, f1_se, chol3d_se);

      sch.execute(exhw, profile);

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

    CCSE_Tensors<T>::deallocate_list(sch,_a02,_a03);
    CCSE_Tensors<T>::deallocate_list(sch,r1_vo,r2_vvoo,t1_vo,t2_vvoo);
    CCSE_Tensors<T>::deallocate_list(sch,f1_oo,f1_ov,f1_vo,f1_vv,
                          chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv);
    sch.deallocate(d_e,_a01V).execute();

    return std::make_tuple(residual,energy);
}
