#include "diis.hpp"
#include "ccsd_util.hpp"
#include "macdecls.h"
#include "ga-mpi.h"

using namespace tamm;

bool debug = false;

TiledIndexSpace o_alpha,v_alpha,o_beta,v_beta;
Tensor<double> f1_aa_oo, f1_aa_ov, f1_aa_vv;
Tensor<double> f1_bb_oo, f1_bb_ov, f1_bb_vv;
Tensor<double> t2_aaaa;
Tensor<double> chol3d_aa_oo, chol3d_aa_ov, chol3d_aa_vv;
Tensor<double> chol3d_bb_oo, chol3d_bb_ov, chol3d_bb_vv;
Tensor<double> _a004_aaaa, _a004_abab;

Tensor<double> t2_aaaa_temp;
Tensor<double> i0_temp;
Tensor<double> _a01,_a02_aa,_a03_aa;
Tensor<double> _a02, _a01_aa,_a03_aa_vo,_a04_aa,_a05_aa,_a05_bb;
Tensor<double> _a007,_a001_aa,_a001_bb,_a017_aa,_a017_bb;
Tensor<double> _a006_aa,_a006_bb,_a009_aa,_a009_bb,_a021_aa,_a021_bb,_a008_aa,
               _a019_abab,_a020_aaaa,_a020_baba,_a020_baab,_a020_bbbb,_a022_abab;

template<typename T>
void ccsd_e(/* ExecutionContext &ec, */
            Scheduler& sch,
            const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de, 
            const Tensor<T>& t1_aa, const Tensor<T>& t2_abab) { 

    auto [cind] = CI.labels<1>("all");
    
    auto [p1_va, p2_va] = v_alpha.labels<2>("all");
    auto [p1_vb]        = v_beta.labels<1>("all");
    auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
    auto [h1_ob]        = o_beta.labels<1>("all");

    sch
       (t2_aaaa_temp()=0)
       .exact_copy(t2_aaaa(p1_va, p2_va, h1_oa, h2_oa), t2_abab(p1_va, p2_va, h1_oa, h2_oa))
       (t2_aaaa_temp() = t2_aaaa())
       (t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa))
       (t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa))

       (_a01(cind) = t1_aa(p1_va, h1_oa) * chol3d_aa_ov(h1_oa, p1_va, cind))
       (_a02_aa(h1_oa, h2_oa, cind)  = t1_aa(p1_va, h1_oa) * chol3d_aa_ov(h2_oa, p1_va, cind))
       (_a03_aa(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_aa_ov(h1_oa, p1_va, cind))
       (de()  =  2.0 * _a01() * _a01())
       (de() += -1.0 * _a02_aa(h1_oa, h2_oa, cind) * _a02_aa(h2_oa, h1_oa, cind))
       (de() +=  1.0 * _a03_aa(h1_oa, p1_va, cind) * chol3d_aa_ov(h1_oa, p1_va, cind))
       ;

}

template<typename T>
void ccsd_t1(/* ExecutionContext& ec,  */
             Scheduler& sch,
             const TiledIndexSpace& MO,const TiledIndexSpace& CI, 
             Tensor<T>& i0_aa, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab) { 
    
    auto [cind] = CI.labels<1>("all");
    auto [p2] = MO.labels<1>("virt");
    auto [h1] = MO.labels<1>("occ");

    auto [p1_va, p2_va] = v_alpha.labels<2>("all");
    auto [p1_vb]        = v_beta.labels<1>("all");
    auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
    auto [h1_ob]        = o_beta.labels<1>("all");

    sch
       (i0_aa(p2_va, h1_oa)             =  1.0 * f1_aa_ov(h1_oa, p2_va))
       (_a01_aa(h2_oa, h1_oa, cind)     =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_aa_ov(h2_oa, p1_va, cind))                 // ovm
       (_a02(cind)                      =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_aa_ov(h1_oa, p1_va, cind))                 // ovm
    //    (_a02(cind)                      =  2.0 * _a01_aa(h1_oa, h1_oa, cind))
       (_a05_aa(h2_oa, p1_va)           = -1.0 * chol3d_aa_ov(h1_oa, p1_va, cind) * _a01_aa(h2_oa, h1_oa, cind))         // o2vm
       .exact_copy(_a05_bb(h1_ob,p1_vb),_a05_aa(h1_ob,p1_vb))

       (_a03_aa_vo(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_aa_ov(h2_oa, p2_va, cind)) // o2v2m
       (_a04_aa(h2_oa, h1_oa)           =  1.0 * chol3d_aa_ov(h2_oa, p1_va, cind) * _a03_aa_vo(p1_va, h1_oa, cind))      // o2vm
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04_aa(h2_oa, h1_oa))                            // o2v
       (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_aa_ov(h2_oa, p1_va, cind) * _a02(cind))                          // ovm
       (i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05_aa(h1_oa, p2_va))
       (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_aa_vv(p2_va, p1_va, cind) * _a03_aa_vo(p1_va, h1_oa, cind))      // ov2m
       (_a03_aa_vo(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_aa_vv(p2_va, p1_va, cind))                 // ov2m
       (i0_aa(p1_va, h2_oa)            += -1.0 * _a03_aa_vo(p1_va, h2_oa, cind) * _a02(cind))                            // ovm
       (_a03_aa_vo(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02(cind))                                       // ovm
       (_a03_aa_vo(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01_aa(h2_oa, h1_oa, cind))                      // o2vm
       (_a01_aa(h2_oa, h1_oa, cind)    +=  1.0 * chol3d_aa_oo(h2_oa, h1_oa, cind))                                       // o2m
       (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01_aa(h2_oa, h1_oa, cind) * _a03_aa_vo(p2_va, h2_oa, cind))           // o2vm
       (i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1_aa_oo(h2_oa, h1_oa))                           // o2v
       (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1_aa_vv(p2_va, p1_va))                           // ov2
       ;
}

template<typename T>
void ccsd_t2(/* ExecutionContext& ec, */
             Scheduler& sch,
             const TiledIndexSpace& MO,const TiledIndexSpace& CI, 
             Tensor<T>& i0_abab, const Tensor<T>& t1_aa, Tensor<T>& t2_abab) { 

    auto [cind] = CI.labels<1>("all");
    auto [p3, p4] = MO.labels<2>("virt");
    auto [h1, h2] = MO.labels<2>("occ");

    auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
    auto [p1_vb, p2_vb]        = v_beta.labels<2>("all");
    auto [h1_oa, h2_oa, h3_oa] = o_alpha.labels<3>("all");
    auto [h1_ob, h2_ob]        = o_beta.labels<2>("all");

    sch
        (_a017_aa(p1_va, h2_oa, cind)            = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d_aa_ov(h1_oa, p2_va, cind))
        (_a006_aa(h2_oa, h1_oa)                  = -1.0  * chol3d_aa_ov(h2_oa, p2_va, cind) * _a017_aa(p2_va, h1_oa, cind))
        (_a007(cind)                             =  2.0  * chol3d_aa_ov(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa))
        (_a009_aa(h1_oa, h2_oa, cind)            =  1.0  * chol3d_aa_ov(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa))
        (_a021_aa(p2_va, p1_va, cind)            = -0.5  * chol3d_aa_ov(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa))
        (_a021_aa(p2_va, p1_va, cind)           +=  0.5  * chol3d_aa_vv(p2_va, p1_va, cind))
        (_a017_aa(p1_va, h2_oa, cind)           += -2.0  * t1_aa(p2_va, h2_oa) * _a021_aa(p1_va, p2_va, cind))
        (_a008_aa(h2_oa, h1_oa, cind)            =  1.0  * _a009_aa(h2_oa, h1_oa, cind))
        (_a009_aa(h2_oa, h1_oa, cind)           +=  1.0  * chol3d_aa_oo(h2_oa, h1_oa, cind))
        .exact_copy(_a009_bb(h2_ob,h1_ob,cind),_a009_aa(h2_ob,h1_ob,cind))
        .exact_copy(_a021_bb(p2_vb,p1_vb,cind),_a021_aa(p2_vb,p1_vb,cind))
        (_a001_aa(p1_va, p2_va)                  = -2.0  * _a021_aa(p1_va, p2_va, cind) * _a007(cind))
        (_a001_aa(p1_va, p2_va)                 += -1.0  * _a017_aa(p1_va, h2_oa, cind) * chol3d_aa_ov(h2_oa, p2_va, cind))
        (_a006_aa(h2_oa, h1_oa)                 +=  1.0  * _a009_aa(h2_oa, h1_oa, cind) * _a007(cind))
        (_a006_aa(h3_oa, h1_oa)                 += -1.0  * _a009_aa(h2_oa, h1_oa, cind) * _a008_aa(h3_oa, h2_oa, cind))
        (_a019_abab(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009_aa(h2_oa, h1_oa, cind) * _a009_bb(h1_ob, h2_ob, cind))
        (_a020_aaaa(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009_aa(h2_oa, h1_oa, cind) * _a021_aa(p2_va, p1_va, cind))
        .exact_copy(_a020_baba(p2_vb, h2_oa, p1_vb, h1_oa),_a020_aaaa(p2_vb, h2_oa, p1_vb, h1_oa))
        (_a020_aaaa(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004_aaaa(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)) 
        (_a020_baab(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004_aaaa(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)) 
        (_a020_baba(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004_abab(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob))
        (_a017_aa(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d_aa_oo(h1_oa, h2_oa, cind))
        (_a017_aa(p1_va, h2_oa, cind)           += -1.0  * chol3d_aa_ov(h2_oa, p1_va, cind))
        (_a001_aa(p2_va, p1_va)                 += -1.0  * f1_aa_vv(p2_va, p1_va))
        (_a006_aa(h2_oa, h1_oa)                 +=  1.0  * f1_aa_oo(h2_oa, h1_oa))
        (_a006_aa(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1_aa_ov(h2_oa, p1_va))
        .exact_copy(_a017_bb(p1_vb, h1_ob, cind), _a017_aa(p1_vb, h1_ob, cind))
        .exact_copy(_a006_bb(h1_ob, h2_ob), _a006_aa(h1_ob, h2_ob))
        .exact_copy(_a001_bb(p1_vb, p2_vb), _a001_aa(p1_vb, p2_vb))
        .exact_copy(_a021_bb(p1_vb, p2_vb, cind), _a021_aa(p1_vb, p2_vb, cind))
        .exact_copy(_a020_bbbb(p1_vb, h1_ob, p2_vb, h2_ob), _a020_aaaa(p1_vb, h1_ob, p2_vb, h2_ob))
        
        (i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020_bbbb(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob))
        (i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020_baab(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa))
        (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020_baba(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob))
        .exact_copy(i0_temp(p1_vb,p1_va,h2_ob,h1_oa),i0_abab(p1_vb,p1_va,h2_ob,h1_oa))
        (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa))
        (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017_aa(p1_va, h1_oa, cind) * _a017_bb(p1_vb, h2_ob, cind))
        (_a022_abab(p1_va,p2_vb,p2_va,p1_vb)     =  1.0  * _a021_aa(p1_va,p2_va,cind) * _a021_bb(p2_vb,p1_vb,cind))
        (i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * _a022_abab(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob))
        (_a019_abab(h2_oa, h1_ob, h1_oa, h2_ob) +=  0.25 * _a004_abab(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)) 
        (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019_abab(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob))
        (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001_bb(p1_vb, p2_vb))
        (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001_aa(p1_va, p2_va))
        (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006_aa(h1_oa, h2_oa))
        (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006_bb(h1_ob, h2_ob))
        ;

}

template<typename T>
std::tuple<double,double> cd_ccsd_cs_driver(SystemData sys_data, ExecutionContext& ec, 
                   const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                   Tensor<T>& t1_aa, Tensor<T>& t2_abab,
                   Tensor<T>& d_f1, 
                   Tensor<T>& r1_aa, Tensor<T>& r2_abab, std::vector<Tensor<T>>& d_r1s, 
                   std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s, 
                   std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted,
                   Tensor<T>& cv3d, bool ccsd_restart=false, std::string out_fp="") {

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
    const TAMM_SIZE n_vir_alpha = static_cast<TAMM_SIZE>(sys_data.n_vir_alpha);
    
    std::string t1file = out_fp+".t1amp";
    std::string t2file = out_fp+".t2amp";                       

    std::cout.precision(15);

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");
    auto [cind] = CI.labels<1>("all");
    
    const int otiles = O.num_tiles();
    const int vtiles = V.num_tiles();
    const int oatiles = MO("occ_alpha").num_tiles();
    const int obtiles = MO("occ_beta").num_tiles();
    const int vatiles = MO("virt_alpha").num_tiles();
    const int vbtiles = MO("virt_beta").num_tiles();

    o_alpha = {MO("occ"), range(oatiles)};
    v_alpha = {MO("virt"), range(vatiles)};
    o_beta = {MO("occ"), range(obtiles,otiles)};
    v_beta = {MO("virt"), range(vbtiles,vtiles)};

    auto [p1_va, p2_va] = v_alpha.labels<2>("all");
    auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
    auto [h3_oa, h4_oa] = o_alpha.labels<2>("all");
    auto [h3_ob, h4_ob] = o_beta.labels<2>("all");


    t2_aaaa = {{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};

    _a004_aaaa = {{v_alpha,v_alpha,o_alpha,o_alpha},{2,2}};
    _a004_abab = {{v_alpha,v_beta,o_alpha,o_beta},{2,2}};
    
    f1_aa_oo = {{o_alpha,o_alpha},{1,1}};
    f1_aa_ov = {{o_alpha,v_alpha},{1,1}};
    f1_aa_vv = {{v_alpha,v_alpha},{1,1}};

    f1_bb_oo = {{o_beta,o_beta},{1,1}};
    f1_bb_ov = {{o_beta,v_beta},{1,1}};
    f1_bb_vv = {{v_beta,v_beta},{1,1}};

    chol3d_aa_oo = {{o_alpha,o_alpha,CI},{1,1}};
    chol3d_aa_ov = {{o_alpha,v_alpha,CI},{1,1}};
    chol3d_aa_vv = {{v_alpha,v_alpha,CI},{1,1}};

    chol3d_bb_oo = {{o_beta,o_beta,CI},{1,1}};
    chol3d_bb_ov = {{o_beta,v_beta,CI},{1,1}};
    chol3d_bb_vv = {{v_beta,v_beta,CI},{1,1}};

    Scheduler sch{ec};
    sch.allocate(t2_aaaa,  
                f1_aa_oo, f1_aa_ov, f1_aa_vv, f1_bb_oo, f1_bb_ov, f1_bb_vv,
                chol3d_aa_oo, chol3d_aa_ov, chol3d_aa_vv,
                chol3d_bb_oo, chol3d_bb_ov, chol3d_bb_vv);
    sch.allocate(_a004_aaaa,_a004_abab);

    sch
        (chol3d_aa_oo(h3_oa,h4_oa,cind) = cv3d(h3_oa,h4_oa,cind))
        (chol3d_aa_ov(h3_oa,p2_va,cind) = cv3d(h3_oa,p2_va,cind))
        (chol3d_aa_vv(p1_va,p2_va,cind) = cv3d(p1_va,p2_va,cind))
        (chol3d_bb_oo(h3_ob,h4_ob,cind) = cv3d(h3_ob,h4_ob,cind))
        (chol3d_bb_ov(h3_ob,p1_vb,cind) = cv3d(h3_ob,p1_vb,cind))
        (chol3d_bb_vv(p1_vb,p2_vb,cind) = cv3d(p1_vb,p2_vb,cind))

        (f1_aa_oo(h3_oa,h4_oa) = d_f1(h3_oa,h4_oa))
        (f1_aa_ov(h3_oa,p2_va) = d_f1(h3_oa,p2_va))
        (f1_aa_vv(p1_va,p2_va) = d_f1(p1_va,p2_va))
        (f1_bb_oo(h3_ob,h4_ob) = d_f1(h3_ob,h4_ob))
        (f1_bb_ov(h3_ob,p1_vb) = d_f1(h3_ob,p1_vb))
        (f1_bb_vv(p1_vb,p2_vb) = d_f1(p1_vb,p2_vb));

    Tensor<T> d_e{};
    Tensor<T>::allocate(&ec, d_e);

    _a01 = {CI};
    _a02_aa = {{o_alpha,o_alpha,CI},{1,1}}; 
    _a03_aa = {{o_alpha,v_alpha,CI},{1,1}}; 

    t2_aaaa_temp = {v_alpha,v_alpha,o_alpha,o_alpha};
    i0_temp = {v_beta,v_alpha,o_beta,o_alpha};

    sch.allocate(i0_temp,t2_aaaa_temp,_a01,_a02_aa,_a03_aa);

    if(!ccsd_restart) {

        { //allocate all intermediates 

        //T1
        _a02 = {CI};
        _a01_aa = {{o_alpha,o_alpha,CI},{1,1}}; 
        _a03_aa_vo = {{v_alpha,o_alpha,CI},{1,1}}; 
        _a04_aa = {{o_alpha,o_alpha},{1,1}}; 
        _a05_aa = {{o_alpha,v_alpha},{1,1}};    _a05_bb = {{o_beta,v_beta},{1,1}};

        sch.allocate(_a02, _a01_aa,_a03_aa_vo,_a04_aa,_a05_aa,_a05_bb);

        //T2
        _a007 = {CI};
        _a017_aa = {{v_alpha,o_alpha,CI},{1,1}};
        _a017_bb = {{v_beta,o_beta,CI},{1,1}};
        _a006_aa = {{o_alpha,o_alpha},{1,1}};
        _a006_bb = {{o_beta,o_beta},{1,1}};
        _a009_aa = {{o_alpha,o_alpha,CI},{1,1}};
        _a009_bb = {{o_beta,o_beta,CI},{1,1}};
        _a021_aa = {{v_alpha,v_alpha,CI},{1,1}};
        _a021_bb = {{v_beta,v_beta,CI},{1,1}};
        _a008_aa = {{o_alpha,o_alpha,CI},{1,1}};
        _a001_aa = {{v_alpha,v_alpha},{1,1}};
        _a001_bb = {{v_beta,v_beta},{1,1}};
        _a019_abab = {{o_alpha,o_beta,o_alpha,o_beta},{2,2}};
        _a020_aaaa = {{v_alpha,o_alpha,v_alpha,o_alpha},{2,2}};
        _a020_baab = {{v_beta,o_alpha,v_alpha,o_beta},{2,2}};
        _a020_baba = {{v_beta,o_alpha,v_beta,o_alpha},{2,2}};
        _a020_bbbb = {{v_beta,o_beta,v_beta,o_beta},{2,2}};
        _a022_abab = {{v_alpha,v_beta,v_alpha,v_beta},{2,2}};
        
        sch.allocate(_a007,_a001_aa,_a001_bb,_a017_aa,_a017_bb,
        _a006_aa,_a006_bb,_a009_aa,_a009_bb,_a021_aa,_a021_bb,_a008_aa,
        _a019_abab,_a020_aaaa,_a020_baba,
        _a020_baab,_a020_bbbb,_a022_abab
        );

        sch.execute();

        }

        sch
            (r1_aa() = 0)
            (r2_abab() = 0)
            (_a004_aaaa(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_aa_ov(h4_oa, p1_va, cind) * chol3d_aa_ov(h3_oa, p2_va, cind))
            .exact_copy(_a004_abab(p1_va, p1_vb, h3_oa, h3_ob), _a004_aaaa(p1_va, p1_vb, h3_oa, h3_ob))
            ;

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
               ((d_t1s[off])()  = t1_aa())
               ((d_t2s[off])()  = t2_abab())
               .execute();

            ccsd_e(/* ec,  */sch, MO, CI, d_e, t1_aa, t2_abab);
            ccsd_t1(/* ec,  */sch, MO, CI, r1_aa, t1_aa, t2_abab);
            ccsd_t2(/* ec,  */sch, MO, CI, r2_abab, t1_aa, t2_abab);

            #ifdef USE_TALSH
              sch.execute(ExecutionHW::GPU, profile);
            #else
              sch.execute();
            #endif

            std::tie(residual, energy) = rest_cs(ec, MO, r1_aa, r2_abab, t1_aa, t2_abab,
                                            d_e, d_r1_residual, d_r2_residual, p_evl_sorted,
                                            zshiftl, n_occ_alpha, n_vir_alpha);

            update_r2(ec, r2_abab());

            sch((d_r1s[off])() = r1_aa())
                ((d_r2s[off])() = r2_abab())
                .execute();

            const auto timer_end = std::chrono::high_resolution_clock::now();
            auto iter_time = std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

            iteration_print(sys_data, ec.pg(), iter, residual, energy, iter_time);

            if(writet && ( ((iter+1)%writet_iter == 0) || (residual < thresh) ) ) {
                write_to_disk(t1_aa,t1file);
                write_to_disk(t2_abab,t2file);
            }        

            if(residual < thresh) { 
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
        std::vector<Tensor<T>> next_t{t1_aa, t2_abab};
        diis<T>(ec, rs, ts, next_t);

    }

    //deallocate all intermediates 
    sch.deallocate(_a02, _a01_aa,_a03_aa_vo,_a04_aa,_a05_aa,_a05_bb);
    sch.deallocate(_a007,_a001_aa,_a001_bb,_a017_aa,_a017_bb,
        _a006_aa,_a006_bb,_a009_aa,_a009_bb,_a021_aa,_a021_bb,_a008_aa,
        _a019_abab,_a020_aaaa,_a020_baba,
        _a020_baab,_a020_bbbb,_a022_abab 
        ); //t2
    sch.deallocate(d_r1_residual, d_r2_residual);

    } //no restart
    else {
        ccsd_e(/* ec,  */sch, MO, CI, d_e, t1_aa, t2_abab);
        sch.execute();
        energy = get_scalar(d_e);
        residual = 0.0;
    }

    sys_data.ccsd_corr_energy  = energy;

    if(ec.pg().rank() == 0) {
        sys_data.results["output"]["CCSD"]["n_iterations"] =   niter+1;
        sys_data.results["output"]["CCSD"]["final_energy"]["correlation"] =  energy;
        sys_data.results["output"]["CCSD"]["final_energy"]["total"] =  sys_data.scf_energy+energy;
        write_json_data(sys_data,"CCSD");
    }

    sch.deallocate(i0_temp,t2_aaaa_temp,_a01,_a02_aa,_a03_aa);
    sch.deallocate(d_e,_a004_aaaa,_a004_abab);
    
    sch.deallocate(t2_aaaa, 
                    f1_aa_oo, f1_aa_ov, f1_aa_vv, f1_bb_oo, f1_bb_ov, f1_bb_vv,
                    chol3d_aa_oo, chol3d_aa_ov, chol3d_aa_vv,
                    chol3d_bb_oo, chol3d_bb_ov, chol3d_bb_vv).execute();

    return std::make_tuple(residual,energy);

}
