// #define CATCH_CONFIG_RUNNER

#include "diis.hpp"
#include "ccsd_t/ccsd_t_gpu.hpp"
// #include "macdecls.h"
// #include "ga-mpi.h"


using namespace tamm;

bool debug = false;

template<typename T>
void ccsd_e(/* ExecutionContext &ec, */
            Scheduler& sch,
            const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de, const Tensor<T>& t1,
            const Tensor<T>& t2, const Tensor<T>& f1, Tensor<T>& chol3d) { 

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");

    Tensor<T> _a01{CI};
    Tensor<T> _a02{{O,O,CI},{1,1}};
    Tensor<T> _a03{{O,V,CI},{1,1}};

    auto [cind] = CI.labels<1>("all");
    auto [p1, p2, p3, p4, p5] = MO.labels<5>("virt");
    auto [h3, h4, h5, h6]     = MO.labels<4>("occ");

    // Scheduler sch{ec};
    sch.allocate(_a01,_a02,_a03);

    sch 
        (_a01(cind) = t1(p3, h4) * chol3d(h4, p3,cind))
        (_a02(h4, h6, cind) = t1(p3, h4) * chol3d(h6, p3, cind))
        (de() =  0.5 * _a01() * _a01())
        (de() += -0.5 * _a02(h4, h6, cind) * _a02(h6, h4, cind))
        (_a03(h4, p2, cind) = t2(p1, p2, h3, h4) * chol3d(h3, p1, cind))
        (de() += 0.5 * _a03(h4, p1, cind) * chol3d(h4, p1, cind));
        ;
    
    sch.deallocate(_a01,_a02,_a03);
    // sch.execute();
}

template<typename T>
void ccsd_t1(/* ExecutionContext& ec,  */
             Scheduler& sch,
             const TiledIndexSpace& MO,const TiledIndexSpace& CI, 
             Tensor<T>& i0, const Tensor<T>& t1, const Tensor<T>& t2, 
             const Tensor<T>& f1, Tensor<T>& chol3d) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    
    Tensor<T> _a01{{O,O,CI},{1,1}};
    Tensor<T> _a02{CI};
    Tensor<T> _a03{{V,O,CI},{1,1}};
    Tensor<T> _a04{{O,O},{1,1}};
    Tensor<T> _a05{{O,V},{1,1}};
    Tensor<T> _a06{{O,O,CI},{1,1}};
    
    auto [cind] = CI.labels<1>("all");
    auto [p1, p2, p3, p4, p5, p6, p7, p8] = MO.labels<8>("virt");
    auto [h1, h2, h3, h4, h5, h6, h7, h8] = MO.labels<8>("occ");

    // Scheduler sch{ec};
    sch
        .allocate(_a01, _a02, _a03, _a04, _a05, _a06)
        (_a01() = 0)
        (_a02() = 0)
        (_a03() = 0)
        (_a04() = 0)
        (_a05() = 0)
        (_a06() = 0)
        (i0(p2, h1) = f1(p2, h1))
        (_a01(h2, h1, cind) +=  1.0 * t1(p1, h1) * chol3d(h2, p1, cind))         // ovm
        (_a02(cind)         +=  1.0 * t1(p3, h3) * chol3d(h3, p3, cind))         // ovm
        (_a03(p1, h1, cind) +=  1.0 * t2(p1, p3, h2, h1) * chol3d(h2, p3, cind)) // o2v2m
        (_a04(h2, h1)       +=  1.0 * chol3d(h2, p1, cind) * _a03(p1, h1, cind)) // o2vm
        (i0(p2, h1)         +=  1.0 * t1(p2, h2) * _a04(h2, h1))                 // o2v
        (i0(p1, h2)         +=  1.0 * chol3d(p1, h2, cind) * _a02(cind))         // ovm
        (_a05(h2, p1)       += -1.0 * chol3d(h3, p1, cind) * _a01(h2, h3, cind)) // o2vm
        (i0(p2, h1)         +=  1.0 * t2(p1, p2, h2, h1) * _a05(h2, p1))         // o2v
        (i0(p2, h1)         += -1.0 * chol3d(p2, p1, cind) * _a03(p1, h1, cind)) // ov2m
        (_a03(p2, h2, cind) += -1.0 * t1(p1, h2) * chol3d(p2, p1, cind))         // ov2m
        (i0(p1, h2)         += -1.0 * _a03(p1, h2, cind) * _a02(cind))           // ovm
        (_a03(p2, h3, cind) += -1.0 * t1(p2, h3) * _a02(cind))                   // ovm
        (_a03(p2, h3, cind) +=  1.0 * t1(p2, h2) * _a01(h2, h3, cind))           // o2vm
        (_a01(h3, h1, cind) +=  1.0 * chol3d(h3, h1, cind))                      // o2m
        (i0(p2, h1)         +=  1.0 * _a01(h3, h1, cind) * _a03(p2, h3, cind))   // o2vm
        (i0(p2, h1)         += -1.0 * t1(p2, h7) * f1(h7, h1))                 // o2v
        (i0(p2, h1)         +=  1.0 * t1(p3, h1) * f1(p2, p3))                   // ov2
        
        .deallocate(_a01, _a02, _a03, _a04, _a05, _a06);
        // .execute();
}

template<typename T>
void ccsd_t2(/* ExecutionContext& ec, */
             Scheduler& sch,
             const TiledIndexSpace& MO,const TiledIndexSpace& CI, 
             Tensor<T>& i0, const Tensor<T>& t1, Tensor<T>& t2, 
             const Tensor<T>& f1, Tensor<T>& chol3d, Tensor<T>& _a004) {
                 
    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");
    const TiledIndexSpace &N = MO("all");

    auto [cind] = CI.labels<1>("all");
    auto [p1, p2, p3, p4, p5, p6, p7, p8, p9] = MO.labels<9>("virt");
    auto [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11] = MO.labels<11>("occ");

    // Scheduler sch{ec};

    Tensor<T> _a001{{V,V}, {1,1}};
    Tensor<T> _a002{{V,V}, {1,1}};
    // Tensor<T> _a004{{V,V,O,O}, {2,2}};
    Tensor<T> _a006{{O,O}, {1,1}};
    Tensor<T> _a007{CI};
    Tensor<T> _a008{{O,O,CI}, {1,1}};
    Tensor<T> _a009{{O,O,CI}, {1,1}};
    Tensor<T> _a017{{V,O,CI}, {1,1}};
    Tensor<T> _a019{{O,O,O,O}, {2,2}};
    Tensor<T> _a020{{V,O,V,O}, {2,2}};
    Tensor<T> _a021{{V,V,CI}, {1,1}};
    Tensor<T> _a022{{V,V,V,V}, {2,2}};
    Tensor<T> i0_temp{{V,V,O,O}, {2,2}};

 //------------------------------CD------------------------------
    sch.allocate(_a001, _a002, _a006, _a007, 
                 _a008, _a009, _a017, _a019, _a020, _a021,
                 _a022, i0_temp); // _a004

    sch (_a001(p1, p2) = 0)
        (_a006(h3, h2) = 0)
        (_a019(h3, h4, h1, h2) = 0)
        (_a020(p3, h3, p4, h2) = 0)
        (i0(p3, p4, h1, h2) = 0)
        (i0_temp(p3, p4, h1, h2) = 0)
        ;
    
    sch (_a007(cind) = 0)
        (_a008(h3, h1, cind) = 0)
        (_a009(h3, h2, cind) = 0)
        (_a017(p3, h2, cind) = 0)
        (_a021(p3, p1, cind) = 0)
        (_a022(p3,p4,p2,p1) = 0)
        // (_a004(p1, p2, h4, h3) = 0)
        ;

    sch (_a017(p3, h2, cind) += -1.0 * t2(p1, p3, h3, h2) * chol3d(h3, p1, cind))
        (_a006(h4, h1) += -1.0 * chol3d(h4, p2, cind) * _a017(p2, h1, cind))
        (_a007(cind)     +=  1.0 * chol3d(h4, p1, cind) * t1(p1, h4))
        (_a009(h3, h2, cind) +=  1.0 * chol3d(h3, p1, cind) * t1(p1, h2))
        (_a021(p3, p1, cind) += -0.5 * chol3d(h3, p1, cind) * t1(p3, h3))
        (_a021(p3, p1, cind) +=  0.5 * chol3d(p3, p1, cind))
        (_a017(p3, h2, cind) += -2.0 * t1(p2, h2) * _a021(p3, p2, cind))
        (_a008(h3, h1, cind) +=  1.0 * _a009(h3, h1, cind))//t1
        (_a009(h3, h1, cind) +=  1.0 * chol3d(h3, h1, cind))
        ;
            
    sch (_a001(p4, p2) += -2.0 * _a021(p4, p2, cind) * _a007(cind))
        (_a001(p4, p2) += -1.0 * _a017(p4, h2, cind) * chol3d(h2, p2, cind))
        (_a006(h4, h1) +=  1.0 * _a009(h4, h1, cind) * _a007(cind))
        (_a006(h4, h1) += -1.0 * _a009(h3, h1, cind) * _a008(h4, h3, cind))
        (_a019(h4, h3, h1, h2) +=  0.25 * _a009(h4, h1, cind) * _a009(h3, h2, cind)) 
        (_a020(p4, h4, p1, h1) += -2.0  * _a009(h4, h1, cind) * _a021(p4, p1, cind))
        ;
        
    sch (_a017(p3, h2, cind) +=  1.0 * t1(p3, h3) * chol3d(h3, h2, cind))
        (_a017(p3, h2, cind) += -1.0 * chol3d(p3, h2, cind))
        (i0_temp(p3, p4, h1, h2) +=  0.5 * _a017(p3, h1, cind) * _a017(p4, h2, cind))
        ;
    
    sch (_a022(p3,p4,p2,p1) += _a021(p3,p2,cind) * _a021(p4,p1,cind))
        (i0_temp(p3, p4, h1, h2) += 1.0 * _a022(p3, p4, p2, p1) * t2(p2,p1,h1,h2))
        // (_a004(p1, p2, h4, h3) +=  1.0   * chol3d(p1, h4, cind) * chol3d(p2, h3, cind))
        (_a019(h3, h4, h1, h2) += -0.125 * _a004(p1, p2, h4, h3) * t2(p1,p2,h1,h2))
        (_a020(p1, h3, p4, h2) +=  0.5   * _a004(p2, p4, h3, h1) * t2(p1,p2,h1,h2)) 
        ;

    sch (_a001(p4, p1) += -1 * f1(p4, p1))
        (i0_temp(p3, p4, h1, h2) += -0.5 * t2(p3, p2, h1, h2) * _a001(p4, p2))
        (i0_temp(p3, p4, h1, h2) +=  1.0 * _a019(h4, h3, h1, h2) * t2(p3, p4, h4, h3))
        (i0_temp(p3, p4, h1, h2) +=  1.0 * _a020(p4, h4, p1, h1) * t2(p3, p1, h4, h2))
        ;

    sch (_a006(h9, h1) += f1(h9, h1))
        (_a006(h9, h1) += t1(p8, h1) * f1(h9, p8));

    sch (i0_temp(p3, p4, h2, h1) += -0.5 * t2(p3, p4, h3, h1) * _a006(h3, h2))
        (i0(p3, p4, h1, h2) +=  1.0 * i0_temp(p3, p4, h1, h2))
        (i0(p3, p4, h2, h1) += -1.0 * i0_temp(p3, p4, h1, h2))
        (i0(p4, p3, h1, h2) += -1.0 * i0_temp(p3, p4, h1, h2))
        (i0(p4, p3, h2, h1) +=  1.0 * i0_temp(p3, p4, h1, h2))    
        ;

    sch.deallocate(_a001, _a006, _a007, 
                 _a008, _a009, _a017, _a019, _a020, _a021,
                 _a022, i0_temp); //_a004
    //-----------------------------CD----------------------------------
    
    // sch.execute();

}


template<typename T>
std::tuple<double,double> ccsd_driver(ExecutionContext& ec, const TiledIndexSpace& MO,
                    const TiledIndexSpace& CI,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, 
                   Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s, 
                   std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s, 
                   std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted,
                   int maxiter, double thresh,
                   double zshiftl, int ndiis, 
                   const TAMM_SIZE& noab,
                   Tensor<T>& cv3d) {

    std::cout.precision(15);

    double residual = 0.0;
    double energy = 0.0;

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");
    auto [cind] = CI.labels<1>("all");
    auto [p1, p2] = MO.labels<2>("virt");
    auto [h3, h4] = MO.labels<2>("occ");

    Scheduler sch{ec};
    Tensor<T> _a004{{V,V,O,O}, {2,2}};
    sch.allocate(_a004)
    (_a004(p1, p2, h4, h3) = 1.0 * cv3d(p1, h4, cind) * cv3d(p2, h3, cind))
    .execute();

    for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
          const auto timer_start = std::chrono::high_resolution_clock::now();

          int off = iter - titer;

          Tensor<T> d_e{};
          Tensor<T> d_r1_residual{};
          Tensor<T> d_r2_residual{};

          Tensor<T>::allocate(&ec, d_e, d_r1_residual, d_r2_residual);

          sch(d_e() = 0)(d_r1_residual() = 0)(d_r2_residual() = 0)
             ((d_t1s[off])() = d_t1())((d_t2s[off])() = d_t2())
             .execute();

          ccsd_e(/* ec,  */sch, MO, CI, d_e, d_t1, d_t2, d_f1, cv3d);
          ccsd_t1(/* ec,  */sch, MO, CI, d_r1, d_t1, d_t2, d_f1, cv3d);
          ccsd_t2(/* ec,  */sch, MO, CI, d_r2, d_t1, d_t2, d_f1, cv3d, _a004);

          sch.execute();
        //   GA_Sync();
          std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_t1, d_t2,
                                            d_e, p_evl_sorted, zshiftl, noab);

          update_r2(ec, d_r2());

          sch((d_r1s[off])() = d_r1())((d_r2s[off])() = d_r2())
            .execute();

          const auto timer_end = std::chrono::high_resolution_clock::now();
          auto iter_time = std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

          iteration_print(ec.pg(), iter, residual, energy, iter_time);
          Tensor<T>::deallocate(d_e, d_r1_residual, d_r2_residual);

          if(residual < thresh) { break; }
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
  sch.deallocate(_a004).execute();

  return std::make_tuple(residual,energy);


}

void ccsd_driver();
std::string filename; //bad, but no choice

int main( int argc, char* argv[] )
{
    if(argc<2){
        std::cout << "Please provide an input file!\n";
        return 1;
    }

    filename = std::string(argv[1]);
    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!\n";
        return 1;
    }

    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    ccsd_driver();
    
    GA_Terminate();
    MPI_Finalize();

    return 0;
}


void ccsd_driver() {

    // std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    RuntimeEngine re;
    ExecutionContext ec{pg, &distribution, mgr, &re};
    auto rank = ec.pg().rank();

    //TODO: read from input file, assume no freezing for now
    TAMM_SIZE freeze_core    = 0;
    TAMM_SIZE freeze_virtual = 0;

    auto [options_map, ov_alpha, nao, hf_energy, shells, shell_tile_map, C_AO, F_AO, AO_opt, AO_tis] 
                    = hartree_fock_driver<T>(ec,filename);

    CCSDOptions ccsd_options = options_map.ccsd_options;
    debug = ccsd_options.debug;
    if(rank == 0) ccsd_options.print();

    int maxiter    = ccsd_options.maxiter;
    double thresh  = ccsd_options.threshold;
    double zshiftl = 0.0;
    size_t ndiis   = 5;

    auto [MO,total_orbitals] = setupMOIS(ccsd_options.tilesize,
                    nao,ov_alpha,freeze_core,freeze_virtual);

    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,chol_count, max_cvecs, CV] = cd_svd_ga_driver<T>
                        (options_map, ec, MO, AO_opt, ov_alpha, nao, freeze_core,
                                freeze_virtual, C_AO, F_AO, shells, shell_tile_map);

    TiledIndexSpace N = MO("all");

    auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s] 
            = setupTensors(ec,MO,d_f1,ndiis);

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    auto [residual, corr_energy] = ccsd_driver<T>(
            ec, MO, CV, d_t1, d_t2, d_f1, 
            d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
            p_evl_sorted, 
            maxiter, thresh, zshiftl, ndiis, 
            2 * ov_alpha, cholVpr);

    ccsd_stats(ec, hf_energy,residual,corr_energy,thresh);

    auto cc_t2 = std::chrono::high_resolution_clock::now();
    double ccsd_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) std::cout << "\nTime taken for Cholesky CCSD: " << ccsd_time << " secs\n";

    free_tensors(d_r1, d_r2, d_f1);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);

    ec.flush_and_sync();

    Tensor<T> d_v2 = setupV2<T>(ec,MO,cholVpr,chol_count, total_orbitals, ov_alpha, nao - ov_alpha);
    Tensor<T>::deallocate(cholVpr);

    cc_t1 = std::chrono::high_resolution_clock::now();

    auto n_alpha = ov_alpha;
    TAMM_SIZE n_beta{nao - ov_alpha};

    Index noab=MO("occ").num_tiles();
    Index nvab=MO("virt").num_tiles();
    std::vector<int> k_spin;
    for(auto x=0;x<noab/2;x++) k_spin.push_back(1);
    for(auto x=noab/2;x<noab;x++) k_spin.push_back(2);
    for(auto x=0;x<nvab/2;x++) k_spin.push_back(1);
    for(auto x=nvab/2;x<nvab;x++) k_spin.push_back(2);

    if(rank==0) cout << "\nCCSD(T)\n";

    auto [energy1,energy2] = ccsd_t_driver(ec,k_spin,n_alpha,n_beta,MO,d_t1,d_t2,d_v2,
                p_evl_sorted,hf_energy+corr_energy,ccsd_options.icuda);

    if (rank==0 && energy1!=-999){

        cout << "CCSD[T] correction energy / hartree  = " << energy1 << endl;
        cout << "CCSD[T] correlation energy / hartree = " << corr_energy+energy1 << endl;
        cout << "CCSD[T] total energy / hartree       = " << hf_energy+corr_energy+energy1 << endl;

        cout << "CCSD(T) correction energy / hartree  = " << energy2 << endl;
        cout << "CCSD(T) correlation energy / hartree = " << corr_energy+energy2 << endl;
        cout << "CCSD(T) total energy / hartree       = " << hf_energy+corr_energy+energy2 << endl;
    
    }

    free_tensors(d_t1, d_t2, d_v2);

    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    // delete ec;

    cc_t2 = std::chrono::high_resolution_clock::now();
    ccsd_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0 && energy1!=-999) std::cout << "\nTime taken for CCSD(T): " << ccsd_time << " secs\n";


}
