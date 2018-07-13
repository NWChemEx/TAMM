
#include "ga.h"
#include "mpi.h"
#include "macdecls.h"
#include "ga-mpi.h"
#include "tamm/tamm.hpp"

using namespace tamm;

template<typename T>
OpList ccsd_e(const TiledIndexSpace& MO, Tensor<T>& de, const Tensor<T>& t1,
              const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> i1{O, V};

    return {i1("h6", "p5") = f1("h6", "p5"),
            i1("h6", "p5") += 0.5 * t1("p3", "h4") * v2("h4", "h6", "p3", "p5"), 
            de() = 0,
            de() += t1("p5", "h6") * i1("h6", "p5"),
            de() += 0.25 * t2("p1", "p2", "h3", "h4") * v2("h3", "h4", "p1", "p2")};
}

template<typename T>
OpList ccsd_t1(const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, const Tensor<T>& t2, 
             const Tensor<T>& f1, const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> t1_2_1{O, O};
    Tensor<T> t1_2_2_1{O, V};
    Tensor<T> t1_3_1{V, V};
    Tensor<T> t1_5_1{O, V};
    Tensor<T> t1_6_1{O, O, V, V};

    return {i0("p2", "h1") = f1("p2", "h1"),
            t1_2_1("h7", "h1") = f1("h7", "h1"),
            t1_2_2_1("h7", "p3") = f1("h7", "p3"),
            t1_2_2_1("h7", "p3") += -1.0 * t1("p5", "h6") * v2("h6", "h7", "p3", "p5"),
            t1_2_1("h7", "h1") += t1("p3", "h1") * t1_2_2_1("h7", "p3"),
            t1_2_1("h7", "h1") += -1.0 * t1("p4", "h5") * v2("h5", "h7", "h1", "p4"),
            t1_2_1("h7", "h1") += -0.5 * t2("p3", "p4", "h1", "h5") * v2("h5", "h7", "p3", "p4"),
            i0("p2", "h1") += -1.0 * t1("p2", "h7") * t1_2_1("h7", "h1"),
            t1_3_1("p2", "p3") = f1("p2", "p3"),
            t1_3_1("p2", "p3") += -1.0 * t1("p4", "h5") * v2("h5", "p2", "p3", "p4"),
            i0("p2", "h1") += t1("p3", "h1") * t1_3_1("p2", "p3"),
            i0("p2", "h1") += -1.0 * t1("p3", "h4") * v2("h4", "p2", "h1", "p3"),
            t1_5_1("h8", "p7") = f1("h8", "p7"),
            t1_5_1("h8", "p7") += t1("p5", "h6") * v2("h6", "h8", "p5", "p7"),
            i0("p2", "h1") += t2("p2", "p7", "h1", "h8") * t1_5_1("h8", "p7"),
            t1_6_1("h4", "h5", "h1", "p3") = v2("h4", "h5", "h1", "p3"),
            t1_6_1("h4", "h5", "h1", "p3") += -1.0 * t1("p6", "h1") * v2("h4", "h5", "p3", "p6"), 
            i0("p2", "h1") += -0.5 * t2("p2", "p3", "h4", "h5") * t1_6_1("h4", "h5", "h1", "p3"),
            i0("p2", "h1") += -0.5 * t2("p3", "p4", "h1", "h5") * v2("h5", "p2", "p3", "p4")};
}

template<typename T>
OpList ccsd_t2(const TiledIndexSpace& MO, Tensor<T>& i0,
               const Tensor<T>& t1, Tensor<T>& t2, 
               const Tensor<T>& f1, const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");

    Tensor<T> t2_2_1{O, V, O, O};
    Tensor<T> t2_2_2_1{O, O, O, O};
    Tensor<T> t2_2_2_2_1{O, O, O, V};
    Tensor<T> t2_2_4_1{O, V};
    Tensor<T> t2_2_5_1{O, O, O, V};
    Tensor<T> t2_4_1{O, O};
    Tensor<T> t2_4_2_1{O, V};
    Tensor<T> t2_5_1{V, V};
    Tensor<T> t2_6_1{O, O, O, O};
    Tensor<T> t2_6_2_1{O, O, O, V};
    Tensor<T> t2_7_1{O, V, O, V};
    Tensor<T> vt1t1_1{O, V, O, O};


    return {i0("p3", "p4", "h1", "h2") = v2("p3", "p4", "h1", "h2"),
            t2_2_1("h10", "p3", "h1", "h2") = v2("h10", "p3", "h1", "h2"),
            t2_2_2_1("h10", "h11", "h1", "h2") = -1.0 * v2("h10", "h11", "h1", "h2"),
            t2_2_2_2_1("h10", "h11", "h1", "p5") = v2("h10", "h11", "h1", "p5"),
            t2_2_2_2_1("h10", "h11", "h1", "p5") += -0.5 * t1("p6", "h1") * v2("h10", "h11", "p5", "p6"),
            t2_2_2_1("h10", "h11", "h1", "h2") += t1("p5", "h1") * t2_2_2_2_1("h10", "h11", "h2", "p5"),
            t2_2_2_1("h10", "h11", "h1", "h2") += -0.5 * t2("p7", "p8", "h1", "h2") * v2("h10", "h11", "p7", "p8"),
            t2_2_1("h10", "p3", "h1", "h2") += 0.5 * t1("p3", "h11") * t2_2_2_1("h10", "h11", "h1", "h2"),
            t2_2_4_1("h10", "p5") = f1("h10", "p5"),
            t2_2_4_1("h10", "p5") += -1.0 * t1("p6", "h7") * v2("h7", "h10", "p5", "p6"),
            t2_2_1("h10", "p3", "h1", "h2") += -1.0 * t2("p3", "p5", "h1", "h2") * t2_2_4_1("h10", "p5"),
            t2_2_5_1("h7", "h10", "h1", "p9") = v2("h7", "h10", "h1", "p9"),
            t2_2_5_1("h7", "h10", "h1", "p9") += t1("p5", "h1") * v2("h7", "h10", "p5", "p9"),
            t2_2_1("h10", "p3", "h1", "h2") += t2("p3", "p9", "h1", "h7") * t2_2_5_1("h7", "h10", "h2", "p9"),
            t2("p1", "p2", "h3", "h4") += 0.5 * t1("p1", "h3") * t1("p2", "h4"),
            t2_2_1("h10", "p3", "h1", "h2") += 0.5 * t2("p5", "p6", "h1", "h2") * v2("h10", "p3", "p5", "p6"),
            t2("p1", "p2", "h3", "h4") += -0.5 * t1("p1", "h3") * t1("p2", "h4"),
            i0("p3", "p4", "h1", "h2") += -1.0 * t1("p3", "h10") * t2_2_1("h10", "p4", "h1", "h2"),
            i0("p3", "p4", "h1", "h2") += -1.0 * t1("p5", "h1") * v2("p3", "p4", "h2", "p5"),
            t2_4_1("h9", "h1") = f1("h9", "h1"),
            t2_4_2_1("h9", "p8") = f1("h9", "p8"),
            t2_4_2_1("h9", "p8") += t1("p6", "h7") * v2("h7", "h9", "p6", "p8"),
            t2_4_1("h9", "h1") += t1("p8", "h1") * t2_4_2_1("h9", "p8"),
            t2_4_1("h9", "h1") += -1.0 * t1("p6", "h7") * v2("h7", "h9", "h1", "p6"),
            t2_4_1("h9", "h1") += -0.5 * t2("p6", "p7", "h1", "h8") * v2("h8", "h9", "p6", "p7"),
            i0("p3", "p4", "h1", "h2") += -1.0 * t2("p3", "p4", "h1", "h9") * t2_4_1("h9", "h2"),
            t2_5_1("p3", "p5") = f1("p3", "p5"),
            t2_5_1("p3", "p5") += -1.0 * t1("p6", "h7") * v2("h7", "p3", "p5", "p6"),
            t2_5_1("p3", "p5") += -0.5 * t2("p3", "p6", "h7", "h8") * v2("h7", "h8", "p5", "p6"),
            i0("p3", "p4", "h1", "h2") += 1 * t2("p3", "p5", "h1", "h2") * t2_5_1("p4", "p5"),
            t2_6_1("h9", "h11", "h1", "h2") = -1 * v2("h9", "h11", "h1", "h2"),
            t2_6_2_1("h9", "h11", "h1", "p8") = v2("h9", "h11", "h1", "p8"),
            t2_6_2_1("h9", "h11", "h1", "p8") += 0.5 * t1("p6", "h1") * v2("h9", "h11", "p6", "p8"),
            t2_6_1("h9", "h11", "h1", "h2") += t1("p8", "h1") * t2_6_2_1("h9", "h11", "h2", "p8"),
            t2_6_1("h9", "h11", "h1", "h2") += -0.5 * t2("p5", "p6", "h1", "h2") * v2("h9", "h11", "p5", "p6"),
            i0("p3", "p4", "h1", "h2") += -0.5 * t2("p3", "p4", "h9", "h11") *t2_6_1("h9", "h11", "h1", "h2"),
            t2_7_1("h6", "p3", "h1", "p5") = v2("h6", "p3", "h1", "p5"),
            t2_7_1("h6", "p3", "h1", "p5") += -1.0 * t1("p7", "h1") * v2("h6", "p3", "p5", "p7"),
            t2_7_1("h6", "p3", "h1", "p5") += -0.5 * t2("p3", "p7", "h1", "h8") * v2("h6", "h8", "p5", "p7"),
            i0("p3", "p4", "h1", "h2") += -1 * t2("p3", "p5", "h1", "h6") * t2_7_1("h6", "p4", "h2", "p5"),
            vt1t1_1("h5", "p3", "h1", "h2") = 0,
            vt1t1_1("h5", "p3", "h1", "h2") += -2.0 * t1("p6", "h1") * v2("h5", "p3", "h2", "p6"),
            i0("p3", "p4", "h1", "h2") += -0.5 * t1("p3", "h5") * vt1t1_1("h5", "p4", "h1", "h2"),
            t2("p1", "p2", "h3", "h4") += 0.5 * t1("p1", "h3") * t1("p2", "h4"),
            i0("p3", "p4", "h1", "h2") += 0.5 * t2("p5", "p6", "h1", "h2") * v2("p3", "p4", "p5", "p6"),
            t2("p1", "p2", "h3", "h4") += -0.5 * t1("p1", "h3") * t1("p2", "h4")
            };
}
// class Scheduler;
template<typename T>
void jacobi(Scheduler& sch, const Tensor<T>& d_r, const Tensor<T>& d_t, T shift,
            bool transpose, const Tensor<T>& EVL) {
    //@todo implement
}

/**
 *
 * @tparam T
 * @param MO
 * @param p_evl_sorted
 * @return pair of residual and energy
 */
template<typename T>
std::pair<double, double> rest(ExecutionContext& ec, const TiledIndexSpace& MO,
                               const Tensor<T>& d_r1, const Tensor<T>& d_r2,
                               const Tensor<T>& d_t1, const Tensor<T>& d_t2,
                               const Tensor<T>& de, const Tensor<T>& EVL,
                               T zshiftl) {
    T residual, energy;
    Scheduler sch{ec};
    Tensor<T> d_r1_residual{}, d_r2_residual{};
    sch
      .allocate(d_r1_residual, d_r2_residual)
      (d_r1_residual() = d_r1() * d_r1())
      (d_r2_residual() = d_r2() * d_r2())
      ([&](Scheduler& sch) {
          T r1, r2;
          d_r1_residual.get({}, span<T>(&r1, sizeof(T)));
          d_r2_residual.get({}, span<T>(&r2, sizeof(T)));
          residual = std::max(0.5 * std::sqrt(r1), 0.5 * std::sqrt(r2));
          de.get({}, span<T>(&energy, sizeof(T)));
      })
      ([&](Scheduler& sch) {
          jacobi(sch, d_r1, d_t1, -1.0 * zshiftl, false, EVL);
      })
      ([&](Scheduler& sch) {
          jacobi(sch, d_r2, d_t2, -2.0 * zshiftl, false, EVL);
      })
      .deallocate(d_r1_residual, d_r2_residual)
      .execute();
    ;
    return {residual, energy};
}

template<typename T>
void ccsd_driver(const TiledIndexSpace& MO, const Tensor<T>& d_f1,
                 const Tensor<T>& d_v2, double threshold) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");

    Tensor<T> de{};
    Tensor<T> i1{};
    Tensor<T> i2{};

    // Tensor<T> i0{V,O};
    Tensor<T> d_t1{V, O};
    Tensor<T> d_t2{V, V, O, O};

    //@todo initial t1 guess
    //@todo initial t2 guess

    //ProcGroup pg{MPI_COMM_SELF};
    //auto mgr = MemoryManagerLocal::create_coll(pg);
    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext ec{pg,&distribution,mgr};

    TiledIndexSpace UnitTiledMO{MO.index_space(), 1};
    Tensor<T> d_evl{N};
    //@todo Set EVL to have local distribution (one copy in each MPI rank)
    Tensor<T>::allocate(ec, d_evl);

    //auto ecsp = std::shared_ptr<ExecutionContext>(&ec);
  
    Scheduler{ec}
        (d_evl("n1") = 0.0)
        .execute();

    Tensor<T>::allocate(ec, d_t1, d_t2);

    T energy        = 0.0;
    T residual      = 1000 /*some large number*/;
    const T zshiftl = 0.0;

    auto ccsd_e_dag = make_dag(ccsd_e<T>, MO, de, d_t1, d_t2, d_f1, d_v2);
    auto ccsd_t1_dag = make_dag(ccsd_t1<T>, MO, i1, d_t1, d_t2, d_f1, d_v2);
    auto ccsd_t2_dag = make_dag(ccsd_t2<T>, MO, i2, d_t1, d_t2, d_f1, d_v2);

    while(residual > threshold) {
        Scheduler::execute(ccsd_e_dag);
        Scheduler::execute(ccsd_t1_dag);
        Scheduler::execute(ccsd_t2_dag);
        std::tie(residual, energy) =
          rest(ec, MO, i1, i2, d_t1, d_t2, de, d_evl, zshiftl);
        break; //@todo remove once iterative procedure is implemented
    }
    Tensor<T>::deallocate(d_evl, d_t1, d_t2);
}

int main( int argc, char* argv[] )
{
    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Construction of tiled index space MO from sketch
    IndexSpace MO_IS{range(0, 200),
                     {{"occ", {range(0, 100)}}, {"virt", {range(100, 200)}}}};
    TiledIndexSpace MO{MO_IS, 10};

    const TiledIndexSpace& N = MO("all");

    Tensor<double> f1{N, N};
    Tensor<double> v2{N, N, N, N};

    //@todo construct f1
    //@todo construct v2

    ccsd_driver<double>(MO, f1, v2, 1e-10);
    GA_Terminate();
    MPI_Finalize();
}
