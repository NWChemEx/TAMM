#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>
#include <tamm/tamm.hpp>

using namespace tamm;

template<typename T>
OpList four_index_transform(const TiledIndexSpace& AO, const TiledIndexSpace& MSO, 
                            Tensor<T> tC, Tensor<T> tV) {
    TiledIndexLabel f1, f2, f3, f4;
    TiledIndexLabel p, q, r, s, E;

    std::tie(f1, f2, f3, f4) = AO.labels<4>("all");
    std::tie(p, q, r, s)     = MSO.labels<4>("all");

    Tensor<T> I0{AO, AO, AO, AO};
    Tensor<T> I1{MSO, AO, AO, AO};
    Tensor<T> I2{MSO, MSO, AO, AO};
    Tensor<T> I3{MSO, MSO, MSO, AO};

    return {I1(p, f2, f3, f4) = tC(f1, p) * I0(f1, f2, f3, f4),
            I2(p, r, f3, f4)  = tC(f2, r) * I1(p, f2, f3, f4),
            I3(p, r, q, f4)   = tC(f3, q) * I2(p, r, f3, f4),
            tV(p, r, q, s)    = tC(f4, s) * I3(p, r, q, f4)};
}

template<typename T>
OpList two_index_transform(const TiledIndexSpace& AO, const TiledIndexSpace& MSO, 
                           Tensor<T> tC, Tensor<T> tF_ao, Tensor<T> tF_mso) {
    TiledIndexLabel f1, f2;
    TiledIndexLabel p, q, E;

    std::tie(f1, f2) = AO.labels<2>("all");
    std::tie(p, q)   = MSO.labels<2>("all");

    Tensor<T> I0{MSO, AO};
    return {I0(p, f2)    = tC(f1, p) * tF_ao(f1, f2),
            tF_mso(p, q) = tC(f1, q) * I0(f1, p)};
}

TEST_CASE("2/4-Index Transform") {
    // Construction of tiled index space MSO and AO from scratch
    IndexSpace MSO_IS{range(0, 200),
                      {{"occ", {range(0, 100)}}, {"virt", {range(100, 200)}}}};
    IndexSpace AO_IS{range(0, 200)};

    TiledIndexSpace MSO{MSO_IS, 10};
    TiledIndexSpace AO{AO_IS, 10};

    const TiledIndexSpace& N_MSO = MSO("all");
    const TiledIndexSpace& N_AO  = AO("all");

    Tensor<double> tC{N_AO, N_MSO};
    Tensor<double> tF_ao{N_AO, N_AO};
    Tensor<double> tF_mso{N_MSO, N_MSO};
    Tensor<double> tV{N_MSO, N_MSO, N_MSO, N_MSO};

    //@todo construct tC
    //@todo construct tF_ao
    //@todo construct tF_mso
    //@todo construct tV

    auto two_index_transform_dag = make_dag(two_index_transform<double>, AO, MSO, tC, tF_ao, tF_mso);
    auto four_index_transform_dag = make_dag(four_index_transform<double>, AO, MSO, tC, tV);

    CHECK_NOTHROW(Scheduler::execute(two_index_transform_dag));
    CHECK_NOTHROW(Scheduler::execute(four_index_transform_dag));
}
