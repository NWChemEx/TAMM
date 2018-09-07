#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>
#include <tamm/tamm.hpp>

using tamm::ExecutionContext;
using tamm::IndexSpace;
using tamm::Scheduler;
using tamm::Tensor;
using tamm::TiledIndexLabel;
using tamm::TiledIndexSpace;
using tamm::range;

template<typename T>
void four_index_transform(ExecutionContext* ec, const TiledIndexSpace& AO,
                          const TiledIndexSpace& MSO, Tensor<T> tC,
                          Tensor<T> tV) {
    TiledIndexLabel f1, f2, f3, f4;
    TiledIndexLabel p, q, r, s, E;

    std::tie(f1, f2, f3, f4) = AO.labels<4>("all");
    std::tie(p, q, r, s)     = MSO.labels<4>("all");

    Tensor<T> I0{AO, AO, AO, AO};
    Tensor<T> I1{MSO, AO, AO, AO};
    Tensor<T> I2{MSO, MSO, AO, AO};
    Tensor<T> I3{MSO, MSO, MSO, AO};

    Scheduler{ec} //(I0(f1, f2, f2, f4) = integral_function())
      .allocate(I0, I1, I2,
                I3)(I1(p, f2, f3, f4) = tC(f1, p) * I0(f1, f2, f3, f4))(
        I2(p, r, f3, f4) = tC(f2, r) * I1(p, f2, f3, f4))(
        I3(p, r, q, f4) = tC(f3, q) * I2(p, r, f3, f4))(
        tV(p, r, q, s) = tC(f4, s) * I3(p, r, q, f4))
      .deallocate(I0, I1, I2, I3)
      .execute();
}

template<typename T>
void two_index_transform(ExecutionContext* ec, const TiledIndexSpace& AO,
                         const TiledIndexSpace& MSO, Tensor<T> tC,
                         Tensor<T> tF_ao, Tensor<T> tF_mso) {
    TiledIndexLabel f1, f2;
    TiledIndexLabel p, q, E;

    std::tie(f1, f2) = AO.labels<2>("all");
    std::tie(p, q)   = MSO.labels<2>("all");

    Tensor<T> I0{MSO, AO};
    Scheduler{ec}
      .allocate(I0)(I0(p, f2) = tC(f1, p) * tF_ao(f1, f2))(
        tF_mso(p, q) = tC(f1, q) * I0(p, f1))
      .deallocate(I0)
      .execute();
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

    using T = double;

    Tensor<T> tC{N_AO, N_MSO};
    Tensor<T> tF_ao{N_AO, N_AO};
    Tensor<T> tF_mso{N_MSO, N_MSO};
    Tensor<T> tV{N_MSO, N_MSO, N_MSO, N_MSO};

    //@todo construct tC
    //@todo construct tF_ao
    //@todo construct tF_mso
    //@todo construct tV
    ExecutionContext* ec = new ExecutionContext();

    CHECK_NOTHROW(two_index_transform<double>(ec, AO, MSO, tC, tF_ao, tF_mso));
    CHECK_NOTHROW(four_index_transform<double>(ec, AO, MSO, tC, tV));
}
