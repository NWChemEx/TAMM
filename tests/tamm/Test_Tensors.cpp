#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include <tamm/index_space.h>
#include <tamm/tensor_impl.h>
using namespace tamm;

template<typename T>
void tensor_contruction(const TiledIndexSpace& T_AO,
                        const TiledIndexSpace& T_MO,
                        const TiledIndexSpace& T_ATOM,
                        const TiledIndexSpace& T_AO_ATOM) {
    TiledIndexLabel A, r, s, mu, mu_A;

    A              = T_ATOM.label("all");
    std::tie(r, s) = T_MO.labels<2>("all");
    mu             = T_AO.label("all");
    mu_A           = T_AO_ATOM.label("all");

    // Tensor Q{T_ATOM, T_MO, T_MO}, C{T_AO,T_MO}, SC{T_AO,T_MO};
    Tensor<T> Q{A, r, s}, C{mu, r}, SC{mu, s};

    Q(A, r, s) = 0.5 * C(mu_A(A), r) * SC(mu_A(A), s);
    Q(A, r, s) += 0.5 * C(mu_A(A), s) * SC(mu_A(A), r);
}

TEST_CASE("Dependent Index construction and usage") {
    IndexSpace AO{range(0, 20)};
    IndexSpace MO{range(0, 40)};
    IndexSpace ATOM{{0, 1, 2, 3, 4}};

    std::map<IndexVector, IndexSpace> ao_atom_relation{
      /*atom 0*/ {IndexVector{0}, IndexSpace{AO, IndexVector{3, 4, 7}}},
      /*atom 1*/ {IndexVector{1}, IndexSpace{AO, IndexVector{1, 5, 7}}},
      /*atom 2*/ {IndexVector{2}, IndexSpace{AO, IndexVector{1, 9, 11}}},
      /*atom 3*/ {IndexVector{3}, IndexSpace{AO, IndexVector{11, 14}}},
      /*atom 4*/ {IndexVector{4}, IndexSpace{AO, IndexVector{2, 5, 13, 17}}}};

    IndexSpace AO_ATOM{/*dependent spaces*/ {ATOM},
                       /*reference space*/ AO,
                       /*relation*/ ao_atom_relation};

    TiledIndexSpace T_AO{AO}, T_MO{MO}, T_ATOM{ATOM}, T_AO_ATOM{AO_ATOM};

    CHECK_NOTHROW(tensor_contruction<double>(T_AO, T_MO, T_ATOM, T_AO_ATOM));
}