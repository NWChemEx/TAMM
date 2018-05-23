#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include <tamm/tamm.hpp>

using namespace tamm;

using Atom = std::tuple<int, double, double, double>; // libint2::Atom
using Contraction =
  std::tuple<int, bool, std::vector<double>>; // libint2::Shell::Contraction
using Shell =
  std::tuple<std::vector<double>, std::vector<Contraction>,
             std::array<double, 3>, std::vector<double>>; // libint2::Shell

auto one_body_overlap_integral_lambda = [](auto& val) {};
auto one_body_kinetic_integral_lambda = [](auto& val) {};
auto one_body_nuclear_integral_lambda = [](auto& val) {};

template<typename T>
void compute_soad(const std::vector<Atom>& atoms, Tensor<T>& D) {}
template<typename T>
void compute_2body_fock(const std::vector<Shell>& shells, const Tensor<T>& D,
                        Tensor<T>& F) {}
template<typename T>
auto eigen_solve(const Tensor<T>& F, const Tensor<T>& S) {
    return std::make_tuple(Tensor<T>{}, double{});
}

template<typename T>
void norm(const Tensor<T>& temp, const Tensor<T>& rmsd) {}

template<typename T>
double get_scalar(const Tensor<T>& tensor) {
    return double{};
}

template<typename TensorType>
void hartree_fock(ExecutionContext& ec, const TiledIndexSpace& AO,
                  Tensor<TensorType>& C, Tensor<TensorType>& F) {
    const std::vector<Atom> atoms;
    const std::vector<Shell> shells;

    Scheduler sch{ec};

    const TiledIndexSpace& N = AO("all");
    const TiledIndexSpace& O = AO("occ");

    TiledIndexLabel i, j, k;
    TiledIndexLabel io, jo, ko;

    std::tie(i, j, k)    = AO.labels<3>("all");
    std::tie(io, jo, ko) = AO.labels<3>("occ");

    Tensor<TensorType> S{N, N, one_body_overlap_integral_lambda};
    Tensor<TensorType> T{N, N, one_body_kinetic_integral_lambda};
    Tensor<TensorType> V{N, N, one_body_nuclear_integral_lambda};

    Tensor<TensorType> H{N, N};

    sch.allocate(H, T, V)(H(i, j) = T(i, j))(H(i, j) += V(i, j)).execute();

    Tensor<TensorType> D{N, N};

    compute_soad(atoms, D);

    const int maxiter{100};
    const double conv{1e-12};
    int iter{0};

    Tensor<TensorType> ehf, ediff, rmsd;
    Tensor<TensorType> eps{N, N};

    do {
        ++iter;
        // Save a copy of energy and the density
        Tensor<TensorType> ehf_last;
        Tensor<TensorType> D_last{N, N};

        sch
          .allocate(ehf_last,
                    D_last)(ehf_last() = ehf())(D_last(i, j) = D(i, j))
          .execute();

        // build a new Fock matrix
        sch(F(i, j) = H(i, j)).execute();

        compute_2body_fock(shells, D, F); // accumulate into F

        // solve F C = e S C
        // Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F,
        // S);  eps = gen_eig_solver(F,S).eigenvalues();  C =
        // gen_eig_solver(F,S).eigenvectors();
        double eps;
        std::tie(C, eps) = eigen_solve(F, S);

        // compute density, D = C(occ) . C(occ)T
        // C_occ(ao,bo) = C(ao,bo); //C.leftCols(ndocc);
        // C_occ_transpose(ao,bo) = C_occ(bo,ao);

        sch(D(io, jo) = C(io, ko) * C(ko, jo)).execute();

        Tensor<TensorType> tmp1{N, N}, tmp2{N, N};
        // compute HF energy
        // ehf() += D(i, j) * (H(i, j) + F(i, j));

        sch
          .allocate(tmp1, tmp2)(ehf() = 0.0)(tmp1(i, j) = H(i, j))(
            tmp1(i, j) += F(i, j))(ehf() = D(i, j) * tmp1(i, j))
          // compute difference with last iteration
          (ediff() = ehf())(ediff() += -1.0 * ehf_last())(tmp2(i, j) = D(i, j))(
            tmp2(i, j) += -1.0 * D_last(i, j))
          .execute();

        norm(tmp2, rmsd); // rmsd() = tmp2(a,b).norm();

        sch(rmsd() = tmp2(i, j) * tmp2(i, j)).execute();
        // e.g.:Tensor<TensorType> rmsd_local{AllocationModel::replicated};
        // e.g.:rmsd_local(a) = rmsd(a);
        // e.g.: rmsd(a) +=  rmsd_local(a);
        // TODO: only put rmsd_local in process 0 to rmsd
    } while(
      ((fabs(get_scalar(ediff) > conv) || (fabs(get_scalar(rmsd)) > conv)) &&
       (iter < maxiter)));
}

TEST_CASE("HartreeFock testcase") {
    // Construction of tiled index space AO from scratch
    IndexSpace AO_IS{range(0, 200),
                     {{"occ", {range(0, 100)}}, {"virt", {range(100, 200)}}}};

    TiledIndexSpace AO{AO_IS, 10};

    const TiledIndexSpace& N = AO("all");
    const TiledIndexSpace& O = AO("occ");

    Tensor<double> C{O, O};
    Tensor<double> F{N, N};

    //@todo construct C
    //@todo construct F

    ExecutionContext ec;

    CHECK_NOTHROW(hartree_fock<double>(ec, AO, C, F));
}