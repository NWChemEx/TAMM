// #define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_RUNNER
#include <catch/catch.hpp>

#include "ga-mpi.h"
#include "ga.h"
#include "macdecls.h"
#include "mpi.h"
#include "tamm/tamm.hpp"

using namespace tamm;

template<typename T>
std::ostream& operator << (std::ostream &os, std::vector<T>& vec){
    os << "[";
    for(auto &x: vec)
        os << x << ",";
    os << "]\n";
    return os;
}

template<typename T>
void print_tensor(Tensor<T> &t){
    auto lt = t();
    for (auto it: t.loop_nest())
    {
        auto blockid = internal::translate_blockid(it, lt);
        TAMM_SIZE size = t.block_size(blockid);
        std::vector<T> buf(size);
        t.get(blockid, buf);
        std::cout << "block" << blockid;
        for (TAMM_SIZE i = 0; i < size;i++)
         std::cout << buf[i] << " ";
        std::cout << std::endl;
    }

}


template<typename T>
void check_value(LabeledTensor<T> lt, T val) {
    LabelLoopNest loop_nest{lt.labels()};
    T ref_val = val;
    for(const auto& itval : loop_nest) {
        const IndexVector blockid = internal::translate_blockid(itval, lt);
        size_t size               = lt.tensor().block_size(blockid);
        std::vector<T> buf(size);
        lt.tensor().get(blockid, buf);
        if(lt.tensor().is_non_zero(blockid)){
            ref_val = val;
        }else {
            ref_val = (T)0;
        }
        for(TAMM_SIZE i = 0; i < size; i++) {
            REQUIRE(std::fabs(buf[i] - ref_val) < 1.0e-10);
        }
    }
}

template<typename T>
void check_value(Tensor<T>& t, T val) {
    check_value(t(), val);
}

template<typename T>
void tensor_contruction(const TiledIndexSpace& T_AO,
                        const TiledIndexSpace& T_MO,
                        const TiledIndexSpace& T_ATOM,
                        const TiledIndexSpace& T_AO_ATOM) {
    TiledIndexLabel A, r, s, mu, mu_A;

    A              = T_ATOM.label("all", 1);
    std::tie(r, s) = T_MO.labels<2>("all");
    mu             = T_AO.label("all", 1);
    mu_A           = T_AO_ATOM.label("all", 1);

    // Tensor Q{T_ATOM, T_MO, T_MO}, C{T_AO,T_MO}, SC{T_AO,T_MO};
    Tensor<T> Q{A, r, s}, C{mu, r}, SC{mu, s};

    Q(A, r, s) = 0.5 * C(mu_A(A), r) * SC(mu_A(A), s);
    Q(A, r, s) += 0.5 * C(mu_A(A), s) * SC(mu_A(A), r);
}

#if 0
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

TEST_CASE("Tensor Declaration Syntax") {
    using Tensor = Tensor<double>;

    {
        // Scalar value
        Tensor T{};
    }

    {
        // Vector of length 10
        IndexSpace is{range(10)};
        TiledIndexSpace tis{is};

        Tensor T{tis};
    }

    {
        // Matrix of size 10X20
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        TiledIndexSpace tis1{is1};
        TiledIndexSpace tis2{is2};

        Tensor T{tis1, tis2};
    }

    {
        // Matrix of size 10X20X30
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{range(30)};

        TiledIndexSpace tis1{is1}, tis2{is2}, tis3{is3};

        Tensor T{tis1, tis2, tis3};
    }

    {
        // Vector from two different subspaces
        IndexSpace is{range(10)};
        IndexSpace is1{is, range(0, 4)};
        IndexSpace is2{is, range(4, is.size())};

        IndexSpace is3{{is1, is2}};

        TiledIndexSpace tis{is3};

        Tensor T{tis};
    }

    {
        // Matrix with split rows -- subspaces of lengths 4 and 6
        IndexSpace is1{range(10)};
        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{{is2, is3}};
        IndexSpace is5{range(20)};

        TiledIndexSpace tis4{is4}, tis5{is5};

        Tensor T{tis4, tis5};
    }

    {
        // Matrix with split columns -- subspaces of lengths 12 and 8
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{is2, range(0, 12)};
        IndexSpace is4{is2, range(12, is2.size())};

        IndexSpace is5{{is3, is4}};

        TiledIndexSpace tis1{is1}, tis5{is5};

        Tensor T{tis1, tis5};
    }

    {
        // Matrix with split rows and columns
        IndexSpace is1{range(10)};
        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{{is2, is3}};

        IndexSpace is5{range(20)};
        IndexSpace is6{is5, range(0, 12)};
        IndexSpace is7{is5, range(12, is5.size())};

        IndexSpace is8{{is6, is7}};

        TiledIndexSpace tis4{is4}, tis8{is8};

        Tensor T{tis4, tis8};
    }

    {
        // Tensor with first dimension split -- subspaces of lengths 4 and 6
        IndexSpace is1{range(10)};
        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{{is2, is3}};
        IndexSpace is5{range(20)};
        IndexSpace is6{range(30)};

        TiledIndexSpace tis4{is4}, tis5{is5}, tis6{is6};

        Tensor T{tis4, tis5, tis6};
    }

    {
        // Tensor with second dimension split -- subspaces of lengths 12 and 8
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{is2, range(0, 12)};
        IndexSpace is4{is2, range(12, is2.size())};

        IndexSpace is5{{is3, is4}};
        IndexSpace is6{range(30)};

        TiledIndexSpace tis1{is1}, tis5{is5}, tis6{is6};

        Tensor T{tis1, tis5, tis6};
    }

    {
        // Tensor with third dimension split -- subspaces of lengths 13 and 17
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{range(30)};

        IndexSpace is4{is3, range(0, 13)};
        IndexSpace is5{is3, range(13, is3.size())};

        IndexSpace is6{{is4, is5}};

        TiledIndexSpace tis1{is1}, tis2{is2}, tis6{is6};

        Tensor T{tis1, tis2, tis6};
    }

    {
        // Tensor with first and second dimensions split
        IndexSpace is1{range(10)};
        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{{is2, is3}};

        IndexSpace is5{range(20)};
        IndexSpace is6{is5, range(0, 12)};
        IndexSpace is7{is5, range(12, is5.size())};

        IndexSpace is8{{is6, is7}};

        IndexSpace is9{range(30)};

        TiledIndexSpace tis4{is4}, tis8{is8}, tis9{is9};

        Tensor T{tis4, tis8, tis9};
    }

    {
        // Tensor with first and third dimensions split
        IndexSpace is1{range(10)};
        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{{is2, is3}};

        IndexSpace is5{range(20)};
        IndexSpace is6{range(30)};
        IndexSpace is7{is6, range(0, 13)};
        IndexSpace is8{is5, range(13, is6.size())};

        IndexSpace is9{{is7, is8}};

        TiledIndexSpace tis4{is4}, tis5{is5}, tis9{is9};

        Tensor T{tis4, tis5, tis9};
    }

    {
        // Tensor with second and third dimensions split
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{is2, range(0, 12)};
        IndexSpace is4{is2, range(12, is2.size())};

        IndexSpace is5{{is3, is4}};

        IndexSpace is6{range(30)};
        IndexSpace is7{is6, range(0, 13)};
        IndexSpace is8{is5, range(13, is6.size())};

        IndexSpace is9{{is7, is8}};

        TiledIndexSpace tis1{is1}, tis5{is5}, tis9{is9};

        Tensor T{tis1, tis5, tis9};
    }

    {
        // Tensor with first, second and third dimensions split
        IndexSpace is1{range(10)};
        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{{is2, is3}};

        IndexSpace is5{range(20)};
        IndexSpace is6{is5, range(0, 12)};
        IndexSpace is7{is5, range(12, is5.size())};

        IndexSpace is8{{is6, is7}};

        IndexSpace is9{range(30)};
        IndexSpace is10{is9, range(0, 13)};
        IndexSpace is11{is9, range(13, is9.size())};
        IndexSpace is12{{is10, is11}};

        TiledIndexSpace tis4{is4}, tis8{is8}, tis12{is12};

        Tensor T{tis4, tis8, tis12};
    }

    {
        // Vector with more than one split of subspaces
        IndexSpace is1{range(10)};

        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{is2, range(0, 1)};
        IndexSpace is5{is2, range(1, is2.size())};

        IndexSpace is{{is4, is5, is3}};
        TiledIndexSpace tis{is};
        Tensor T{tis};
    }

    {
        // Matrix with more than one split of first dimension
        IndexSpace is1{range(10)};

        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{is2, range(0, 1)};
        IndexSpace is5{is2, range(1, is2.size())};

        IndexSpace is6{{is4, is5, is3}};

        IndexSpace is7{range(20)};

        IndexSpace is8{is7, range(0, 12)};
        IndexSpace is9{is7, range(12, is7.size())};

        IndexSpace is10{{is8, is9}};

        TiledIndexSpace tis6{is6}, tis10{is10};
        Tensor T{tis6, tis10};
    }

    {
        // Vector with odd number elements from one space and even number
        // elements from another
        IndexSpace is1{range(0, 10, 2)};
        IndexSpace is2{range(1, 10, 2)};
        IndexSpace is{{is1, is2}};
        TiledIndexSpace tis{is};
        Tensor T{tis};
    }

    {
        // Matrix with odd rows from one space and even from another
        IndexSpace is1{range(0, 10, 2)};
        IndexSpace is2{range(1, 10, 2)};
        IndexSpace is3{{is1, is2}};

        IndexSpace is4{range(20)};

        TiledIndexSpace tis3{is3}, tis4{is4};
        Tensor T{tis3, tis4};
    }
}
#endif

TEST_CASE("Spin Tensor Construction") {
    using T = double;
    IndexSpace SpinIS{range(0, 20),
                      {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}},
                      {{Spin{1}, {range(0, 5), range(10, 15)}},
                       {Spin{2}, {range(5, 10), range(15, 20)}}}};

    IndexSpace IS{range(0, 20)};

    TiledIndexSpace SpinTIS{SpinIS, 5};
    TiledIndexSpace TIS{IS, 5};

    std::vector<SpinPosition> spin_mask_2D{SpinPosition::lower,
                                           SpinPosition::upper};

    TiledIndexLabel i, j, k, l;
    std::tie(i, j) = SpinTIS.labels<2>("all");
    std::tie(k, l) = TIS.labels<2>("all");

    bool failed = false;
    try {
        TiledIndexSpaceVec t_spaces{SpinTIS, SpinTIS};
        Tensor<T> tensor{t_spaces, spin_mask_2D};
    } catch(const std::string& e) {
        std::cerr << e << '\n';
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        IndexLabelVec t_lbls{i, j};
        Tensor<T> tensor{t_lbls, spin_mask_2D};
    } catch(const std::string& e) {
        std::cerr << e << '\n';
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{TIS, TIS};

        Tensor<T> tensor{t_spaces, spin_mask_2D};
    } catch(const std::string& e) {
        std::cerr << e << '\n';
        failed = true;
    }
    REQUIRE(!failed);
#if 0
    failed = false;
    try {
        IndexLabelVec t_lbls{k, l};
        Tensor<T> tensor{t_lbls, spin_mask_2D};
    } catch(const std::string& e) {
        std::cerr << e << '\n';
        failed = true;
    }
    REQUIRE(failed);

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{TIS, SpinTIS};
        Tensor<T> tensor{t_spaces, spin_mask_2D};
    } catch(const std::string& e) {
        std::cerr << e << '\n';
        failed = true;
    }
    REQUIRE(failed);

    failed = false;
    try {
        IndexLabelVec t_lbls{i, k};
        Tensor<T> tensor{t_lbls, spin_mask_2D};
    } catch(const std::string& e) {
        std::cerr << e << '\n';
        failed = true;
    }
    REQUIRE(failed);
#endif
    {
        REQUIRE((SpinTIS.spin(0) == Spin{1}));
        REQUIRE((SpinTIS.spin(1) == Spin{2}));
        REQUIRE((SpinTIS.spin(2) == Spin{1}));
        REQUIRE((SpinTIS.spin(3) == Spin{2}));

        REQUIRE((SpinTIS("occ").spin(0) == Spin{1}));
        REQUIRE((SpinTIS("occ").spin(1) == Spin{2}));

        REQUIRE((SpinTIS("virt").spin(0) == Spin{1}));
        REQUIRE((SpinTIS("virt").spin(1) == Spin{2}));

    }

    TiledIndexSpace tis_3{SpinIS, 3};

    {
        REQUIRE((tis_3.spin(0) == Spin{1}));
        REQUIRE((tis_3.spin(1) == Spin{1}));
        REQUIRE((tis_3.spin(2) == Spin{2}));
        REQUIRE((tis_3.spin(3) == Spin{2}));
        REQUIRE((tis_3.spin(4) == Spin{1}));
        REQUIRE((tis_3.spin(5) == Spin{1}));
        REQUIRE((tis_3.spin(6) == Spin{2}));
        REQUIRE((tis_3.spin(7) == Spin{2}));

        REQUIRE((tis_3("occ").spin(0) == Spin{1}));
        REQUIRE((tis_3("occ").spin(1) == Spin{1}));
        REQUIRE((tis_3("occ").spin(2) == Spin{2}));
        REQUIRE((tis_3("occ").spin(3) == Spin{2}));

        REQUIRE((tis_3("virt").spin(0) == Spin{1}));
        REQUIRE((tis_3("virt").spin(1) == Spin{1}));
        REQUIRE((tis_3("virt").spin(2) == Spin{2}));
        REQUIRE((tis_3("virt").spin(3) == Spin{2}));

    }

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> tensor{t_spaces, spin_mask_2D};
        tensor.allocate(ec);
        Scheduler{ec}(tensor() = 42).execute();
        check_value(tensor, (T)42);
        tensor.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {

        TiledIndexSpaceVec t_spaces{tis_3("occ"), tis_3("virt")};
        Tensor<T> tensor{t_spaces, spin_mask_2D};
        tensor.allocate(ec);

        Scheduler{ec}(tensor() = 42).execute();
        check_value(tensor, (T)42);
        tensor.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {

        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> T1{t_spaces, spin_mask_2D};
        Tensor<T> T2{t_spaces, spin_mask_2D};
        T1.allocate(ec);
        T2.allocate(ec);

        Scheduler{ec}(T2() = 3)(T1() = T2()).execute();
        check_value(T2, (T)3);
        check_value(T1, (T)3);
 
        T1.deallocate();
        T2.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {

        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> T1{t_spaces, {1,1}};
        Tensor<T> T2{t_spaces, {1,1}};
        T1.allocate(ec);
        T2.allocate(ec);

        Scheduler{ec}(T1() = 42)(T2() = 3)(T1() += T2()).execute();
        check_value(T2, (T)3);
        check_value(T1, (T)45);

        T1.deallocate();
        T2.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {

        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> T1{t_spaces, {1,1}};
        Tensor<T> T2{t_spaces, {1,1}};
        T1.allocate(ec);
        T2.allocate(ec);

        Scheduler{ec}(T1() = 42)(T2() = 3)(T1() += 2 * T2()).execute();
        check_value(T2, (T)3);
        check_value(T1, (T)48);
       
        T1.deallocate();
        T2.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {

        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> T1{t_spaces, {1,1}};
        Tensor<T> T2{t_spaces, {1,1}};
        Tensor<T> T3{t_spaces, {1,1}};

        T1.allocate(ec);
        T2.allocate(ec);
        T3.allocate(ec);        

        Scheduler{ec}(T1() = 42)(T2() = 3)(T3() = 4)(T1() += T3() * T2()).execute();
        check_value(T3, (T)4);
        check_value(T2, (T)3);
        check_value(T1, (T)54);
       
        T1.deallocate();
        T2.deallocate();
        T3.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int res = Catch::Session().run(argc, argv);
    GA_Terminate();
    MPI_Finalize();

    return res;
}