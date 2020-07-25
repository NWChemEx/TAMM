#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <tamm/index_space.hpp>
#include <tamm/tiled_index_space.hpp>
using namespace tamm;

void check_indices(IndexSpace is, IndexVector iv) {
    int i = 0;
    REQUIRE(is.num_indices() == iv.size());
    for(const auto& index : is) { REQUIRE(index == iv[i++]); }
}

TEST_CASE("/* Test Case Description */") {
    IndexSpace is1{range(10)};
    IndexSpace is2{range(0,10)};
    IndexSpace is3{range(0,20)};
    IndexSpace is4{range(0,20,2)};
    IndexSpace is5{range(10),
                   {{"occ", {range(0, 5)}},
                    {"virt", {range(5, 10)}}}};

    IndexSpace is6{is3, range(10)};
    IndexSpace is7{{is5("occ"), is5("virt")}};

    REQUIRE(is1 == is1);
    if(is1 == is1){
        std::cout << "self-comparison" << std::endl;
    }

    REQUIRE(is1 == is2);
    if (is1 == is2) {
        std::cout << "is1 == is2" << std::endl;
    }

    REQUIRE(is1 != is3);
    if (is1 != is3) {
        std::cout << "is1 != is2" << std::endl;
    }   

    REQUIRE(is1 != is4);
    if (is1 != is4) {
        std::cout << "is1 != is4" << std::endl;
    }

    REQUIRE(is3 != is4);
    if (is3 != is4) {
        std::cout << "is3 != is4" << std::endl;
    }

    REQUIRE(is1 == is5);
    if (is1 == is5) {
        std::cout << "is1 == is5" << std::endl;
    }

    REQUIRE(is1 == is6);
    if (is1 == is6) {
        std::cout << "is1 == is6" << std::endl;
    }

    REQUIRE(is1 == is7);
    if (is1 == is7) {
        std::cout << "is1 == is7" << std::endl;
    }

    REQUIRE(is2 == is5);
    if (is2 == is5) {
        std::cout << "is2 == is5" << std::endl;
    }

    REQUIRE(is2 == is6);
    if (is2 == is6) {
        std::cout << "is2 == is6" << std::endl;
    }

    REQUIRE(is2 == is7);
    if (is2 == is7) {
        std::cout << "is2 == is7" << std::endl;
    }

    REQUIRE(is5 == is5);
    if (is5 == is5) {
        std::cout << "is5 == is5" << std::endl;
    }

    REQUIRE(is5 == is6);
    if (is5 == is6) {
        std::cout << "is5 == is6" << std::endl;
    }

    REQUIRE(is5 == is7);
    if (is5 == is7) {
        std::cout << "is5 == is7" << std::endl;
    }

    REQUIRE(is6 == is7);
    if (is6 == is7) {
        std::cout << "is6 == is7" << std::endl;
    }

}

TEST_CASE("IndexSpace construction with ranges ") {
    IndexSpace is{range(10)};
    IndexVector iv{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // check if the indices are equal
    check_indices(is, iv);
    // check subspace named "all"
    check_indices(is("all"), iv);

    IndexSpace is1{range(10, 20)};
    IndexVector iv1{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    // check if the indices are equal
    check_indices(is1, iv1);
    // check subspace named "all"
    check_indices(is1("all"), iv1);

    IndexSpace is2{range(10, 20, 2)};
    IndexVector iv2{10, 12, 14, 16, 18};
    // check if the indices are equal
    check_indices(is2, iv2);
    // check subspace named "all"
    check_indices(is2("all"), iv2);
}

TEST_CASE("IndexSpace construction with set of indices") {
    IndexSpace is1{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    IndexVector iv1{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    // check if the indices are equal
    check_indices(is1, iv1);
    // check subspace named "all"
    check_indices(is1("all"), iv1);
}

TEST_CASE("IndexSpace construction with named subspaces") {
    IndexSpace is2{{10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                   {{"occ", {range(0, 5)}},
                    {"virt", {range(5, 10)}},
                    {"alpha", {range(0, 3), range(5, 8)}},
                    {"beta", {range(3, 5), range(8, 10)}}}};

    // check indices for full space
    check_indices(is2, {10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
    // check indices for subspace named "all"
    check_indices(is2("all"), {10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
    // check indices for subspace named "occ"
    check_indices(is2("occ"), {10, 11, 12, 13, 14});
    // check indices for subspace named "virt"
    check_indices(is2("virt"), {15, 16, 17, 18, 19});
    // check indices for subspace named "alpha"
    check_indices(is2("alpha"), {10, 11, 12, 15, 16, 17});
    // check indices for subspace named "beta"
    check_indices(is2("beta"), {13, 14, 18, 19});
}

TEST_CASE("IndexSpace construction by range with named subspaces"){

    IndexSpace is{range(10),
        {{"occ",  {range(2, 5)}},
        {"virt", {range(4, 9)}}
        }};
    // check indices for full space
    check_indices(is, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    // check indices for subspace named all
    check_indices(is("all"), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    // check indices for subspace named occ
    check_indices(is("occ"), {2, 3, 4});
    // check indices for subspace named virt
    check_indices(is("virt"), {4, 5, 6, 7, 8});
}

TEST_CASE("Retrieval of a point in IndexSpace") {
    IndexSpace is{12, 11, 14, 24, 9, 8, 7};

    Index i = 24;
    // check ith indice of IndexSpace
    REQUIRE(is.index(Index{3}) == i);

    i = 8;
    // check ith indice of IndexSpace
    REQUIRE(is[Index{5}] == i);
} 

TEST_CASE("IndexSpace construction by concatenation of other disjoint IndexSpaces") {

    IndexSpace is1{2, 4, 5, 6, 7, 8};
    IndexSpace is2{3, 7, 9};

    IndexSpace is{{is1, is2}};

    // check if the indices are equal
    IndexVector is3{2, 4, 5, 6, 7, 8, 3, 7, 9};
    check_indices(is, is3);
}
TEST_CASE("IndexSpace construction with named subspaces by concatenation of other disjoint indexspaces") {

    IndexSpace is1{2, 4, 5};
    IndexSpace is2{1, 3};
    IndexSpace is3{3, 6};



    IndexSpace is{{is1, is2, is3}, 
        //Not clear of required named subspace for each indexSpace is1->temp1, is2->temp2, is3->temp3
         {"temp1", "temp2", "temp3"},
        {{"occ",   {range(2, 3)}},
         {"virt",  {range(1, 4)}},
         {"alpha", {range(0, 7, 2)}},
         {"beta",  {range(1, 7, 2)}}}};

    // check indices for full space
    check_indices(is, {2, 4, 5, 1, 3, 3, 6});
    // check indices for subspace named all
    check_indices(is("all"), {2, 4, 5, 1, 3, 3, 6});
    // check indices for subspace named occ
    check_indices(is("occ"), {5});
    // check indices for subspace named virt
    check_indices(is("virt"), {4, 5, 1});
    // check indices for subspace named alpha
    check_indices(is("alpha"), {2, 5, 3, 6});
    // check indices for subspace named beta
    check_indices(is("beta"), {4, 1, 3});
}

TEST_CASE("IndexSpace construction by subspacing") {
    IndexSpace is{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    IndexSpace sub_is{is, range(2, 8)};

    // check indices for created (sub)IndexSpace
    check_indices(sub_is, {7, 6, 5, 4, 3, 2});
    // check indices for subspace name "all"
    check_indices(sub_is("all"), {7, 6, 5, 4, 3, 2});
}

TEST_CASE("IndexSpace construction by subspacing with named subspaces") {
    IndexSpace is{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    IndexSpace sub_is{
      is, range(2, 8), {{"occ", {range(0, 3)}}, {"virt", {range(3, 6)}}}};

    // check indices for created (sub)IndexSpace
    check_indices(sub_is, {7, 6, 5, 4, 3, 2});
    // check indices for subspace name "all"
    check_indices(sub_is("all"), {7, 6, 5, 4, 3, 2});
    // check indices for subspace name "occ"
    check_indices(sub_is("occ"), {7, 6, 5});
    // check indices for subspace name "virt"
    check_indices(sub_is("virt"), {4, 3, 2});
}

TEST_CASE("IndexSpace construction by aggregating with other index spaces") {
    IndexSpace temp_is1{{10, 12, 14, 16, 18}};
    IndexSpace temp_is2{{1, 3, 5, 7, 9}};

    IndexSpace agg_is{{temp_is1, temp_is2}};
    // check indices for created (aggregated)IndexSpace
    check_indices(agg_is, {10, 12, 14, 16, 18, 1, 3, 5, 7, 9});
    // check indices for subspace name "all"
    check_indices(agg_is("all"), {10, 12, 14, 16, 18, 1, 3, 5, 7, 9});
}

TEST_CASE("IndexSpace construction by aggregating with subnames") {
    IndexSpace temp_is{{10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                       {{"occ", {range(0, 5)}},
                        {"virt", {range(5, 10)}},
                        {"alpha", {range(0, 3), range(5, 8)}},
                        {"beta", {range(3, 5), range(8, 10)}}}};

    IndexSpace temp_is1{range(20, 30),
                        {{"occ", {range(0, 5)}},
                         {"virt", {range(5, 10)}},
                         {"alpha", {range(0, 3), range(5, 8)}},
                         {"beta", {range(3, 5), range(8, 10)}}}};

    IndexSpace agg_is{{temp_is, temp_is1},
                      {"occ", "virt"},
                      {{"local", {range(8, 13)}}},
                      {{"alpha", {"occ:alpha", "virt:alpha"}},
                       {"beta", {"occ:beta", "virt:beta"}}}};

    IndexVector full_indices = construct_index_vector(range(10, 30));
    // check indices for created (sub)IndexSpace
    check_indices(agg_is, full_indices);
    // check indices for subspace name "all"
    check_indices(agg_is("all"), full_indices);
    // check indices for subspace name "occ"
    check_indices(agg_is("occ"), construct_index_vector(range(10, 20)));
    // check indices for subspace name "virt"
    check_indices(agg_is("virt"), construct_index_vector(range(20, 30)));
    // check indices for subspace name "local"
    check_indices(agg_is("local"), {18, 19, 20, 21, 22});
    // check indices for subspace name "alpha"
    check_indices(agg_is("alpha"),
                  {10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27});
    // check indices for subspace name "beta"
    check_indices(agg_is("beta"), {13, 14, 18, 19, 23, 24, 28, 29});
}

TEST_CASE("IndexSpace construction using dependent IndexSpaces") {
    IndexSpace is1{range(10, 20)};
    IndexSpace is2{range(10, 20, 2)};
    IndexSpace is3{{10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                   {{"occ", {range(0, 5)}},
                    {"virt", {range(5, 10)}},
                    {"alpha", {range(0, 3), range(5, 8)}},
                    {"beta", {range(3, 5), range(8, 10)}}}};

    IndexSpace temp_is1{0, 1, 2};
    IndexSpace temp_is2{8, 9};

    TiledIndexSpace t_is1{temp_is1, 2};
    TiledIndexSpace t_is2{temp_is2, 1};

    std::map<IndexVector, IndexSpace> dep_relation{
      {{0, 0}, is1}, {{0, 1}, is2}, {{1, 0}, temp_is2},
      {{1, 1}, is2}};

    // Dependent IndexSpace construction using dependency relation
    IndexSpace dep_is{{t_is1, t_is2}, dep_relation};

    // Check the dependency relation
    REQUIRE(dep_is(IndexVector{0, 0}) == is1);
    REQUIRE(dep_is(IndexVector{0, 1}) == is2);
    REQUIRE(dep_is(IndexVector{1, 0}) == temp_is2);
    REQUIRE(dep_is(IndexVector{1, 1}) == is2);

}

TEST_CASE(
  "IndexSpace construction for sub AO space dependent over ATOM index space") {
    IndexSpace AO{range(0, 20)};
    IndexSpace ATOM{{0, 1, 2, 3, 4}};

    TiledIndexSpace T_ATOM{ATOM, 2};
    std::map<IndexVector, IndexSpace> ao_atom_relation{
      /*atom 0*/ {IndexVector{0}, IndexSpace{AO, IndexVector{3, 4, 7}}},
      /*atom 1*/ {IndexVector{1}, IndexSpace{AO, IndexVector{1, 5, 7}}},
      /*atom 2*/ {IndexVector{2}, IndexSpace{AO, IndexVector{1, 9, 11}}},
      /*atom 3*/ {IndexVector{3}, IndexSpace{AO, IndexVector{11, 14}}},
      /*atom 4*/ {IndexVector{4}, IndexSpace{AO, IndexVector{2, 5, 13, 17}}}};

    CHECK_NOTHROW(IndexSpace{/*dependent spaces*/ {T_ATOM},
                             /*reference space*/ AO,
                             /*relation*/ ao_atom_relation});
}

