#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

#include "tamm/index_space.h"
using namespace tamm;

void check_indices(IndexSpace is, IndexVector iv) {
    int i = 0;
    REQUIRE(is.size() == iv.size());
    for(const auto& index : is) { REQUIRE(index == iv[i++]); }
}

TEST_CASE("IndexSpace construction with ranges ") {
    IndexSpace is{range(10)};
    IndexVector iv{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // check if the indices are equal
    check_indices(is, iv);
    // check subspace named "all"
    check_indices(is("all"), iv);

    IndexSpace is1{range(0, 10)};
    IndexVector iv1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // check if the indices are equal
    check_indices(is1, iv1);
    // check subspace named "all"
    check_indices(is1("all"), iv1);

    IndexSpace is2{range(0, 10, 2)};
    IndexVector iv2{0, 2, 4, 6, 8};
    // check if the indices are equal
    check_indices(is2, iv2);
    // check subspace named "all"
    check_indices(is2("all"), iv2);
}

TEST_CASE("IndexSpace construction with set of indices") {
    IndexSpace is1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    IndexVector iv1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // check if the indices are equal
    check_indices(is1, iv1);
    // check subspace named "all"
    check_indices(is1("all"), iv1);
}

TEST_CASE("IndexSpace construction with named subspaces") {
    IndexSpace is2{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {{"occ", {range(0, 5)}},
                    {"virt", {range(5, 10)}},
                    {"alpha", {range(0, 3), range(5, 8)}},
                    {"beta", {range(3, 5), range(8, 10)}}}};

    // check indices for full space
    check_indices(is2, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    // check indices for subspace named "all"
    check_indices(is2("all"), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    // check indices for subspace named "occ"
    check_indices(is2("occ"), {0, 1, 2, 3, 4});
    // check indices for subspace named "virt"
    check_indices(is2("virt"), {5, 6, 7, 8, 9});
    // check indices for subspace named "alpha"
    check_indices(is2("alpha"), {0, 1, 2, 5, 6, 7});
    // check indices for subspace named "beta"
    check_indices(is2("beta"), {3, 4, 8, 9});
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

    IndexSpace sub_is{is, range(2, 8),
                      {{"occ", {range(0, 3)}},
                       {"virt", {range(3, 6)}}}};

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
  IndexSpace temp_is1{{10,12,14,16,18}};
  IndexSpace temp_is2{{1,3,5,7,9}};

  IndexSpace agg_is{{temp_is1, temp_is2}};
  // check indices for created (aggregated)IndexSpace
  check_indices(agg_is, {10,12,14,16,18,1,3,5,7,9});
  // check indices for subspace name "all"
  check_indices(agg_is("all"), {10,12,14,16,18,1,3,5,7,9});
}

TEST_CASE("IndexSpace construction by aggregating with subnames") {
  IndexSpace temp_is{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                     {{"occ", {range(0, 5)}},
                      {"virt", {range(5, 10)}},
                      {"alpha", {range(0, 3), range(5, 8)}},
                      {"beta", {range(3, 5), range(8, 10)}}}};

  IndexSpace temp_is1{range(10, 20),
                      {{"occ", {range(0, 5)}},
                       {"virt", {range(5, 10)}},
                       {"alpha", {range(0, 3), range(5, 8)}},
                       {"beta", {range(3, 5), range(8, 10)}}}};

  IndexSpace agg_is{{temp_is, temp_is1},
                    {"occ", "virt"},
                    {{"local", {range(8, 13)}}},
                    {{"alpha", {"occ:alpha", "virt:alpha"}},
                     {"beta", {"occ:beta", "virt:beta"}}}};

  IndexVector full_indices = construct_index_vector(range(0,20));
  // check indices for created (sub)IndexSpace
  check_indices(agg_is, full_indices);
  // check indices for subspace name "all"
  check_indices(agg_is("all"), full_indices);
  // check indices for subspace name "occ"
  check_indices(agg_is("occ"), construct_index_vector(range(0,10)));
  // check indices for subspace name "virt"
  check_indices(agg_is("virt"), construct_index_vector(range(10,20)));
  // check indices for subspace name "local"
  check_indices(agg_is("local"), {8, 9, 10, 11, 12});
  // check indices for subspace name "alpha"
  check_indices(agg_is("alpha"), {0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17});
  // check indices for subspace name "beta"
  check_indices(agg_is("beta"), {3, 4, 8, 9, 13, 14, 18, 19});

}


TEST_CASE("IndexSpace construction using dependent IndexSpaces") {
    IndexSpace is1{range(0, 10)}; 
    IndexSpace is2{range(0, 10, 2)};
    IndexSpace is3{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {{"occ", {range(0, 5)}},
                    {"virt", {range(5, 10)}},
                    {"alpha", {range(0, 3), range(5, 8)}},
                    {"beta", {range(3, 5), range(8, 10)}}}};

    IndexSpace temp_is1{0,1,2};
    IndexSpace temp_is2{8,9};

    std::map<IndexVector, IndexSpace> dep_relation{ {{0,8}, is1},
                                                    {{0,9}, is2},
                                                    {{1,8}, temp_is2},
                                                    {{1,9}, is2},
                                                    {{2,8}, is1},
                                                    {{2,9}, is3}};

    // Dependent IndexSpace construction using dependency relation
    IndexSpace dep_is{{temp_is1, temp_is2}, dep_relation};
    
    // Check the dependency relation
    REQUIRE(dep_is(IndexVector{0,8}) == is1);
    REQUIRE(dep_is(IndexVector{0,9}) == is2);
    REQUIRE(dep_is(IndexVector{1,8}) == temp_is2);
    REQUIRE(dep_is(IndexVector{1,9}) == is2);
    REQUIRE(dep_is(IndexVector{2,8}) == is1);
    REQUIRE(dep_is(IndexVector{2,9}) == is3);
    
}

TEST_CASE("TiledIndexSpace construction") {

    IndexSpace is{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {{"occ", {range(0, 5)}},
                    {"virt", {range(5, 10)}},
                    {"alpha", {range(0, 3), range(5, 8)}},
                    {"beta", {range(3, 5), range(8, 10)}}}};
    // TiledIndexSpace construction with tile size of 5
    TiledIndexSpace t5_is{is, 5};
    
    // TiledIndexSpace construction with tile size of 5
    TiledIndexSpace t3_is{is, 3};
    

    // Check if the reference index space is equal to argument
    REQUIRE(t5_is.index_space() == is);
    REQUIRE(t3_is.index_space() == is);

    TiledIndexLabel i,j,k;

    // Construct TiledIndexLabels from TiledIndexSpace
    std::tie(i,j) = t5_is.labels<2>("all");
    // Check reference TiledIndexSpace on the labels
    REQUIRE(i.tiled_index_space().index_space() == t5_is.index_space());
     REQUIRE(j.tiled_index_space().index_space() == t5_is.index_space());
    // Construct TiledIndexLabels from TiledIndexSpace
    k = t3_is.label("all", 1);
    // Check reference TiledIndexSpace on the label
    REQUIRE(k.tiled_index_space().index_space() == t3_is.index_space());

}