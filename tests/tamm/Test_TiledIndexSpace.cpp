#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include <iostream>
#include <tamm/tamm.hpp>

using namespace tamm;


void print_dependency(const TiledIndexSpace& tis) {
    auto dep_map = tis.tiled_dep_map();
    std::cerr << "Dependency Map" << std::endl;
   
    for(const auto& [key, subtis] : dep_map) {
        std::cerr << "( ";
        for(const auto& idx : key) {
            std::cerr << idx << " ";
        }
        std::cerr << ") -> ";

        std::cerr << "{ ";
        for(const auto& idx : subtis.ref_indices()) {
            std::cerr << idx << " ";
        }
        std::cerr << "}" << std::endl;
    }
}


TEST_CASE("TiledIndexSpace construction") {
    IndexSpace is{{10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
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

    TiledIndexLabel i, j, k;

    // Construct TiledIndexLabels from TiledIndexSpace
    std::tie(i, j) = t5_is.labels<2>("all");
    // Check reference TiledIndexSpace on the labels
    REQUIRE(i.tiled_index_space() == t5_is);
    REQUIRE(j.tiled_index_space() == t5_is);
    // Construct TiledIndexLabels from TiledIndexSpace
    k = t3_is.label("all", 1);
    // Check reference TiledIndexSpace on the label
    REQUIRE(k.tiled_index_space() == t3_is);
}

TEST_CASE("TiledIndexSpace construction with multiple tile size") {
    IndexSpace is1{range(10, 20)};

    TiledIndexSpace tis1{is1, {2, 3, 5}};

    std::vector<IndexVector> tiled_iv{
      {10, 11}, {12, 13, 14}, {15, 16, 17, 18, 19}};

    for(size_t i = 0; i < tis1.num_tiles(); i++) {
        auto it     = tis1.block_begin(i);
        auto it_ref = tiled_iv[i].begin();

        while(it != tis1.block_end(i)) {
            REQUIRE((*it) == (*it_ref));
            it++;
            it_ref++;
        }
    }
}

TEST_CASE("TiledIndexSpace construction with multiple tile size,named "
          "subspaces") {
    IndexSpace is1{range(10, 20),
                   {{"occ", {range(0, 5)}}, {"virt", {range(5, 10)}}}};

    TiledIndexSpace tis1{is1, {2, 3, 5}};

    std::vector<IndexVector> tiled_iv{
      {10, 11}, {12, 13, 14}, {15, 16, 17, 18, 19}};

    for(size_t i = 0; i < tis1.num_tiles(); i++) {
        auto it     = tis1.block_begin(i);
        auto it_ref = tiled_iv[i].begin();

        while(it != tis1.block_end(i)) {
            REQUIRE((*it) == (*it_ref));
            it++;
            it_ref++;
        }
    }
}

TEST_CASE("TiledIndexSpace tiling check") {
    // Create a (range-based) IndexSpace with name subspaces
    IndexSpace tempIS1{range(10, 50),
                       {{"occ", {range(0, 20)}},
                        {"virt", {range(20, 40)}},
                        {"alpha", {range(0, 13), range(20, 33)}},
                        {"beta", {range(13, 20), range(33, 40)}}}};

    // Create an (range-based) IndexSpace with name subspaces
    IndexSpace tempIS2{range(50, 90),
                       {{"occ", {range(0, 20)}},
                        {"virt", {range(20, 40)}},
                        {"alpha", {range(0, 8), range(20, 28)}},
                        {"beta", {range(8, 20), range(28, 40)}}}};

    // Create an (aggregation-based) IndexSpace with named subspaces and
    // reference named subspaces
    IndexSpace tempIS3{{tempIS1, tempIS2},
                       {"occ", "virt"},
                       {/*No extra named subspace*/},
                       {{"alpha", {"occ:alpha", "virt::alpha"}},
                        {"beta", {"occ:beta", "virt:beta"}}}};

    // TiledIndexSpace construction with tile size of 10
    TiledIndexSpace t10_is{tempIS3, 10};

    // Reference tiled index vectors with respect to each name subspace
    std::vector<IndexVector> tiled_iv{
      {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, // occ + alpha
      {20, 21, 22},                             // occ + alpha
      {23, 24, 25, 26, 27, 28, 29},             // occ + beta
      {30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, // occ + alpha
      {40, 41, 42},                             // occ + alpha
      {43, 44, 45, 46, 47, 48, 49},             // occ + beta
      {50, 51, 52, 53, 54, 55, 56, 57},         // virt + alpha
      {58, 59, 60, 61, 62, 63, 64, 65, 66, 67}, // virt + beta
      {68, 69},                                 // virt + beta
      {70, 71, 72, 73, 74, 75, 76, 77},         // virt + alpha
      {78, 79, 80, 81, 82, 83, 84, 85, 86, 87}, // virt + beta
      {88, 89}};                                // virt + beta

    for(size_t i = 0; i < t10_is.num_tiles(); i++) {
        auto it     = t10_is.block_begin(i);
        auto it_ref = tiled_iv[i].begin();

        while(it != t10_is.block_end(i)) {
            REQUIRE((*it) == (*it_ref));
            it++;
            it_ref++;
        }
    }
}

TEST_CASE("TiledIndexSpace tiling with different name subspaces") {
    // Create a (range-based) IndexSpace with name subspaces
    IndexSpace tempIS1{range(10, 50),
                       {{"occ", {range(0, 20)}},
                        {"virt", {range(20, 40)}},
                        {"alpha", {range(0, 12), range(20, 33)}},
                        {"beta", {range(13, 20), range(33, 40)}}}};

    // Create an (range-based) IndexSpace with name subspaces
    IndexSpace tempIS2{range(50, 90),
                       {{"occ", {range(0, 20)}},
                        {"virt", {range(20, 40)}},
                        {"alpha", {range(0, 8), range(20, 28)}},
                        {"beta", {range(8, 20), range(28, 40)}}}};

    // Create an (aggregation-based) IndexSpace with named subspaces and
    // reference named subspaces
    IndexSpace tempIS3{{tempIS1, tempIS2},
                       {"occ", "virt"},
                       {/*No extra named subspace*/},
                       {{"alpha", {"occ:alpha", "virt::alpha"}},
                        {"beta", {"occ:beta", "virt:beta"}}}};

    // TiledIndexSpace construction with tile size of 10
    TiledIndexSpace t10_is{tempIS3, 10};

    // Reference tiled index vectors with respect to each name subspace
    std::vector<IndexVector> tiled_iv{
      {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, // occ + alpha
      {20, 21},                                 // occ + alpha
      {22},                                     // occ
      {23, 24, 25, 26, 27, 28, 29},             // occ + beta
      {30, 31, 32, 33, 34, 35, 36, 37, 38, 39}, // occ + alpha
      {40, 41, 42},                             // occ + alpha
      {43, 44, 45, 46, 47, 48, 49},             // occ + beta
      {50, 51, 52, 53, 54, 55, 56, 57},         // virt + alpha
      {58, 59, 60, 61, 62, 63, 64, 65, 66, 67}, // virt + beta
      {68, 69},                                 // virt + beta
      {70, 71, 72, 73, 74, 75, 76, 77},         // virt + alpha
      {78, 79, 80, 81, 82, 83, 84, 85, 86, 87}, // virt + beta
      {88, 89}};                                // virt + beta

    for(size_t i = 0; i < t10_is.num_tiles(); i++) {
        auto it     = t10_is.block_begin(i);
        auto it_ref = tiled_iv[i].begin();

        while(it != t10_is.block_end(i)) {
            REQUIRE((*it) == (*it_ref));
            it++;
            it_ref++;
        }
    }
}


TEST_CASE("TiledIndexSpace construction checks") {
    bool failed = false;
    IndexSpace simple_is{range(10, 20)};
    IndexSpace named_is{range(10, 20),
                        {{"occ", {range(0, 5)}}, {"virt", {range(5, 10)}}}};

    /* Default tiling constructor*/
    // Test default tiling
    try {
        TiledIndexSpace tis{simple_is};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // Test default tiling on named index space
    try {
        TiledIndexSpace tis{named_is};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // Test different tile sizes
    try {
        TiledIndexSpace tis{simple_is, 5};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // Test different tile sizes on named
    try {
        TiledIndexSpace tis{named_is, 5};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // Test tile size larger then index space size
    try {
        TiledIndexSpace tis{simple_is, 20};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // Test tile size larger then index space size on named
    try {
        TiledIndexSpace tis{named_is, 20};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    /* Sub-TiledIndexSpace construction */
    
    // with default tiling (= 1)
    TiledIndexSpace tis_full{simple_is};
    TiledIndexSpace tis_named{named_is};

    // Test with individual indicies
    try {
        TiledIndexSpace tis{tis_full, IndexVector{0}};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // Test with individual indicies
    try {
        TiledIndexSpace tis{tis_full, IndexVector{0,1}};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // Test with individual indicies larger then size
    try {
        TiledIndexSpace tis{tis_full, IndexVector{100}};
    } catch(...) { failed = true; }
    REQUIRE(failed);
    failed = false;

    // Test with ranges
    try {
        TiledIndexSpace tis{tis_full, range(1)};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // Test with ranges
    try {
        TiledIndexSpace tis{tis_full, range(0,2)};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // Test with ranges larger then size
    try {
        TiledIndexSpace tis{tis_full, range(100)};
    } catch(...) { failed = true; }
    REQUIRE(failed);
    failed = false;

    // Test for compatibilities of named subspaces
    {
        const auto& N = tis_named("all");
        const auto& O = tis_named("occ");
        const auto& V = tis_named("virt");

        // REQUIRE(!O.is_compatible_with(V));
        // REQUIRE(!V.is_compatible_with(O));
        // REQUIRE(!N.is_compatible_with(O));
        // REQUIRE(!N.is_compatible_with(V));
        REQUIRE(O.is_compatible_with(N));
        REQUIRE(V.is_compatible_with(N));
    }
    
}

TEST_CASE("TiledIndexSpace construction with Spin attributes") {
    IndexSpace is_att{range(0, 20),
                      {
                        {"occ", {range(0,10)}},
                        {"virt", {range(10,20)}}
                      },
                      {
                        {Spin{2}, {range(0,5), range(10,15)}},
                        {Spin{1}, {range(5,10), range(15,20)}}
                      }
                    };

    REQUIRE(is_att.has_spin() == true);

    std::vector<Tile> tilesizes{3,2,3,2,3,2,3,2};

    TiledIndexSpace tis_att{is_att, 3};
    TiledIndexSpace tis_att2{is_att, tilesizes};

    REQUIRE(tis_att.num_tiles() == 8);
    REQUIRE(tis_att2.num_tiles() == 8);

    for (size_t i = 0; i < tis_att.num_tiles(); i++) {
        
        REQUIRE(tis_att.tile_size(i) == tilesizes[i]);
        REQUIRE(tis_att2.tile_size(i) == tilesizes[i]);

    }
    
}

TEST_CASE("TiledIndexSpace operations") {

    TiledIndexSpace AOs{IndexSpace{range(7)}};
    TiledIndexSpace MOs{IndexSpace{range(4)}};

    std::map<IndexVector, TiledIndexSpace> new_dep{
        {
            {{0, 0}, TiledIndexSpace{AOs, IndexVector{0,3,4}}},           
            {{1, 5}, TiledIndexSpace{AOs, IndexVector{0,3,6}}},
            {{2, 0}, TiledIndexSpace{AOs, IndexVector{1,3,5}}},
            {{2, 5}, TiledIndexSpace{AOs, IndexVector{0,5}}}
        }
    };

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_q{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{0,3,4}}},           
            {{1}, TiledIndexSpace{AOs, IndexVector{0,3,6}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{1,3,5}}}
        }
    };

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_d{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{1,3,5}}},
            {{1}, TiledIndexSpace{AOs, IndexVector{0,1,2}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2,4}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1,6}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            // {{5}, TiledIndexSpace{AOs, IndexVector{0,1,2}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{0,1,2}}}
        }
    };

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_c{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{3}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{2}}}
        }
    };

    TiledIndexSpace test_tis{AOs, {MOs, AOs}, new_dep};

    TiledIndexSpace tSubAO_AO_Q{AOs, {MOs}, dep_nu_mu_q};

    TiledIndexSpace tSubAO_AO_D{AOs, {AOs}, dep_nu_mu_d};

    TiledIndexSpace tSubAO_AO_C{AOs, {AOs}, dep_nu_mu_c};

    std::cerr << "tSubAO_AO_Q ";
    print_dependency(tSubAO_AO_Q);
    
    auto inv_tSubAO_AO_Q = invert_tis(tSubAO_AO_Q);
    std::cerr << "inv_tSubAO_AO_Q ";
    print_dependency(inv_tSubAO_AO_Q);

    auto comp_tSubAO_AO_Q_D = compose_tis(tSubAO_AO_Q, tSubAO_AO_D);
    std::cerr << "tSubAO_AO_D ";
    print_dependency(tSubAO_AO_D);
    std::cerr << "comp_tSubAO_AO_Q_D ";
    print_dependency(comp_tSubAO_AO_Q_D);

    auto union_tSubAO_AO_D_C = union_tis(tSubAO_AO_D, tSubAO_AO_C);
    std::cerr << "tSubAO_AO_D ";
    print_dependency(tSubAO_AO_D);
    std::cerr << "tSubAO_AO_C ";
    print_dependency(tSubAO_AO_C);
    std::cerr << "union_tSubAO_AO_D_C ";
    print_dependency(union_tSubAO_AO_D_C);

    auto project_test_tis = project_tis(test_tis, MOs);
    std::cerr << "test_tis ";
    print_dependency(test_tis);
    std::cerr << "project_test_tis ";
    print_dependency(project_test_tis);

    auto project_MO_Q = project_tis(tSubAO_AO_Q, MOs);
    std::cerr << "tSubAO_AO_Q ";
    print_dependency(tSubAO_AO_Q);
    std::cerr << "project_MO_Q " << std::endl;
    std::cerr << "{ ";
    for(const auto& idx : project_MO_Q.ref_indices()) {
        std::cerr << idx << " ";
    }
    std::cerr << "}" << std::endl;

    TiledIndexSpace tis_1{AOs, IndexVector{1,2,5}};
    TiledIndexSpace tis_2{AOs, IndexVector{2,3,6}};

    auto u_tis12 = union_tis(tis_1, tis_2);
    std::cerr << "u_tis12 " << std::endl;
    std::cerr << "{ ";
    for(const auto& idx : u_tis12.ref_indices()) {
        std::cerr << idx << " ";
    }
    std::cerr << "}" << std::endl;

}
