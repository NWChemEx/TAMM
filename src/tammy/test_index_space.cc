// #include <memory>
// #include <vector>
// #include <iostream>

#include "index_space_sketch.h"

using namespace tammy;

void printIndices(const IndexSpace& is) {
    std::cout << "Indices: ";
    for(const auto& point : is) { std::cout << point << " "; }
    std::cout << std::endl;
}

int main() {
    // Construct an index space spanning from 0 to N-1 => [0,N)
    IndexSpace is1{range(10)}; // indicies = {0,1,2,3,4,5,6,7,8,9}
    std::cout << "Full is1 \t";
    printIndices(is1);

    // By specifying the indicies it represents -
    // IndexSpace with subspace and attributes
    // indicies = {0,1,2,3,4,5,6,7,8,9}
    IndexSpace is2{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {{"occ", {range(0, 5)}},
                    {"virt", {range(5, 10)}},
                    {"alpha", {range(0, 3), range(5, 8)}},
                    {"beta", {range(3, 5), range(8, 10)}}},
                   {{Spin{1}, {range(2, 5), range(7, 10)}},
                    {Spin{2}, {range(0, 2), range(5, 7)}}}};

    std::cout << "Full is2 \t";
    printIndices(is2);

    std::cout << "all_is2 \t";
    printIndices(is2("all"));

    std::cout << "occ_is2 \t";
    printIndices(is2("occ"));

    std::cout << "virt_is2 \t";
    printIndices(is2("virt"));

    std::cout << "alpha_is2 \t";
    printIndices(is2("alpha"));

    std::cout << "beta_is2 \t";
    printIndices(is2("beta"));

    IndexSpace is3{0, 1, 2, 3, 4}; // indicies = {0,1,2,3,4}
    std::cout << "Full is3 \t";
    printIndices(is3);

    // By giving a range - IndexSpace(Range r)
    IndexSpace is4{range(0, 10)}; // indicies = {0,1,2,3,4,5,6,7,8,9}
    std::cout << "Full is4 \t";
    printIndices(is4);

    IndexSpace is5{range(5, 10)}; // indicies = {5,6,7,8,9}
    std::cout << "Full is5 \t";
    printIndices(is5);

    // Constructing index spaces from other index spaces
    // IndexSpace(IndexSpace& is1, IndexSpace& is2)
    IndexSpace is7{
      std::vector<IndexSpace>{is3, is5}}; // indicies = {0,1,2,3,4,5,6,7,8,9}
    std::cout << "Aggregated is7 \t";
    printIndices(is7);
    // printIndices(is7("occ"));

    IndexSpace is8{
      {is5, is2("occ")},
      {"occ", "virt"},
      {{"alpha", {range(0, 3), range(5, 8)}},
       {"beta",
        {range(3, 5), range(8, 10)}}}}; // indicies = {5,6,7,8,9,0,1,2,3,4}

    std::cout << "Aggregated is8 \t";
    printIndices(is8);

    std::cout << "occ_is8 \t";
    printIndices(is8("occ"));

    std::cout << "virt_is8 \t";
    printIndices(is8("virt"));

    std::cout << "alpha_is8 \t";
    printIndices(is8("alpha"));

    std::cout << "beta_is8 \t";
    printIndices(is8("beta"));

    IndexSpace temp_is{range(10, 20),
                       {{"occ", {range(0, 5)}},
                        {"virt", {range(5, 10)}},
                        {"alpha", {range(0, 3), range(5, 8)}},
                        {"beta", {range(3, 5), range(8, 10)}}},
                       {{Spin{1}, {range(2, 5), range(7, 10)}},
                        {Spin{2}, {range(0, 2), range(5, 7)}}}};

    IndexSpace is18{{temp_is, is2},
                    {"occ", "virt"},
                    {{"local", {range(2, 5)}}},
                    {{"alpha", {"occ:alpha", "virt:alpha"}},
                     {"beta", {"occ:beta", "virt:beta"}}}};

    std::cout << "Aggregated is18 \t";
    printIndices(is18);
    std::cout << "local_is18 \t";
    printIndices(is18("local"));
    std::cout << "alpha_is18 \t";
    printIndices(is18("alpha"));
    std::cout << "beta_is18 \t";
    printIndices(is18("beta"));

    // Disjoint aggregation
    IndexSpace is9{{is3, is5}}; // indicies = {0,1,2,3,4,5,6,7,8,9}
    std::cout << "Aggregated is9 \t";
    printIndices(is9);

    // Non-disjoint aggregation - This will give an error as it is not allowed
    // IndexSpace is10{{is3, is3}}; // indicies = {0,1,2,3,4,0,1,2,3,4}
    // std::cout << "Aggregated is10 \t";
    // printIndices(is10);

    // Sub-space by permuting the indicies of another index space
    // IndexSpace(IndexSpace& ref, range r)

    // By specifying sub-space with range
    IndexSpace is11{is8, range(2, 6)}; // indicies = {0,1,2,3}
    std::cout << "SubSpace is11 \t";
    printIndices(is11);

    // By specifying range
    IndexSpace is12{is1,
                    range(0, 9, 2),
                    {{"occ", {range(0, 2)}},
                     {"virt", {range(2, 5)}}}}; // indicies = {4,5,6,7,8,9}
    std::cout << "SubSpace is12 \t";
    printIndices(is12);
    std::cout << "occ_is12 \t";
    printIndices(is12("occ"));
    std::cout << "virt_is12 \t";
    printIndices(is12("virt"));

    TiledIndexSpace tis{is18, 10};

    TiledIndexSpace tis2{tis, "occ"};

    TiledIndexLabel i, j, k;
    std::tie(i, j, k) = tis.range_labels<3>("occ", 4);
    std::cout << "label i " << i.get_label() << std::endl;
    std::cout << "label j " << j.get_label() << std::endl;
    std::cout << "label k " << k.get_label() << std::endl;

    // Get the index value from an index space
    // By using point method - Point IndexSpace::point(Index i)
    const Index& i4 = is4.point(Index{4}); // index 	i => 4
    std::cout << "Index is4[4]\t" << i4 << std::endl;

    // By using operator[] - Point& IndexSpace::operator[](Index i)
    const Index& j4 = is5[Index{4}]; // index j => 9
    std::cout << "Index is5[4]\t" << j4 << std::endl;

    return 1;
}
