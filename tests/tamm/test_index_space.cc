// #include <memory>
// #include <vector>
// #include <iostream>

#include "tamm/index_space.h"

using namespace tamm;

/**
 * @brief Helper method for printing indices
 *
 * @param [in] is input IndexSpace
 */
void printIndices(const IndexSpace& is) {
    std::cout << "Indices: ";
    for(const auto& index : is) { std::cout << index << " "; }
    std::cout << std::endl;
}

int main() {
    // Construct an index space spanning from 0 to N-1\t=> [0,N)
    IndexSpace is1{range(10)}; // indicies = {0,1,2,3,4,5,6,7,8,9}
    std::cout << "Full is1\t=>\t";
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

    std::cout << "Full is2\t=>\t";
    printIndices(is2);

    std::cout << "is2(\"all\")\t=>\t";
    printIndices(is2("all"));

    std::cout << "is2(\"occ\")\t=>\t";
    printIndices(is2("occ"));

    std::cout << "is2(\"virt\")\t=>\t";
    printIndices(is2("virt"));

    std::cout << "is2(\"alpha\")\t=>\t";
    printIndices(is2("alpha"));

    std::cout << "is2(\"beta\")\t=>\t";
    printIndices(is2("beta"));

    IndexSpace is3{0, 1, 2, 3, 4}; // indicies = {0,1,2,3,4}
    std::cout << "Full is3\t=>\t";
    printIndices(is3);

    // By giving a range - IndexSpace(Range r)
    IndexSpace is4{range(0, 10, 2)}; // indicies = {0,2,4,6,8}
    std::cout << "Full is4\t=>\t";
    printIndices(is4);

    IndexSpace is5{range(5, 10)}; // indicies = {5,6,7,8,9}
    std::cout << "Full is5\t=>\t";
    printIndices(is5);

    // Constructing index spaces from other index spaces
    // IndexSpace(IndexSpace& is1, IndexSpace& is2)
    IndexSpace is7{{is3, is5}}; // indicies = {0,1,2,3,4,5,6,7,8,9}
    std::cout << "Aggregated is7\t=>\t";
    printIndices(is7);
    // printIndices(is7("occ"));

    IndexSpace is8{{is5, is2("occ")},
                   {"occ", "virt"},
                   {{"alpha", {range(0, 3), range(5, 8)}},
                    {"beta", {range(3, 5), range(8, 10)}}}};
    // indicies = {5,6,7,8,9,0,1,2,3,4}

    std::cout << "Aggregated is8\t=>\t";
    printIndices(is8);

    std::cout << "is8(\"occ\")\t=>\t";
    printIndices(is8("occ"));

    std::cout << "is8(\"virt\")\t=>\t";
    printIndices(is8("virt"));

    std::cout << "is8(\"alpha\")\t=>\t";
    printIndices(is8("alpha"));

    std::cout << "is8(\"beta\")\t=>\t";
    printIndices(is8("beta"));

    IndexSpace temp_is{range(10, 20),
                       {{"occ", {range(0, 5)}},
                        {"virt", {range(5, 10)}},
                        {"alpha", {range(0, 3), range(5, 8)}},
                        {"beta", {range(3, 5), range(8, 10)}}},
                       {{Spin{1}, {range(2, 5), range(7, 10)}},
                        {Spin{2}, {range(0, 2), range(5, 7)}}}};

    IndexSpace is9{{temp_is, is2},
                   {"occ", "virt"},
                   {{"local", {range(2, 5)}}},
                   {{"alpha", {"occ:alpha", "virt:alpha"}},
                    {"beta", {"occ:beta", "virt:beta"}}}};

    std::cout << "Aggregated is9\t=>\t";
    printIndices(is9);

    std::cout << "is9(\"occ\")\t=>\t";
    printIndices(is9("occ"));

    std::cout << "is9(\"virt\")\t=>\t";
    printIndices(is9("virt"));

    std::cout << "is9(\"local\")\t=>\t";
    printIndices(is9("local"));

    std::cout << "is9(\"alpha\")\t=>\t";
    printIndices(is9("alpha"));

    std::cout << "is9(\"beta\")\t=>\t";
    printIndices(is9("beta"));

    // Disjoint aggregation
    IndexSpace is10{{is3, is5}}; // indicies = {0,1,2,3,4,5,6,7,8,9}
    std::cout << "Aggregated is10\t=>\t";
    printIndices(is10);

    // By specifying sub-space with range
    IndexSpace is11{is8, range(2, 6)}; // indicies = {7,8,9,0}
    std::cout << "SubSpace is11\t=>\t";
    printIndices(is11);

    // By specifying range
    IndexSpace is12{is1,
                    range(0, 9, 2),
                    {{"occ", {range(0, 2)}},
                     {"virt", {range(2, 5)}}}}; // indicies = {0,2,4,6,8}
    std::cout << "SubSpace is12\t=>\t";
    printIndices(is12);
    std::cout << "is12(\"occ\")\t=>\t";
    printIndices(is12("occ"));
    std::cout << "is12(\"virt\")\t=>\t";
    printIndices(is12("virt"));

    
    IndexSpace temp_is1{0,1,2};
    IndexSpace temp_is2{8,9};

    std::map<IndexVector, IndexSpace> dep_relation{ {{0,8}, is1},
                                                    {{0,9}, is2},
                                                    {{1,8}, temp_is2},
                                                    {{1,9}, is2},
                                                    {{2,8}, is1},
                                                    {{2,9}, is9}};

    // Dependent IndexSpace construction using relation
    IndexSpace{{temp_is1, temp_is2},
                dep_relation};


    // TiledIndexSpace construction with tile size
    TiledIndexSpace tis{is9, 10};

    // Create a new TiledIndexSpace from a named subspace ("occ")
    TiledIndexSpace tis2{tis, "occ"};

    TiledIndexLabel i, j, k;
    std::tie(i, j, k) = tis.labels<3>("occ");
    std::cout << "TiledIndexLabel i " << i.get_label() << std::endl;
    std::cout << "TiledIndexLabel j " << j.get_label() << std::endl;
    std::cout << "TiledIndexLabel k " << k.get_label() << std::endl;

    // Get the index value from an index space
    // By using point method - Point IndexSpace::point(Index i)
    const Index& i4 = is4.index(Index{4}); // index 	i\t=> 8
    std::cout << "Index is4[4]\t" << i4 << std::endl;

    // By using operator[] - Point& IndexSpace::operator[](Index i)
    const Index& j4 = is5[Index{4}]; // index j\t=> 9
    std::cout << "Index is5[4]\t" << j4 << std::endl;

    return 1;
}
