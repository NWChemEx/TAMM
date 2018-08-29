#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include "tamm/tamm.hpp"

using namespace tamm;

template<typename T>
void tensor_contruction(const TiledIndexSpace& T_AO,
                        const TiledIndexSpace& T_MO,
                        const TiledIndexSpace& T_ATOM,
                        const TiledIndexSpace& T_AO_ATOM) {
    TiledIndexLabel A, r, s, mu, mu_A;

    A              = T_ATOM.label("all",1);
    std::tie(r, s) = T_MO.labels<2>("all");
    mu             = T_AO.label("all",1);
    mu_A           = T_AO_ATOM.label("all",1);

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

TEST_CASE("Tensor Declaration Syntax") {
    using Tensor = Tensor<double>;

    {
        //Scalar value
        Tensor T{};
    }

    {
        //Vector of length 10
        IndexSpace is{range(10)};
        TiledIndexSpace tis{is};
        
        Tensor T{tis};
    }

    {
        //Matrix of size 10X20
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        TiledIndexSpace tis1{is1};
        TiledIndexSpace tis2{is2};

        Tensor T{tis1, tis2};

    }

    {
        //Matrix of size 10X20X30
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{range(30)};

        TiledIndexSpace tis1{is1}, tis2{is2}, tis3{is3};

        Tensor T{tis1, tis2, tis3};
    }

    {
        //Vector from two different subspaces
        IndexSpace is{range(10)};
        IndexSpace is1{is, range(0, 4)};
        IndexSpace is2{is, range(4, is.size())};

        IndexSpace is3{{is1, is2}};

        TiledIndexSpace tis{is3};

        Tensor T{tis};

    }

    {
        //Matrix with split rows -- subspaces of lengths 4 and 6
        IndexSpace is1{range(10)};
        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{{is2, is3}};
        IndexSpace is5{range(20)};

        TiledIndexSpace tis4{is4}, tis5{is5};

        Tensor T{tis4, tis5};

    }

    {
        //Matrix with split columns -- subspaces of lengths 12 and 8
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{is2, range(0, 12)};
        IndexSpace is4{is2, range(12, is2.size())};

        IndexSpace is5{{is3, is4}};

        TiledIndexSpace tis1{is1}, tis5{is5};

        Tensor T{tis1, tis5};


    }

    {
        //Matrix with split rows and columns 
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
        //Tensor with first dimension split -- subspaces of lengths 4 and 6
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
        //Tensor with second dimension split -- subspaces of lengths 12 and 8
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{is2, range(0, 12)};
        IndexSpace is4{is2, range(12, is2.size())};

        IndexSpace is5{{is3, is4}};
        IndexSpace is6{range(30)};

        TiledIndexSpace tis1{is1}, tis5{is5}, tis6{is6} ;

        Tensor T{tis1, tis5, tis6};


    }

    {
        //Tensor with third dimension split -- subspaces of lengths 13 and 17
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{range(30)};

        IndexSpace is4{is3, range(0,13)};
        IndexSpace is5{is3, range(13, is3.size())};

        IndexSpace is6{{is4, is5}};

        TiledIndexSpace tis1{is1}, tis2{is2}, tis6{is6} ;

        Tensor T{tis1, tis2, tis6};


    }

    {
        //Tensor with first and second dimensions split
        IndexSpace is1{range(10)};
        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{{is2, is3}};

        IndexSpace is5{range(20)};
        IndexSpace is6{is5, range(0, 12)};
        IndexSpace is7{is5, range(12, is5.size())};

        IndexSpace is8{{is6, is7}};

        IndexSpace is9{range(30)};

        TiledIndexSpace tis4{is4}, tis8{is8}, tis9{is9} ;

        Tensor T{tis4, tis8, tis9};
    }

    {
        //Tensor with first and third dimensions split
        IndexSpace is1{range(10)};
        IndexSpace is2{is1, range(0, 4)};
        IndexSpace is3{is1, range(4, is1.size())};

        IndexSpace is4{{is2, is3}};

        IndexSpace is5{range(20)};
        IndexSpace is6{range(30)};
        IndexSpace is7{is6, range(0,13)};
        IndexSpace is8{is5, range(13, is6.size())};

        IndexSpace is9{{is7, is8}};


        TiledIndexSpace tis4{is4}, tis5{is5}, tis9{is9} ;

        Tensor T{tis4, tis5, tis9};
    }

    {
        //Tensor with second and third dimensions split
        IndexSpace is1{range(10)};
        IndexSpace is2{range(20)};
        IndexSpace is3{is2, range(0, 12)};
        IndexSpace is4{is2, range(12, is2.size())};

        IndexSpace is5{{is3, is4}};

        IndexSpace is6{range(30)};
        IndexSpace is7{is6, range(0,13)};
        IndexSpace is8{is5, range(13, is6.size())};

        IndexSpace is9{{is7, is8}};


        TiledIndexSpace tis1{is1}, tis5{is5}, tis9{is9} ;

        Tensor T{tis1, tis5, tis9};
    }

    {
        //Tensor with first, second and third dimensions split
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

        TiledIndexSpace tis4{is4}, tis8{is8}, tis12{is12} ;

        Tensor T{tis4, tis8, tis12};
    }

    {
        //Vector with more than one split of subspaces
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
        //Matrix with more than one split of first dimension
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
        //Vector with odd number elements from one space and even number elements from another
        IndexSpace is1{range(0, 10, 2)};
        IndexSpace is2{range(1, 10, 2)};
        IndexSpace is{{is1, is2}};
        TiledIndexSpace tis{is};
        Tensor T{tis};
    }

    {
        //Matrix with odd rows from one space and even from another
        IndexSpace is1{range(0, 10, 2)};
        IndexSpace is2{range(1, 10, 2)};
        IndexSpace is3{{is1, is2}};

        IndexSpace is4{range(20)};

        TiledIndexSpace tis3{is3}, tis4{is4};
        Tensor T{tis3, tis4};
   
    }

}
