#include <tamm/eigen_solvers.hpp>
#include "test_tamm.hpp"


using namespace tamm;


//Type used to store correct answer's values
using buffer_type = std::vector<double>;
//Type used to associate labels with correct answers
using map_type = std::map<std::string, buffer_type>;

static const std::vector<map_type> input_values {
    { //start map1
        {//start pair1
            "A",
            {3.0, 1.0, -1.0, 1.0, 3.0, -1.0, -1.0, -1.0, 5.0}
        },//end pair1
        {//start pair2
            "B",
            {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}
        }//end pair2
    },//end map1
    {// start map2
        {//start pair 1
            "A",
            {-32.5774,   -7.57883, -0.0144739,          0,          0,
              -1.2401,    -1.2401,
             -7.57883,   -9.20094,  -0.176891,          0,          0,
             -2.90671,   -2.90671,
           -0.0144739,  -0.176891,   -7.41531,          0,          0,
             -1.35687,   -1.35687,
                    0,          0,          0,   -7.34714,          0,
                    0,          0,
                    0,          0,          0,          0,   -7.45882,
             -1.67515,    1.67515,
              -1.2401,   -2.90671,   -1.35687,          0,   -1.67515,
             -4.54017,   -1.07115,
              -1.2401,   -2.90671,   -1.35687,          0,    1.67515,
             -1.07115,   -4.54017}
        },//end pair 1
        {//start pair 2
            "B",
            {           1,     0.236704,  1.04633e-17,            0,
                        0,    0.0384056,    0.0384056,
                 0.236704,            1, -2.46745e-17,            0,
                        0,     0.386139,     0.386139,
              1.04633e-17, -2.46745e-17,            1,            0,
                        0,     0.209728,     0.209728,
                        0,            0,            0,            1,
                        0,            0,            0,
                        0,            0,            0,            0,
                        1,     0.268438,    -0.268438,
                0.0384056,     0.386139,     0.209728,            0,
                 0.268438,            1,     0.181761,
                0.0384056,     0.386139,     0.209728,            0,
                -0.268438,     0.181761,            1}
        }//end pair2
    } //end map 2
};

static const std::vector<map_type> corr_values {
    {//start map 1
        {//start pair 1
            "values",
            {2.0, 3.0, 6.0}
        },//end pair 1
        {//start pair1
            "vectors",
            {-1.0/std::sqrt(2), 1.0/std::sqrt(3), 1.0/std::sqrt(6),
              1.0/std::sqrt(2), 1.0/std::sqrt(3), 1.0/std::sqrt(6),
                           0.0, 1.0/std::sqrt(3), -2.0/std::sqrt(6)}
        }//end pair 1
    },//end map 1
    {//start map 2
        {//start pair 1
            "values",
            {-32.5783075736833325,  -8.0815361611936343,
              -7.5500858983236077,  -7.3639678130646615,
              -7.3471399999999996,  -4.0022984557439845,
              -3.9811092863489610}
        },//end pair 1
        {//start pair 2
            "vectors",
            {1.0015436395837849,-0.2336240822815417,0.0000000000000000,
             0.0856847817702707,0.0000000000000000,0.0482227308257768,
             0.0000000000000000,-0.0071895880557639,1.0579381216544306,
            -0.0000000000000000,-0.3601127449795136,-0.0000000000000000,
            -0.4631214286932498,0.0000000000000000,-0.0002671349347418,
             0.4272878893369705,0.0000000000000000,0.9399416982587350,
            -0.0000000000000001,-0.2129389181675623,0.0000000000000002,
             0.0000000000000000,0.0000000000000000,0.0000000000000042,
             0.0000000000000000,1.0000000000000002,-0.0000000000000000,
            -0.0000000000000001,0.0000000000000000,-0.0000000000000000,
            -1.0610700024287985,-0.0000000000000001,0.0000000000000044,
             0.0000000000000001,0.2965077588850878,0.0018213503629215,
            -0.1492542099495040,0.1377204596696941,-0.0378594314965900,
            -0.0000000000000007,0.7807000773739159,-0.8501409260775487,
             0.0018213503629215,-0.1492542099495042,-0.1377204596696940,
            -0.0378594314965904,0.0000000000000007,0.7807000773739161,
             0.8501409260775485,}
        }//end pair 2
    }//end map 2
};

/**
 * @brief Function designed for comparing a TAMM tensor to an std::vector
 *
 * The idea is that the correct answer is hard-coded into an std::vector (in
 * row major format).  Then the TAMM tensor is generated and this function is
 * called to actually perform the comparison.  At the moment this function has
 * some severe restrictions:
 *   - Equality thresholds are hard-coded (can be made into non-type parameters)
 *   - The tensor is assumed to be made of a single block.
 *
 * This function should really be factored out...
 *
 * @tparam tensor_type Assumed to be an instance of TAMM's Tensor class.
 * @tparam T the type of the scalar elements within the tensor
 * @param t The tamm tensor to check
 * @param corr the values (in row-major format) the tensor should contain.
 * @param absolute If true the comparison will only consider the absolute value
 *        while comparing (useful as some operations are only defined up to a
 *        sign).  By default @p absolute is set to false.
 *
 */
template<typename tensor_type, typename T>
void check_tensor(tensor_type&& t,
                  const std::vector<T>& corr,
                  bool absolute=false) {
    const double eps = 1000*std::numeric_limits<double>::epsilon();
    const double marg = 100*std::numeric_limits<double>::epsilon();
    const long int n = corr.size();
    std::vector<T> buffer(n);
    tamm::IndexVector IV(t.num_modes()); //Assumes one block
    t.get(IV, gsl::span<T>{buffer.data(), n});
    for(long int i = 0; i < n; ++i) {
        const auto lhs = (!absolute ? buffer[i] : std::fabs(buffer[i]));
        const auto rhs = (!absolute ? corr[i] : std::fabs(corr[i]));
        REQUIRE(lhs == Approx(rhs).epsilon(eps).margin(marg));
    }
}


static auto make_tensor(buffer_type buffer, size_t n) {
    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext ec{pg,&distribution,mgr};
    IndexSpace IS{range(0, n)};
    TiledIndexSpace TIS{IS, n};
    Tensor<double> rv{TIS, TIS};
    Tensor<double>::allocate(&ec, rv);
    const unsigned long int n2 = n * n;
    const IndexVector bID{0,0};
    rv.put(bID, span<double>{buffer.data(), n2});
    return rv;

}

TEST_CASE("GeneralizedSelfAdjointEigenSolver"){

    const std::array<size_t, 2> sizes{3, 7};
    const std::array<std::string, 2> names{"Just S.A.", "General & S.A."};

    SECTION("Run via ctor") {
        for(size_t i=0; i < 2; ++i){
            SECTION(names[i]){
                auto A = make_tensor(input_values[i].at("A"), sizes[i]);
                auto B = make_tensor(input_values[i].at("B"), sizes[i]);
                const auto& space = A.tiled_index_spaces()[0];
                GeneralizedSelfAdjointEigenSolver<double> es(A, B, space);
                const auto& evals = es.eigenvalues();
                const auto& evecs = es.eigenvectors();
                check_tensor(evals, corr_values[i].at("values"));
                check_tensor(evecs, corr_values[i].at("vectors"), true);
            }
        }
    }
    SECTION("Via compute fxn"){
        GeneralizedSelfAdjointEigenSolver<double> es;
        for(size_t i=0; i < 2; ++i){
            SECTION(names[i]){
                auto A = make_tensor(input_values[i].at("A"), sizes[i]);
                auto B = make_tensor(input_values[i].at("B"), sizes[i]);
                const auto& space = A.tiled_index_spaces()[0];
                auto& pes = es.compute(A, B, space);
                REQUIRE(&pes == &es); //check chaining via return
                const auto& evals = es.eigenvalues();
                const auto& evecs = es.eigenvectors();
                check_tensor(evals, corr_values[i].at("values"));
                check_tensor(evecs, corr_values[i].at("vectors"), true);
            }
        }
    }
}
