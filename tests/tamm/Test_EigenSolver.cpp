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
            {-0.9999708931193276, 0.1973820032468157, -0.0000000000000002,
              0.0847003495442780, 0.0000000000000000, 0.0396236486911344,
             -0.0000000000000012, 0.0071782980842154, -0.8938202933706008,
              0.0000000000000008, -0.3559754106265013, 0.0000000000000000,
             -0.3805375696821480, 0.0000000000000112, 0.0002667154467560,
             -0.3610027645129435, 0.0000000000000002, 0.9291427106298554,
              0.0000000000000000, -0.1749676292001189, 0.0000000000000052,
             -0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
              0.0000000000000000, 1.0000000000000000, 0.0000000000000000,
              0.0000000000000000, -0.0000000000000000, 0.0000000000000001,
             -0.9835676268102400, -0.0000000000000001, 0.0000000000000000,
             -0.0000000000000002, 0.2394467207324156, -0.0018184902555526,
              0.1261004202355791, 0.1276611207276361, -0.0374244645905306,
              0.0000000000000000, 0.6414855622915050, -0.6865366952794819,
             -0.0018184902555526, 0.1261004202355790, -0.1276611207276360,
             -0.0374244645905292, 0.0000000000000000, 0.6414855622915042,
              0.6865366952794442}
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
                GeneralizedSelfAdjointEigenSolver<double> es(A, B);
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
                auto& pes = es.compute(A, B);
                REQUIRE(&pes == &es); //check chaining via return
                const auto& evals = es.eigenvalues();
                const auto& evecs = es.eigenvectors();
                check_tensor(evals, corr_values[i].at("values"));
                check_tensor(evecs, corr_values[i].at("vectors"), true);
            }
        }
    }
}
