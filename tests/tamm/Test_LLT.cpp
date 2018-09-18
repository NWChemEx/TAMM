#include <tamm/cholesky.hpp>
#include "test_tamm.hpp"

using namespace tamm;

//Type used to store correct answer's values
using buffer_type = std::vector<double>;
//Type used to associate labels with correct answers
using map_type = std::map<std::string, buffer_type>;

static const std::vector<buffer_type> input_values {
    {4.0, -1.0, 2.0, -1.0, 6.0, 0.0, 2.0, 0.0, 5.0},
};

static const std::vector<map_type> corr_values {
    {//start map 1
        {//start pair 1
            "L", { 2.0000000000000000, 0.0000000000000000, 0.0000000000000000,
                  -0.5000000000000000, 2.3979157616563596, 0.0000000000000000,
                   1.0000000000000000, 0.2085144140570748, 1.9891007362952824}
        },//end pair 1
        {//start pair 2
            "U", {2.0000000000000000, -0.5000000000000000, 1.0000000000000000,
                  0.0000000000000000, 2.3979157616563596, 0.2085144140570748,
                  0.0000000000000000, 0.0000000000000000, 1.9891007362952824}
        }
    }//end map 1
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

TEST_CASE("LLT Factorization") {
    SECTION("Run via ctor") {
        auto M = make_tensor(input_values[0], 3);
        LLT<double> llt(M);
        check_tensor(llt.matrix_L(), corr_values[0].at("L"));
        check_tensor(llt.matrix_U(), corr_values[0].at("U"));
    }
    SECTION("Run via compute"){
        auto M = make_tensor(input_values[0], 3);
        LLT<double> llt;
        auto& pllt = llt.compute(M);
        REQUIRE(&pllt == &llt);
        check_tensor(llt.matrix_L(), corr_values[0].at("L"));
        check_tensor(llt.matrix_U(), corr_values[0].at("U"));
    }
}
