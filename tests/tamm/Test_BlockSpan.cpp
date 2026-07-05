// Unit tests for tamm::BlockSpan (block_span.hpp).
//
// Previously only exercised via disabled/print-only Test_OpDAG.  Guards the
// rewrite that replaced the (ill-formed dynamic-rank mdspan) implementation with
// pointer + runtime extents:
//  - flat operator[] maps to row-major offset
//  - num_elements(), rank(), extent(d), block_dims(), block_dims_span()
//  - flat_span(), data()/buf() equivalence
//  - make_block_span factory helpers
//  - is_valid()

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <tamm/block_span.hpp>

#include <numeric>
#include <vector>

using namespace tamm;

TEST_CASE("BlockSpan: default is invalid") {
  BlockSpan<double> bs;
  CHECK_FALSE(bs.is_valid());
  CHECK(bs.num_elements() == 0);
}

TEST_CASE("BlockSpan: construction from pointer + dims (vector)") {
  std::vector<double> buf(2 * 3 * 4);
  std::iota(buf.begin(), buf.end(), 0.0);
  std::vector<size_t> dims{2, 3, 4};

  BlockSpan<double> bs{buf.data(), dims};
  CHECK(bs.is_valid());
  CHECK(bs.rank() == 3);
  CHECK(bs.num_elements() == 24);
  CHECK(bs.extent(0) == 2);
  CHECK(bs.extent(1) == 3);
  CHECK(bs.extent(2) == 4);
  CHECK(bs.buf() == buf.data());
  CHECK(bs.data() == buf.data());
}

TEST_CASE("BlockSpan: block_dims() and block_dims_span() report extents") {
  std::vector<float>  buf(6);
  std::vector<size_t> dims{2, 3};
  BlockSpan<float>    bs{buf.data(), dims};

  const auto& bd = bs.block_dims();
  REQUIRE(bd.size() == 2);
  CHECK(bd[0] == 2);
  CHECK(bd[1] == 3);

  auto sp = bs.block_dims_span();
  REQUIRE(sp.size() == 2);
  CHECK(sp[0] == 2);
  CHECK(sp[1] == 3);
}

TEST_CASE("BlockSpan: flat operator[] is row-major over the buffer") {
  // 2x3x4 row-major: element (i,j,k) lives at i*12 + j*4 + k.
  std::vector<int>    buf(2 * 3 * 4);
  std::iota(buf.begin(), buf.end(), 0);
  std::vector<size_t> dims{2, 3, 4};
  BlockSpan<int>      bs{buf.data(), dims};

  for(size_t i = 0; i < 2; i++)
    for(size_t j = 0; j < 3; j++)
      for(size_t k = 0; k < 4; k++) {
        const size_t flat = i * 12 + j * 4 + k;
        CHECK(bs[flat] == static_cast<int>(flat));
      }

  // mutation through operator[]
  bs[5] = 999;
  CHECK(buf[5] == 999);

  // const access
  const BlockSpan<int>& cbs = bs;
  CHECK(cbs[5] == 999);
}

TEST_CASE("BlockSpan: flat_span covers the whole block") {
  std::vector<double> buf(12, 1.5);
  std::vector<size_t> dims{3, 4};
  BlockSpan<double>   bs{buf.data(), dims};

  auto sp = bs.flat_span();
  REQUIRE(sp.size() == 12);
  double sum = std::accumulate(sp.begin(), sp.end(), 0.0);
  CHECK(sum == doctest::Approx(18.0));
}

TEST_CASE("BlockSpan: make_block_span factory (initializer list + range)") {
  std::vector<double> buf(6, 0.0);

  auto bs1 = make_block_span<double>(buf.data(), {2, 3});
  CHECK(bs1.rank() == 2);
  CHECK(bs1.num_elements() == 6);

  std::vector<size_t> dims{3, 2};
  auto                bs2 = make_block_span<double>(buf.data(), dims);
  CHECK(bs2.rank() == 2);
  CHECK(bs2.num_elements() == 6);
}

TEST_CASE("BlockSpan: rank-1 and scalar-like (rank-1 size 1)") {
  std::vector<double> buf{42.0};
  std::vector<size_t> dims{1};
  BlockSpan<double>   bs{buf.data(), dims};
  CHECK(bs.rank() == 1);
  CHECK(bs.num_elements() == 1);
  CHECK(bs[0] == 42.0);
}

TEST_CASE("BlockSpan: copy preserves view") {
  std::vector<int>    buf{1, 2, 3, 4};
  std::vector<size_t> dims{2, 2};
  BlockSpan<int>      a{buf.data(), dims};
  BlockSpan<int>      b = a;
  CHECK(b.buf() == a.buf());
  CHECK(b.num_elements() == 4);
  CHECK(b[3] == 4);
}
