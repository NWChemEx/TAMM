#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <iostream>
#include <tamm/index_loop_nest.hpp>

using namespace tamm;

/*template<typename T>
std::ostream&
operator << (std::ostream& os, const std::vector<T>& vec) {
  os<<"[";
  for(const auto& v: vec) {
    os<<v<<",";
  }
  os<<"]"<<std::endl;
  return os;
}*/

TEST_CASE("Zero-dimensional index loop nest with index bound constructor") {
  IndexLoopNest iln{};
  int           cnt = 0;
  REQUIRE(iln.begin() != iln.end());
  std::for_each(iln.begin(), iln.end(), [&](const auto&) { cnt += 1; });
  REQUIRE(cnt == 1);
}

TEST_CASE("Zero-dimensional index loop nest with list of arguments constructor") {
  IndexLoopNest iln{{}, {}, {}, {}};
  int           cnt = 0;
  REQUIRE(iln.begin() != iln.end());
  std::for_each(iln.begin(), iln.end(), [&](const auto&) { cnt += 1; });
  REQUIRE(cnt == 1);
}

TEST_CASE("One-dimensional index loop nest with index bound constructor") {
  IndexSpace      is{range(10)};
  TiledIndexSpace tis{is, 1};
  TiledIndexLabel i;
  std::tie(i) = tis.labels<1>("all");

  IndexLoopNest iln{i};
  unsigned      cnt = 0;
  for(auto itr = iln.begin(); itr != iln.end(); itr++, cnt++) {
    // std::cout<<"--"<<*itr<<" cnt="<<cnt<<std::endl;
    REQUIRE(*itr == IndexVector{cnt});
  }
  REQUIRE(cnt == 10);
}

TEST_CASE("One-dimensional index loop nest with list of arguments constructor") {
  IndexSpace      is{range(10)};
  TiledIndexSpace tis{is, 1};
  IndexLoopNest   iln{{tis}, {}, {}, {}};
  unsigned        cnt = 0;
  for(auto itr = iln.begin(); itr != iln.end(); itr++, cnt++) {
    // std::cout<<"--"<<it<<std::endl;
    REQUIRE(*itr == IndexVector{cnt});
  }
  REQUIRE(cnt == 10);
}

TEST_CASE("Two-dimensional square index loop nest with index bound constructor") {
  IndexSpace      is{range(10)};
  TiledIndexSpace tis{is, 1};
  TiledIndexLabel i, j;
  std::tie(i, j) = tis.labels<2>("all");
  IndexLoopNest iln{i, j};
  auto          itr = iln.begin();
  for(unsigned ci = 0; ci < 10; ci++) {
    for(unsigned cj = 0; cj < 10; cj++, itr++) {
      REQUIRE(itr != iln.end());
      REQUIRE(*itr == IndexVector{ci, cj});
    }
  }
  REQUIRE(itr == iln.end());
}

TEST_CASE("Two-dimensional square index loop nest with list of arguments constructor") {
  IndexSpace      is{range(10)};
  TiledIndexSpace tis{is, 1};
  IndexLoopNest   iln{{tis, tis}, {}, {}, {}};
  auto            itr = iln.begin();
  for(unsigned ci = 0; ci < 10; ci++) {
    for(unsigned cj = 0; cj < 10; cj++, itr++) {
      REQUIRE(itr != iln.end());
      REQUIRE(*itr == IndexVector{ci, cj});
    }
  }
  REQUIRE(itr == iln.end());
}

TEST_CASE("Two-dimensional rectangular  index loop nest") {
  const unsigned  ri1 = 9, ri2 = 23;
  IndexSpace      is1{range(ri1)}, is2{range(ri2)};
  TiledIndexSpace tis1{is1, 1}, tis2{is2, 1};
  TiledIndexLabel til1, til2;
  std::tie(til1) = tis1.labels<1>("all");
  std::tie(til2) = tis2.labels<1>("all");

  IndexLoopNest iln{til1, til2};
  auto          itr = iln.begin();
  for(unsigned c1 = 0; c1 < ri1; c1++) {
    for(unsigned c2 = 0; c2 < ri2; c2++, itr++) {
      REQUIRE(itr != iln.end());
      REQUIRE(*itr == IndexVector{c1, c2});
    }
  }
  REQUIRE(itr == iln.end());
}

TEST_CASE("Two-dimensional upper triangular index loop nest") {
  const unsigned  ri = 11;
  IndexSpace      is{range(ri)};
  TiledIndexSpace tis{is, 1};
  TiledIndexLabel i, j;
  std::tie(i, j) = tis.labels<2>("all");
  IndexLoopNest iln{i, j + (IndexLoopBound{j} >= i)};
  auto          itr = iln.begin();
  for(unsigned ci = 0; ci < ri; ci++) {
    for(unsigned cj = ci; cj < ri; cj++, itr++) {
      REQUIRE(itr != iln.end());
      REQUIRE(*itr == IndexVector{ci, cj});
    }
  }
  REQUIRE(itr == iln.end());
}

TEST_CASE("Two-dimensional lower triangular index loop nest") {
  const unsigned  ri = 11;
  IndexSpace      is{range(ri)};
  TiledIndexSpace tis{is, 1};
  TiledIndexLabel i, j;
  std::tie(i, j) = tis.labels<2>("all");
  IndexLoopNest iln{i, j + (IndexLoopBound{j} <= i)};
  auto          itr = iln.begin();
  for(unsigned ci = 0; ci < ri; ci++) {
    for(unsigned cj = 0; cj <= ci; cj++, itr++) {
      REQUIRE(itr != iln.end());
      REQUIRE(*itr == IndexVector{ci, cj});
    }
  }
  REQUIRE(itr == iln.end());
}

TEST_CASE("Two-dimensional diagonal index loop nest") {
  const unsigned  ri = 11;
  IndexSpace      is{range(ri)};
  TiledIndexSpace tis{is, 1};
  TiledIndexLabel i, j;
  std::tie(i, j) = tis.labels<2>("all");
  IndexLoopNest iln{i, j + (IndexLoopBound{j} <= i) + (IndexLoopBound{j} >= i)};
  auto          itr = iln.begin();
  for(unsigned ci = 0; ci < ri; ci++, itr++) {
    REQUIRE(itr != iln.end());
    REQUIRE(*itr == IndexVector{ci, ci});
  }
  REQUIRE(itr == iln.end());
}

TEST_CASE("Three-dimensional diagonal index loop nest") {
  const unsigned  ri = 11;
  IndexSpace      is{range(ri)};
  TiledIndexSpace tis{is, 1};
  TiledIndexLabel i, j, k;
  std::tie(i, j, k) = tis.labels<3>("all");
  IndexLoopNest iln{i, j + (IndexLoopBound{j} <= i) + (IndexLoopBound{j} >= i),
                    IndexLoopBound{k} == j};
  auto          itr = iln.begin();
  for(unsigned ci = 0; ci < ri; ci++, itr++) {
    REQUIRE(itr != iln.end());
    REQUIRE(*itr == IndexVector{ci, ci, ci});
  }
  REQUIRE(itr == iln.end());
}

TEST_CASE("One-dimensional split index loop nest with index bound constructor") {
  IndexSpace      is{range(20), {{"r1", {range(0, 10)}}, {"r2", {range(10, 20)}}}};
  TiledIndexSpace tis{is, 1};
  TiledIndexLabel i;
  std::tie(i) = tis.labels<1>("all");

  IndexLoopNest iln{i};
  unsigned      cnt = 0;
  for(auto itr = iln.begin(); itr != iln.end(); itr++, cnt++) {
    // std::cout<<"--"<<it<<std::endl;
    REQUIRE(*itr == IndexVector{cnt});
  }
  REQUIRE(cnt == 20);
}

TEST_CASE("Two-dimensional split index loop nest with index bound constructor") {
  IndexSpace      is{range(20), {{"r1", {range(0, 10)}}, {"r2", {range(10, 20)}}}};
  TiledIndexSpace tis{is, 10};
  TiledIndexLabel i, j;
  std::tie(i, j) = tis.labels<2>("all");
  IndexLoopNest iln{i, j};
  auto          itr = iln.begin();
  for(unsigned c1 = 0; c1 < 20 / 10; c1++) {
    for(unsigned c2 = 0; c2 < 20 / 10; c2++, itr++) {
      REQUIRE(itr != iln.end());
      REQUIRE(*itr == IndexVector{c1, c2});
    }
  }
  REQUIRE(itr == iln.end());
}

TEST_CASE("Three-dimensional split index loop nest with index bound constructor") {
  Tile            tilesize = 10;
  IndexSpace      is{range(20), {{"r1", {range(0, 10)}}, {"r2", {range(10, 20)}}}};
  TiledIndexSpace tis{is, tilesize};
  TiledIndexLabel i, j, k;
  std::tie(i, j, k) = tis.labels<3>("all");
  IndexLoopNest iln{i, j, k};
  auto          itr = iln.begin();
  for(unsigned c1 = 0; c1 < 20 / tilesize; c1++) {
    for(unsigned c2 = 0; c2 < 20 / tilesize; c2++) {
      for(unsigned c3 = 0; c3 < 20 / tilesize; c3++, itr++) {
        REQUIRE(itr != iln.end());
        REQUIRE(*itr == IndexVector{c1, c2, c3});
      }
    }
  }
  REQUIRE(itr == iln.end());
}

TEST_CASE("Four-dimensional split index loop nest with index bound constructor") {
  Tile            tilesize = 10;
  IndexSpace      is{range(20), {{"r1", {range(0, 10)}}, {"r2", {range(10, 20)}}}};
  TiledIndexSpace tis{is, tilesize};
  TiledIndexLabel i, j, k, l;
  std::tie(i, j, k, l) = tis.labels<4>("all");
  IndexLoopNest iln{i, j, k, l};
  auto          itr = iln.begin();
  for(unsigned c1 = 0; c1 < 20 / tilesize; c1++) {
    for(unsigned c2 = 0; c2 < 20 / tilesize; c2++) {
      for(unsigned c3 = 0; c3 < 20 / tilesize; c3++) {
        for(unsigned c4 = 0; c4 < 20 / tilesize; c4++, itr++) {
          REQUIRE(itr != iln.end());
          REQUIRE(*itr == IndexVector{c1, c2, c3, c4});
        }
      }
    }
  }
  REQUIRE(itr == iln.end());
}
