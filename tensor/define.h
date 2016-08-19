#ifndef __ctce_define_h__
#define __ctce_define_h__

#include <vector>
#include <set>
#include <cstdarg>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace ctce {

/* range of the indices, also used in triangular iterator */
typedef enum {
  TO, TV, TN, RANGE_UB
} RangeType;

typedef enum {
  P1B, P2B, P3B, P4B, P5B, P6B, P7B, P8B, P9B, P10B, P11B, P12B,
  H1B, H2B, H3B, H4B, H5B, H6B, H7B, H8B, H9B, H10B, H11B, H12B
} IndexName;

typedef enum {
  pIndex, hIndex
} IndexType;

const static int pIndexNum = 12;
const static int hIndexNum = 12;
const static int IndexNum = pIndexNum + hIndexNum;

/* cout vector */
template<typename T>
std::ostream&
operator<<(std::ostream& os, const std::vector<T>& v) {
  os<<"(";
  for(typename std::vector<T>::const_iterator itr=v.begin(); itr!=v.end(); ++itr) os << *itr <<" ";
  os<<")";
  return os;
}

/* easy way to generate a new vector, works for primitive data type (not string) */
template <typename T>
std::vector<T>
newVec(int n, ...) {
  std::vector<T> v;
  va_list ap;
  v.resize(n);
  va_start(ap, n);
  for(int i=0; i<n; i++) v[i]=va_arg(ap, T);
  va_end(ap);
  return v;
}

/* compare if v1 and v2 are exactly the same */
template <typename T>
bool
compareVec(const std::vector<T>& v1, const std::vector<T>& v2) {
  if (v1.size()!=v2.size()) return false;
  for (int i=0; i<v1.size(); i++) {
    if (v1[i]!=v2[i]) return false;
  }
  return true;
}

/* compare v1, v2 to get the sign */
template <typename T>
int
countParitySign(const std::vector<T>& v1, std::vector<T> v2) {
  int parity = 0;
  for (int i=0; i<v1.size(); i++) {
    if (v1[i]!=v2[i]) {
      int swap_i = std::find(v2.begin(),v2.end(),v1[i])-v2.begin();
      v2[swap_i] = v2[i];
      v2[i] = v1[i];
      parity ++;
    }
  }
  if (parity%2 == 0) return 1;
  return -1;
}

inline bool
is_permutation(const std::vector<IndexName>& ids) {
  std::set<IndexName> sids(ids.begin(), ids.end());
  return sids.size() == ids.size();
}

inline bool
is_permutation(const std::vector<IndexName>& ids1,
               const std::vector<IndexName>& ids2) {
  std::set<IndexName> sids1(ids1.begin(), ids1.end());
  std::set<IndexName> sids2(ids2.begin(), ids2.end());
  if(ids1.size() != sids1.size()) return false;
  if(ids2.size() != sids2.size()) return false;
  for (int i=0; i<ids1.size(); i++) {
    if(sids2.find(ids1[i]) == sids2.end())
      return false;
  }
  return true;
}

inline std::vector<IndexName>
ivec(IndexName i1) {
  std::vector<IndexName> ret;
  ret.push_back(i1);
  return ret;
}

inline std::vector<IndexName>
ivec(IndexName i1, IndexName i2) {
  std::vector<IndexName> ret = ivec(i1);
  ret.push_back(i2);
  return ret;
}

inline std::vector<IndexName>
ivec(IndexName i1, IndexName i2, IndexName i3) {
  std::vector<IndexName> ret = ivec(i1,i2);
  ret.push_back(i3);
  return ret;
}

inline std::vector<IndexName>
ivec(IndexName i1, IndexName i2, IndexName i3, IndexName i4) {
  std::vector<IndexName> ret = ivec(i1,i2,i3);
  ret.push_back(i4);
  return ret;
}

}

#endif


