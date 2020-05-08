#ifndef CCSD_T_ALL_FUSED_DOUBLES2_HPP_
#define CCSD_T_ALL_FUSED_DOUBLES2_HPP_

#include "tamm/tamm.hpp"
// using namespace tamm;

extern double ccsdt_d2_t2_GetTime;
extern double ccsdt_d2_v2_GetTime;
extern double ccsd_t_data_per_rank;

#define TEST_NEW_KERNEL
#define TEST_NEW_THREAD

// singles data driver
template <typename T>
void ccsd_t_data_d2(
    ExecutionContext& ec, const TiledIndexSpace& MO, const Index noab,
    const Index nvab, std::vector<int>& k_spin, std::vector<size_t>& k_offset,
    Tensor<T>& d_t1,
    Tensor<T>& d_t2,  // d_a
    Tensor<T>& d_v2,  // d_b
    std::vector<T>& k_evl_sorted, std::vector<size_t>& k_range, size_t t_h1b,
    size_t t_h2b, size_t t_h3b, size_t t_p4b, size_t t_p5b, size_t t_p6b,
    std::vector<T>& k_abuf2, std::vector<T>& k_bbuf2, 
    std::vector<int>& d2_flags, std::vector<int>& d2_sizes,
    T* T_d2_t2, T* T_d2_v2, 
    std::vector<int>& sd_t_d2_exec, std::vector<int>& d2_sizes_ext,
    // std::vector<size_t>& sd_t_d2_exec, std::vector<size_t>& d2_sizes_ext,
    bool is_restricted, LRUCache<Index,std::vector<T>>& cache_d2t, LRUCache<Index,std::vector<T>>& cache_d2v) {

  size_t abuf_size2 = k_abuf2.size();
  size_t bbuf_size2 = k_bbuf2.size();

  std::tuple<Index, Index, Index, Index, Index, Index> a3_d2[] = {
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h2b, t_h3b, t_h1b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h3b, t_h2b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h2b, t_h3b, t_h1b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h1b, t_h3b, t_h2b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h2b, t_h3b, t_h1b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h1b, t_h3b, t_h2b)};


  for (auto ia6 = 0; ia6 < 8; ia6++) {
    if (std::get<0>(a3_d2[ia6]) != 0) {
      for (auto ja6 = ia6 + 1; ja6 < 9; ja6++) {  // TODO: ja6 start ?
        if (a3_d2[ia6] == a3_d2[ja6]) {
          a3_d2[ja6] = std::make_tuple(0,0,0,0,0,0);
        }
      }
    }
  }

  const size_t max_dima2 = abuf_size2 / (9*nvab);
  const size_t max_dimb2 = bbuf_size2 / (9*nvab);

  //doubles 2
  // std::vector<size_t> sd_t_d2_exec(9*9*nvab,-1);
  // std::vector<size_t> d2_sizes_ext(9*7*nvab);

  // size_t d2b = 0;
  int d2b = 0;

  for (auto ia6 = 0; ia6 < 9; ia6++) {
    for (Index p7b=noab;p7b<noab+nvab;p7b++) {
      d2_sizes_ext[0 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_h1b];
      d2_sizes_ext[1 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_h2b];
      d2_sizes_ext[2 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_h3b];
      d2_sizes_ext[3 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_p4b];
      d2_sizes_ext[4 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_p5b];
      d2_sizes_ext[5 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_p6b];
      d2_sizes_ext[6 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[p7b];
    }
  }

#ifdef TEST_NEW_KERNEL
  for (Index p7b=noab;p7b<noab+nvab;p7b++) {
    d2_sizes[0 + (p7b - noab) * 7] = (int)k_range[t_h1b];
    d2_sizes[1 + (p7b - noab) * 7] = (int)k_range[t_h2b];
    d2_sizes[2 + (p7b - noab) * 7] = (int)k_range[t_h3b];
    d2_sizes[3 + (p7b - noab) * 7] = (int)k_range[t_p4b];
    d2_sizes[4 + (p7b - noab) * 7] = (int)k_range[t_p5b];
    d2_sizes[5 + (p7b - noab) * 7] = (int)k_range[t_p6b];
    d2_sizes[6 + (p7b - noab) * 7] = (int)k_range[p7b];
  }
#endif

  std::vector<bool> ia6_enabled(9*nvab,false);
  
  //ia6 -- compute which variants are enabled
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];

    if (!((p5b <= p6b) && (h1b <= h2b) && p4b != 0)) {
      continue;
    }
    if (is_restricted && !(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] + k_spin[h1b] + k_spin[h2b] +
               k_spin[h3b] !=
           12)) {
      continue;
    }
    if(!(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] ==
            k_spin[h1b] + k_spin[h2b] + k_spin[h3b])) {
      continue;
    }

    for (Index p7b=noab;p7b<noab+nvab;p7b++) {
      if (!(k_spin[p4b]+k_spin[p7b] == k_spin[h1b]+k_spin[h2b])) {
        continue;
      }
      if (!(k_range[p4b] > 0 && k_range[p5b] > 0 && k_range[p6b] > 0 &&
            k_range[h1b] > 0 && k_range[h2b] > 0 && k_range[h3b] > 0)) {
        continue;
      }
      if(!(h3b <= p7b)) continue;
      ia6_enabled[ia6*nvab + p7b-noab] = true;
    } //end h7b

  } // end ia6

  //ia6 -- compute sizes and permutations
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];
    auto ref_p456_h123 = std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h312 = std::make_tuple(p4b, p5b, p6b, h3b, h1b, h2b);
    auto cur_p456_h132 = std::make_tuple(p4b, p5b, p6b, h1b, h3b, h2b);
    auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    auto cur_p546_h312 = std::make_tuple(p5b, p4b, p6b, h3b, h1b, h2b);
    auto cur_p546_h132 = std::make_tuple(p5b, p4b, p6b, h1b, h3b, h2b);
    auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    auto cur_p564_h312 = std::make_tuple(p5b, p6b, p4b, h3b, h1b, h2b);
    auto cur_p564_h132 = std::make_tuple(p5b, p6b, p4b, h1b, h3b, h2b);

    for (Index p7b=noab;p7b<noab+nvab;p7b++) 
    {
      if(!ia6_enabled[ia6*nvab + p7b-noab]) continue;
    
      if (ref_p456_h123 == cur_p456_h123) {
        sd_t_d2_exec[0 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p456_h312) {
        sd_t_d2_exec[1 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p456_h132) {
        sd_t_d2_exec[2 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p546_h123) {
        sd_t_d2_exec[3 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p546_h312) {
        sd_t_d2_exec[4 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p546_h132) {
        sd_t_d2_exec[5 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p564_h123) {
        sd_t_d2_exec[6 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p564_h312) {
        sd_t_d2_exec[7 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p564_h132) {
        sd_t_d2_exec[8 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
    } //p7b
  }  // end ia6

  //ia6 -- get for t2
  d2b = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];

    for (Index p7b=noab;p7b<noab+nvab;p7b++) 
    {
      if(!ia6_enabled[ia6*nvab + p7b-noab]) continue;

      size_t dim_common = k_range[p7b];
      size_t dima_sort = k_range[p4b]*k_range[h1b]*k_range[h2b];
      size_t dima = dim_common*dima_sort;

      std::vector<T> k_a(dima);
      std::vector<T> k_a_sort(dima);

    IndexVector a_bids_minus_sidx = {p7b-noab,p4b - noab, h1b, h2b};
    auto [hit, value] = cache_d2t.log_access(a_bids_minus_sidx);
    if (hit) {
    // if (false) {
      // std::copy(value.begin(), value.end(), k_abuf2.begin() + d2b * max_dima2);
      // d2b += value.size() / max_dima2;
      k_a_sort = value;
    } else {
      if(p7b<p4b) 
      {
        {
        TimerGuard tg_total{&ccsdt_d2_t2_GetTime};   
        ccsd_t_data_per_rank += dima;                
        d_t2.get({p7b-noab,p4b-noab,h1b,h2b},k_a); //h2b,h1b,p4b-noab,p7b-noab
        }
        // for (auto x=0;x<dima;x++) k_a_sort[x] = -1 * k_a[x];
        int perm[4]={3,2,1,0};
        int size[4]={(int)k_range[p7b],(int)k_range[p4b],(int)k_range[h1b],(int)k_range[h2b]};
        
        auto plan = hptt::create_plan
        (perm, 4, -1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
            NULL, hptt::ESTIMATE, 1, NULL, true);
        plan->execute();
      }
      if(p4b<=p7b) 
      {
        {
        TimerGuard tg_total{&ccsdt_d2_t2_GetTime};  
        ccsd_t_data_per_rank += dima;                 
        d_t2.get({p4b-noab,p7b-noab,h1b,h2b},k_a); //h2b,h1b,p7b-noab,p4b-noab
        }
        int perm[4]={3,2,0,1}; //0,1,3,2
        int size[4]={(int)k_range[p4b],(int)k_range[p7b],(int)k_range[h1b],(int)k_range[h2b]};
        
        auto plan = hptt::create_plan
        (perm, 4, 1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
            NULL, hptt::ESTIMATE, 1, NULL, true);
        plan->execute();
      }
      value = k_a_sort;
    }

      auto ref_p456_h123 =
          std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
      auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
      auto cur_p456_h312 = std::make_tuple(p4b, p5b, p6b, h3b, h1b, h2b);
      auto cur_p456_h132 = std::make_tuple(p4b, p5b, p6b, h1b, h3b, h2b);
      auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
      auto cur_p546_h312 = std::make_tuple(p5b, p4b, p6b, h3b, h1b, h2b);
      auto cur_p546_h132 = std::make_tuple(p5b, p4b, p6b, h1b, h3b, h2b);
      auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
      auto cur_p564_h312 = std::make_tuple(p5b, p6b, p4b, h3b, h1b, h2b);
      auto cur_p564_h132 = std::make_tuple(p5b, p6b, p4b, h1b, h3b, h2b);

      if (ref_p456_h123 == cur_p456_h123) {
        std::copy(k_a_sort.begin(), k_a_sort.end(),
                  // k_abuf2.begin() + d2b * max_dima2);
                  T_d2_t2 + d2b * max_dima2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p456_h312) {
        std::copy(k_a_sort.begin(), k_a_sort.end(),
                  // k_abuf2.begin() + d2b * max_dima2);
                  T_d2_t2 + d2b * max_dima2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p456_h132) {
        std::copy(k_a_sort.begin(), k_a_sort.end(),
                  // k_abuf2.begin() + d2b * max_dima2);
                  T_d2_t2 + d2b * max_dima2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p546_h123) {
        std::copy(k_a_sort.begin(), k_a_sort.end(),
                  // k_abuf2.begin() + d2b * max_dima2);
                  T_d2_t2 + d2b * max_dima2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p546_h312) {
        std::copy(k_a_sort.begin(), k_a_sort.end(),
                  // k_abuf2.begin() + d2b * max_dima2);
                  T_d2_t2 + d2b * max_dima2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p546_h132) {
        std::copy(k_a_sort.begin(), k_a_sort.end(),
                  // k_abuf2.begin() + d2b * max_dima2);
                  T_d2_t2 + d2b * max_dima2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p564_h123) {
        std::copy(k_a_sort.begin(), k_a_sort.end(),
                  // k_abuf2.begin() + d2b * max_dima2);
                  T_d2_t2 + d2b * max_dima2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p564_h312) {
        std::copy(k_a_sort.begin(), k_a_sort.end(),
                  // k_abuf2.begin() + d2b * max_dima2);
                  T_d2_t2 + d2b * max_dima2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p564_h132) {
        std::copy(k_a_sort.begin(), k_a_sort.end(),
                  // k_abuf2.begin() + d2b * max_dima2);
                  T_d2_t2 + d2b * max_dima2);
        d2b++;
      }
    } //p7b
    // auto abuf_end = d2b * max_dima2;
    // value.clear();
    // value.insert(value.end(), k_abuf2.begin() + abuf_start,
    //              k_abuf2.begin() + abuf_end);
    // }  // else cache
  }  // end ia6

  //ia6 -- get for v2
  d2b = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];

    for (Index p7b=noab;p7b<noab+nvab;p7b++) 
    {
      if(!ia6_enabled[ia6*nvab + p7b-noab]) continue;

      size_t dim_common = k_range[p7b];
      size_t dimb_sort = k_range[p5b]*k_range[p6b]*k_range[h3b];
      size_t dimb = dim_common*dimb_sort;

      std::vector<T> k_b_sort(dimb);

      IndexVector b_bids_minus_sidx = {p5b, p6b, h3b, p7b};
      auto [hit, value] = cache_d2v.log_access(b_bids_minus_sidx);
      
      if (hit) {
      // if (false) {
        // std::copy(value.begin(), value.end(), k_bbuf2.begin() + d2b * max_dimb2);
        // d2b += value.size() / max_dimb2;
        k_b_sort = value;
      } else {
        // auto bbuf_start = d2b * max_dimb2;
        {
          TimerGuard tg_total{&ccsdt_d2_v2_GetTime};  
          ccsd_t_data_per_rank += dimb;                 
          d_v2.get({p5b,p6b,h3b,p7b},k_b_sort); //p7b,h3b,p6b,p5b
        }
        value = k_b_sort;
      }

      auto ref_p456_h123 =
          std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h312 = std::make_tuple(p4b, p5b, p6b, h3b, h1b, h2b);
    auto cur_p456_h132 = std::make_tuple(p4b, p5b, p6b, h1b, h3b, h2b);
    auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    auto cur_p546_h312 = std::make_tuple(p5b, p4b, p6b, h3b, h1b, h2b);
    auto cur_p546_h132 = std::make_tuple(p5b, p4b, p6b, h1b, h3b, h2b);
    auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    auto cur_p564_h312 = std::make_tuple(p5b, p6b, p4b, h3b, h1b, h2b);
    auto cur_p564_h132 = std::make_tuple(p5b, p6b, p4b, h1b, h3b, h2b);

      if (ref_p456_h123 == cur_p456_h123) {
        std::copy(k_b_sort.begin(), k_b_sort.end(),
                  // k_bbuf2.begin() + d2b * max_dimb2);
                  T_d2_v2 + d2b * max_dimb2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p456_h312) {
        std::copy(k_b_sort.begin(), k_b_sort.end(),
                  // k_bbuf2.begin() + d2b * max_dimb2);
                  T_d2_v2 + d2b * max_dimb2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p456_h132) {
        std::copy(k_b_sort.begin(), k_b_sort.end(),
                  // k_bbuf2.begin() + d2b * max_dimb2);
                  T_d2_v2 + d2b * max_dimb2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p546_h123) {
        std::copy(k_b_sort.begin(), k_b_sort.end(),
                  // k_bbuf2.begin() + d2b * max_dimb2);
                  T_d2_v2 + d2b * max_dimb2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p546_h312) {
        std::copy(k_b_sort.begin(), k_b_sort.end(),
                  // k_bbuf2.begin() + d2b * max_dimb2);
                  T_d2_v2 + d2b * max_dimb2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p546_h132) {
        std::copy(k_b_sort.begin(), k_b_sort.end(),
                  // k_bbuf2.begin() + d2b * max_dimb2);
                  T_d2_v2 + d2b * max_dimb2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p564_h123) {
        std::copy(k_b_sort.begin(), k_b_sort.end(),
                  // k_bbuf2.begin() + d2b * max_dimb2);
                  T_d2_v2 + d2b * max_dimb2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p564_h312) {
        std::copy(k_b_sort.begin(), k_b_sort.end(),
                  // k_bbuf2.begin() + d2b * max_dimb2);
                  T_d2_v2 + d2b * max_dimb2);
        d2b++;
      }
      if (ref_p456_h123 == cur_p564_h132) {
        std::copy(k_b_sort.begin(), k_b_sort.end(),
                  // k_bbuf2.begin() + d2b * max_dimb2);
                  T_d2_v2 + d2b * max_dimb2);
        d2b++;
      }
    } //p7b
    // auto bbuf_end = d2b * max_dimb2;
    // value.clear();
    // value.insert(value.end(), k_bbuf2.begin() + bbuf_start,
    //              k_bbuf2.end() + bbuf_end);
    // }  // else cache
  }  // end ia6
  // return s1b;
}  // ccsd_t_data_s1

// singles data driver
template <typename T>
void ccsd_t_data_d2_new(bool is_restricted,
    // ExecutionContext& ec, const TiledIndexSpace& MO, 
    const Index noab, const Index nvab, 
    std::vector<int>& k_spin, 
    // std::vector<size_t>& k_offset,
    Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_v2, 
    std::vector<T>& k_evl_sorted, std::vector<size_t>& k_range, 
    size_t t_h1b, size_t t_h2b, size_t t_h3b, 
    size_t t_p4b, size_t t_p5b, size_t t_p6b,
    size_t max_d2_kernels_pertask,
    // 
    size_t size_T_d2_t2,    size_t size_T_d2_v2, 
    // T* T_d2_t2,             T* T_d2_v2, 
    T* df_T_d2_t2,          T* df_T_d2_v2, 
    // 
    // int* df_d2_exec,        int* df_d2_size, 
    // 
    int* df_simple_d2_size, int* df_simple_d2_exec, 
    int* df_num_d2_enabled, 
    // 
    LRUCache<Index,std::vector<T>>& cache_d2t, LRUCache<Index,std::vector<T>>& cache_d2v) {
    
  // 
  size_t abuf_size2 = size_T_d2_t2; //k_abuf2.size();
  size_t bbuf_size2 = size_T_d2_v2; //k_bbuf2.size();

  std::tuple<Index, Index, Index, Index, Index, Index> a3_d2[] = {
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h2b, t_h3b, t_h1b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h3b, t_h2b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h2b, t_h3b, t_h1b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h1b, t_h3b, t_h2b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h2b, t_h3b, t_h1b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h1b, t_h3b, t_h2b)};


  for (auto ia6 = 0; ia6 < 8; ia6++) {
    if (std::get<0>(a3_d2[ia6]) != 0) {
      for (auto ja6 = ia6 + 1; ja6 < 9; ja6++) {  // TODO: ja6 start ?
        if (a3_d2[ia6] == a3_d2[ja6]) {
          a3_d2[ja6] = std::make_tuple(0,0,0,0,0,0);
        }
      }
    }
  }

  const size_t max_dima2 = abuf_size2 / max_d2_kernels_pertask;
  const size_t max_dimb2 = bbuf_size2 / max_d2_kernels_pertask;

  //doubles 2
  // std::vector<size_t> sd_t_d2_exec(9*9*nvab,-1);
  // std::vector<size_t> d2_sizes_ext(9*7*nvab);

  // size_t d2b = 0;
  // int d2b = 0;

#if 0
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    for (Index p7b=noab;p7b<noab+nvab;p7b++) {
      df_d2_size[0 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_h1b];
      df_d2_size[1 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_h2b];
      df_d2_size[2 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_h3b];
      df_d2_size[3 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_p4b];
      df_d2_size[4 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_p5b];
      df_d2_size[5 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[t_p6b];
      df_d2_size[6 + (p7b - noab + (ia6) * nvab) * 7] = (int)k_range[p7b];
    }
  }
#endif

  for (Index p7b=noab;p7b<noab+nvab;p7b++) {
    df_simple_d2_size[0 + (p7b - noab) * 7] = (int)k_range[t_h1b];
    df_simple_d2_size[1 + (p7b - noab) * 7] = (int)k_range[t_h2b];
    df_simple_d2_size[2 + (p7b - noab) * 7] = (int)k_range[t_h3b];
    df_simple_d2_size[3 + (p7b - noab) * 7] = (int)k_range[t_p4b];
    df_simple_d2_size[4 + (p7b - noab) * 7] = (int)k_range[t_p5b];
    df_simple_d2_size[5 + (p7b - noab) * 7] = (int)k_range[t_p6b];
    df_simple_d2_size[6 + (p7b - noab) * 7] = (int)k_range[p7b];
  }


  std::vector<bool> ia6_enabled(9*nvab,false);
  
  //ia6 -- compute which variants are enabled
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];

    if (!((p5b <= p6b) && (h1b <= h2b) && p4b != 0)) {
      continue;
    }
    if (is_restricted && !(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] + k_spin[h1b] + k_spin[h2b] +
               k_spin[h3b] !=
           12)) {
      continue;
    }
    if(!(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] ==
            k_spin[h1b] + k_spin[h2b] + k_spin[h3b])) {
      continue;
    }

    for (Index p7b=noab;p7b<noab+nvab;p7b++) {
      if (!(k_spin[p4b]+k_spin[p7b] == k_spin[h1b]+k_spin[h2b])) {
        continue;
      }
      if (!(k_range[p4b] > 0 && k_range[p5b] > 0 && k_range[p6b] > 0 &&
            k_range[h1b] > 0 && k_range[h2b] > 0 && k_range[h3b] > 0)) {
        continue;
      }
      if(!(h3b <= p7b)) continue;
      ia6_enabled[ia6*nvab + p7b-noab] = true;
    } //end h7b

  } // end ia6

  //ia6 -- compute sizes and permutations
  int idx_offset = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];
    auto ref_p456_h123 = std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h312 = std::make_tuple(p4b, p5b, p6b, h3b, h1b, h2b);
    auto cur_p456_h132 = std::make_tuple(p4b, p5b, p6b, h1b, h3b, h2b);
    auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    auto cur_p546_h312 = std::make_tuple(p5b, p4b, p6b, h3b, h1b, h2b);
    auto cur_p546_h132 = std::make_tuple(p5b, p4b, p6b, h1b, h3b, h2b);
    auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    auto cur_p564_h312 = std::make_tuple(p5b, p6b, p4b, h3b, h1b, h2b);
    auto cur_p564_h132 = std::make_tuple(p5b, p6b, p4b, h1b, h3b, h2b);

    for (Index p7b=noab;p7b<noab+nvab;p7b++) 
    {
      if(!ia6_enabled[ia6*nvab + p7b-noab]) continue;
    
      if (ref_p456_h123 == cur_p456_h123) {
        // df_d2_exec[0 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
        df_simple_d2_exec[0 + (p7b - noab) * 9] = idx_offset;
        // sd_t_d2_exec[0 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p456_h312) {
        // df_d2_exec[1 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
        df_simple_d2_exec[1 + (p7b - noab) * 9] = idx_offset;
        // sd_t_d2_exec[1 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p456_h132) {
        // df_d2_exec[2 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
        df_simple_d2_exec[2 + (p7b - noab) * 9] = idx_offset;
        // sd_t_d2_exec[2 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p546_h123) {
        // df_d2_exec[3 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
        df_simple_d2_exec[3 + (p7b - noab) * 9] = idx_offset;
        // sd_t_d2_exec[3 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p546_h312) {
        // df_d2_exec[4 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
        df_simple_d2_exec[4 + (p7b - noab) * 9] = idx_offset;
        // sd_t_d2_exec[4 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p546_h132) {
        // df_d2_exec[5 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
        df_simple_d2_exec[5 + (p7b - noab) * 9] = idx_offset;
        // sd_t_d2_exec[5 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p564_h123) {
        // df_d2_exec[6 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
        df_simple_d2_exec[6 + (p7b - noab) * 9] = idx_offset;
        // sd_t_d2_exec[6 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p564_h312) {
        // df_d2_exec[7 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
        df_simple_d2_exec[7 + (p7b - noab) * 9] = idx_offset;
        // sd_t_d2_exec[7 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }
      if (ref_p456_h123 == cur_p564_h132) {
        // df_d2_exec[8 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
        df_simple_d2_exec[8 + (p7b - noab) * 9] = idx_offset;
        // sd_t_d2_exec[8 + (p7b - noab + (ia6) * nvab) * 9] = d2b++;
      }

      // 
      idx_offset++;
    } //p7b
  } // end ia6

  //ia6 -- get for t2
  // d2b = 0;
  idx_offset = 0;
  // printf ("[%s] ---------------------------------------------------------------\n", __func__);
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];

    for (Index p7b=noab;p7b<noab+nvab;p7b++) 
    {
      if(!ia6_enabled[ia6*nvab + p7b-noab]) continue;

      size_t dim_common = k_range[p7b];
      size_t dima_sort = k_range[p4b]*k_range[h1b]*k_range[h2b];
      size_t dima = dim_common*dima_sort;

      std::vector<T> k_a(dima);
      std::vector<T> k_a_sort(dima);

      IndexVector a_bids_minus_sidx = {p7b-noab,p4b - noab, h1b, h2b};
      auto [hit, value] = cache_d2t.log_access(a_bids_minus_sidx);
      if (hit) {
        k_a_sort = value;
      } else {
        if(p7b<p4b) 
        {
          {
          TimerGuard tg_total{&ccsdt_d2_t2_GetTime};   
          ccsd_t_data_per_rank += dima;                
          d_t2.get({p7b-noab,p4b-noab,h1b,h2b},k_a); //h2b,h1b,p4b-noab,p7b-noab
          }
          // for (auto x=0;x<dima;x++) k_a_sort[x] = -1 * k_a[x];
          int perm[4]={3,2,1,0};
          int size[4]={(int)k_range[p7b],(int)k_range[p4b],(int)k_range[h1b],(int)k_range[h2b]};
          
          auto plan = hptt::create_plan
          (perm, 4, -1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
              NULL, hptt::ESTIMATE, 1, NULL, true);
          plan->execute();
        }
        if(p4b<=p7b) 
        {
          {
          TimerGuard tg_total{&ccsdt_d2_t2_GetTime};  
          ccsd_t_data_per_rank += dima;                 
          d_t2.get({p4b-noab,p7b-noab,h1b,h2b},k_a); //h2b,h1b,p7b-noab,p4b-noab
          }
          int perm[4]={3,2,0,1}; //0,1,3,2
          int size[4]={(int)k_range[p4b],(int)k_range[p7b],(int)k_range[h1b],(int)k_range[h2b]};
          
          auto plan = hptt::create_plan
          (perm, 4, 1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
              NULL, hptt::ESTIMATE, 1, NULL, true);
          plan->execute();
        }
        value = k_a_sort;
      }

      // auto ref_p456_h123 =
      //     std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
      // auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
      // auto cur_p456_h312 = std::make_tuple(p4b, p5b, p6b, h3b, h1b, h2b);
      // auto cur_p456_h132 = std::make_tuple(p4b, p5b, p6b, h1b, h3b, h2b);
      // auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
      // auto cur_p546_h312 = std::make_tuple(p5b, p4b, p6b, h3b, h1b, h2b);
      // auto cur_p546_h132 = std::make_tuple(p5b, p4b, p6b, h1b, h3b, h2b);
      // auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
      // auto cur_p564_h312 = std::make_tuple(p5b, p6b, p4b, h3b, h1b, h2b);
      // auto cur_p564_h132 = std::make_tuple(p5b, p6b, p4b, h1b, h3b, h2b);

      {
        // to get a unique t2 according to ia6 and nvab
        // printf ("[%s] copy d2_t2 based on idx_offset: %d\n", __func__, idx_offset);
        std::copy(k_a_sort.begin(), k_a_sort.end(), df_T_d2_t2 + (idx_offset * max_dima2));
      }

      // if (ref_p456_h123 == cur_p456_h123) {
      //   std::copy(k_a_sort.begin(), k_a_sort.end(),
      //             T_d2_t2 + d2b * max_dima2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p456_h312) {
      //   std::copy(k_a_sort.begin(), k_a_sort.end(),
      //             T_d2_t2 + d2b * max_dima2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p456_h132) {
      //   std::copy(k_a_sort.begin(), k_a_sort.end(),
      //             T_d2_t2 + d2b * max_dima2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p546_h123) {
      //   std::copy(k_a_sort.begin(), k_a_sort.end(),
      //             T_d2_t2 + d2b * max_dima2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p546_h312) {
      //   std::copy(k_a_sort.begin(), k_a_sort.end(),
      //             T_d2_t2 + d2b * max_dima2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p546_h132) {
      //   std::copy(k_a_sort.begin(), k_a_sort.end(),
      //             T_d2_t2 + d2b * max_dima2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p564_h123) {
      //   std::copy(k_a_sort.begin(), k_a_sort.end(),
      //             T_d2_t2 + d2b * max_dima2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p564_h312) {
      //   std::copy(k_a_sort.begin(), k_a_sort.end(),
      //             T_d2_t2 + d2b * max_dima2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p564_h132) {
      //   std::copy(k_a_sort.begin(), k_a_sort.end(),
      //             T_d2_t2 + d2b * max_dima2);
      //   d2b++;
      // }

      // 
      idx_offset++;
    } //p7b
  } // end ia6
  // printf ("[%s] ---------------------------------------------------------------\n", __func__);
  //ia6 -- get for v2
  // d2b = 0;
  idx_offset = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];

    for (Index p7b=noab;p7b<noab+nvab;p7b++) 
    {
      if(!ia6_enabled[ia6*nvab + p7b-noab]) continue;

      size_t dim_common = k_range[p7b];
      size_t dimb_sort = k_range[p5b]*k_range[p6b]*k_range[h3b];
      size_t dimb = dim_common*dimb_sort;

      std::vector<T> k_b_sort(dimb);

      IndexVector b_bids_minus_sidx = {p5b, p6b, h3b, p7b};
      auto [hit, value] = cache_d2v.log_access(b_bids_minus_sidx);
      
      if (hit) {
        k_b_sort = value;
      } else {
        // auto bbuf_start = d2b * max_dimb2;
        {
          TimerGuard tg_total{&ccsdt_d2_v2_GetTime};  
          ccsd_t_data_per_rank += dimb;                 
          d_v2.get({p5b,p6b,h3b,p7b},k_b_sort); //p7b,h3b,p6b,p5b
        }
        value = k_b_sort;
      }

      // auto ref_p456_h123 =
      //     std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
      // auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
      // auto cur_p456_h312 = std::make_tuple(p4b, p5b, p6b, h3b, h1b, h2b);
      // auto cur_p456_h132 = std::make_tuple(p4b, p5b, p6b, h1b, h3b, h2b);
      // auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
      // auto cur_p546_h312 = std::make_tuple(p5b, p4b, p6b, h3b, h1b, h2b);
      // auto cur_p546_h132 = std::make_tuple(p5b, p4b, p6b, h1b, h3b, h2b);
      // auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
      // auto cur_p564_h312 = std::make_tuple(p5b, p6b, p4b, h3b, h1b, h2b);
      // auto cur_p564_h132 = std::make_tuple(p5b, p6b, p4b, h1b, h3b, h2b);

      {
        // to get a unique v2 according to ia6 and nvab
        // printf ("[%s] copy d2_v2 based on idx_offset: %d\n", __func__, idx_offset);
        std::copy(k_b_sort.begin(), k_b_sort.end(), df_T_d2_v2 + idx_offset * max_dimb2);
      }

      // if (ref_p456_h123 == cur_p456_h123) {
      //   std::copy(k_b_sort.begin(), k_b_sort.end(),
      //             T_d2_v2 + d2b * max_dimb2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p456_h312) {
      //   std::copy(k_b_sort.begin(), k_b_sort.end(),
      //             T_d2_v2 + d2b * max_dimb2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p456_h132) {
      //   std::copy(k_b_sort.begin(), k_b_sort.end(),
      //             T_d2_v2 + d2b * max_dimb2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p546_h123) {
      //   std::copy(k_b_sort.begin(), k_b_sort.end(),
      //             T_d2_v2 + d2b * max_dimb2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p546_h312) {
      //   std::copy(k_b_sort.begin(), k_b_sort.end(),
      //             T_d2_v2 + d2b * max_dimb2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p546_h132) {
      //   std::copy(k_b_sort.begin(), k_b_sort.end(),
      //             T_d2_v2 + d2b * max_dimb2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p564_h123) {
      //   std::copy(k_b_sort.begin(), k_b_sort.end(),
      //             T_d2_v2 + d2b * max_dimb2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p564_h312) {
      //   std::copy(k_b_sort.begin(), k_b_sort.end(),
      //             T_d2_v2 + d2b * max_dimb2);
      //   d2b++;
      // }
      // if (ref_p456_h123 == cur_p564_h132) {
      //   std::copy(k_b_sort.begin(), k_b_sort.end(),
      //             T_d2_v2 + d2b * max_dimb2);
      //   d2b++;
      // }

      // 
      idx_offset++;
    } //p7b
  } // end ia6

  // 
  *df_num_d2_enabled = idx_offset;
} // ccsd_t_data_s1

template <typename T>
void ccsd_t_data_d2_info_only(
    bool is_restricted, const Index noab, const Index nvab, 
    std::vector<int>& k_spin, 
    std::vector<T>& k_evl_sorted, std::vector<size_t>& k_range, 
    size_t t_h1b, size_t t_h2b, size_t t_h3b, 
    size_t t_p4b, size_t t_p5b, size_t t_p6b,
    int* df_simple_d2_size, int* df_simple_d2_exec, 
    int* num_enabled_kernels, size_t& comm_data_elems)
{    
  // 
  std::tuple<Index, Index, Index, Index, Index, Index> a3_d2[] = {
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h2b, t_h3b, t_h1b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h3b, t_h2b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h2b, t_h3b, t_h1b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h1b, t_h3b, t_h2b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h2b, t_h3b, t_h1b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h1b, t_h3b, t_h2b)};


  for (auto ia6 = 0; ia6 < 8; ia6++) {
    if (std::get<0>(a3_d2[ia6]) != 0) {
      for (auto ja6 = ia6 + 1; ja6 < 9; ja6++) {  // TODO: ja6 start ?
        if (a3_d2[ia6] == a3_d2[ja6]) {
          a3_d2[ja6] = std::make_tuple(0,0,0,0,0,0);
        }
      }
    }
  }


  for (Index p7b=noab;p7b<noab+nvab;p7b++) {
    df_simple_d2_size[0 + (p7b - noab) * 7] = (int)k_range[t_h1b];
    df_simple_d2_size[1 + (p7b - noab) * 7] = (int)k_range[t_h2b];
    df_simple_d2_size[2 + (p7b - noab) * 7] = (int)k_range[t_h3b];
    df_simple_d2_size[3 + (p7b - noab) * 7] = (int)k_range[t_p4b];
    df_simple_d2_size[4 + (p7b - noab) * 7] = (int)k_range[t_p5b];
    df_simple_d2_size[5 + (p7b - noab) * 7] = (int)k_range[t_p6b];
    df_simple_d2_size[6 + (p7b - noab) * 7] = (int)k_range[p7b];
  }


  std::vector<bool> ia6_enabled(9*nvab,false);
  
  //ia6 -- compute which variants are enabled
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];

    if (!((p5b <= p6b) && (h1b <= h2b) && p4b != 0)) {
      continue;
    }
    if (is_restricted && !(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] + k_spin[h1b] + k_spin[h2b] +
               k_spin[h3b] !=
           12)) {
      continue;
    }
    if(!(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] ==
            k_spin[h1b] + k_spin[h2b] + k_spin[h3b])) {
      continue;
    }

    for (Index p7b=noab;p7b<noab+nvab;p7b++) {
      if (!(k_spin[p4b]+k_spin[p7b] == k_spin[h1b]+k_spin[h2b])) {
        continue;
      }
      if (!(k_range[p4b] > 0 && k_range[p5b] > 0 && k_range[p6b] > 0 &&
            k_range[h1b] > 0 && k_range[h2b] > 0 && k_range[h3b] > 0)) {
        continue;
      }
      if(!(h3b <= p7b)) continue;
      ia6_enabled[ia6*nvab + p7b-noab] = true;
    } //end h7b

  } // end ia6

  //ia6 -- compute sizes and permutations
  int idx_offset = 0;
  int detailed_stats[nvab][9];

  for (Index idx_nvab = 0; idx_nvab < nvab; idx_nvab++)
  {
    detailed_stats[idx_nvab][0] = 0;
    detailed_stats[idx_nvab][1] = 0;
    detailed_stats[idx_nvab][2] = 0;
    detailed_stats[idx_nvab][3] = 0;
    detailed_stats[idx_nvab][4] = 0;
    detailed_stats[idx_nvab][5] = 0;
    detailed_stats[idx_nvab][6] = 0;
    detailed_stats[idx_nvab][7] = 0;
    detailed_stats[idx_nvab][8] = 0;
  }


  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d2[ia6];
    auto ref_p456_h123 = std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h312 = std::make_tuple(p4b, p5b, p6b, h3b, h1b, h2b);
    auto cur_p456_h132 = std::make_tuple(p4b, p5b, p6b, h1b, h3b, h2b);
    auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    auto cur_p546_h312 = std::make_tuple(p5b, p4b, p6b, h3b, h1b, h2b);
    auto cur_p546_h132 = std::make_tuple(p5b, p4b, p6b, h1b, h3b, h2b);
    auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    auto cur_p564_h312 = std::make_tuple(p5b, p6b, p4b, h3b, h1b, h2b);
    auto cur_p564_h132 = std::make_tuple(p5b, p6b, p4b, h1b, h3b, h2b);

    
    for (Index p7b=noab;p7b<noab+nvab;p7b++) 
    {
      detailed_stats[p7b - noab][0] = 0;
      detailed_stats[p7b - noab][1] = 0;
      detailed_stats[p7b - noab][2] = 0;
      detailed_stats[p7b - noab][3] = 0;
      detailed_stats[p7b - noab][4] = 0;
      detailed_stats[p7b - noab][5] = 0;
      detailed_stats[p7b - noab][6] = 0;
      detailed_stats[p7b - noab][7] = 0;
      detailed_stats[p7b - noab][8] = 0;

      int idx_new_offset = 0;
      if(!ia6_enabled[ia6*nvab + p7b-noab]) continue;
    
      size_t dim_common = k_range[p7b];
      size_t dima_sort = k_range[p4b]*k_range[h1b]*k_range[h2b];
      size_t dimb_sort = k_range[p5b]*k_range[p6b]*k_range[h3b];
      comm_data_elems += dim_common * (dima_sort + dimb_sort);

      if (ref_p456_h123 == cur_p456_h123) {
        df_simple_d2_exec[0 + (p7b - noab) * 9] = idx_offset;
        *num_enabled_kernels = *num_enabled_kernels + 1;
        idx_new_offset++;
        detailed_stats[p7b - noab][0] = detailed_stats[p7b - noab][0] + 1;
      }
      if (ref_p456_h123 == cur_p456_h312) {
        df_simple_d2_exec[1 + (p7b - noab) * 9] = idx_offset;
        *num_enabled_kernels = *num_enabled_kernels + 1;
        idx_new_offset++;
        detailed_stats[p7b - noab][1] = detailed_stats[p7b - noab][1] + 1;
      }
      if (ref_p456_h123 == cur_p456_h132) {
        df_simple_d2_exec[2 + (p7b - noab) * 9] = idx_offset;
        *num_enabled_kernels = *num_enabled_kernels + 1;
        idx_new_offset++;
        detailed_stats[p7b - noab][2] = detailed_stats[p7b - noab][2] + 1;
      }
      if (ref_p456_h123 == cur_p546_h123) {
        df_simple_d2_exec[3 + (p7b - noab) * 9] = idx_offset;
        *num_enabled_kernels = *num_enabled_kernels + 1;
        idx_new_offset++;
        detailed_stats[p7b - noab][3] = detailed_stats[p7b - noab][3] + 1;
      }
      if (ref_p456_h123 == cur_p546_h312) {
        df_simple_d2_exec[4 + (p7b - noab) * 9] = idx_offset;
        *num_enabled_kernels = *num_enabled_kernels + 1;
        idx_new_offset++;
        detailed_stats[p7b - noab][4] = detailed_stats[p7b - noab][4] + 1;
      }
      if (ref_p456_h123 == cur_p546_h132) {
        df_simple_d2_exec[5 + (p7b - noab) * 9] = idx_offset;
        *num_enabled_kernels = *num_enabled_kernels + 1;
        idx_new_offset++;
        detailed_stats[p7b - noab][5] = detailed_stats[p7b - noab][5] + 1;
      }
      if (ref_p456_h123 == cur_p564_h123) {
        df_simple_d2_exec[6 + (p7b - noab) * 9] = idx_offset;
        *num_enabled_kernels = *num_enabled_kernels + 1;
        idx_new_offset++;
        detailed_stats[p7b - noab][6] = detailed_stats[p7b - noab][6] + 1;
      }
      if (ref_p456_h123 == cur_p564_h312) {
        df_simple_d2_exec[7 + (p7b - noab) * 9] = idx_offset;
        *num_enabled_kernels = *num_enabled_kernels + 1;
        idx_new_offset++;
        detailed_stats[p7b - noab][7] = detailed_stats[p7b - noab][7] + 1;
      }
      if (ref_p456_h123 == cur_p564_h132) {
        df_simple_d2_exec[8 + (p7b - noab) * 9] = idx_offset;
        *num_enabled_kernels = *num_enabled_kernels + 1;
        idx_new_offset++;
        detailed_stats[p7b - noab][8] = detailed_stats[p7b - noab][8] + 1;
      }

      // 
      idx_offset++;
      // printf ("[%s] d2, nvab=%2d, #: %d\n", __func__, p7b - noab, idx_new_offset);
      // printf ("[%s] d2, nvab=%2d, #: %d >> %d,%d,%d,%d,%d,%d,%d,%d,%d\n", __func__, p7b - noab, idx_new_offset,
      // detailed_stats[p7b - noab][0], detailed_stats[p7b - noab][1], detailed_stats[p7b - noab][2], 
      // detailed_stats[p7b - noab][3], detailed_stats[p7b - noab][4], detailed_stats[p7b - noab][5], 
      // detailed_stats[p7b - noab][6], detailed_stats[p7b - noab][7], detailed_stats[p7b - noab][8]);
    } //p7b
  } // end ia6
} // ccsd_t_data_s1


#endif //CCSD_T_ALL_FUSED_DOUBLES2_HPP_