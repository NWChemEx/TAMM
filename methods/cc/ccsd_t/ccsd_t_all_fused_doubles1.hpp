
#ifndef CCSD_T_ALL_FUSED_DOUBLES1_HPP_
#define CCSD_T_ALL_FUSED_DOUBLES1_HPP_

#include "tamm/tamm.hpp"
// using namespace tamm;

extern double ccsd_t_t2_GetTime;
extern double ccsd_t_v2_GetTime;
extern double ccsd_t_data_per_rank;

// singles data driver
template <typename T>
void ccsd_t_data_d1(
    ExecutionContext& ec, const TiledIndexSpace& MO, const Index noab,
    const Index nvab, std::vector<int>& k_spin, std::vector<size_t>& k_offset,
    Tensor<T>& d_t1,
    Tensor<T>& d_t2,  // d_a
    Tensor<T>& d_v2,  // d_b
    std::vector<T>& k_evl_sorted, std::vector<size_t>& k_range, size_t t_h1b,
    size_t t_h2b, size_t t_h3b, size_t t_p4b, size_t t_p5b, size_t t_p6b,
    std::vector<T>& k_abuf1, std::vector<T>& k_bbuf1,
    std::vector<size_t>& sd_t_d1_exec, std::vector<size_t>& d1_sizes_ext,
    bool is_restricted, LRUCache<Index,std::vector<T>>& cache_d1t, LRUCache<Index,std::vector<T>>& cache_d1v) {

  size_t abuf_size1 = k_abuf1.size();
  size_t bbuf_size1 = k_bbuf1.size();

  std::tuple<Index, Index, Index, Index, Index, Index> a3_d1[] = {
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p5b, t_p6b, t_p4b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p5b, t_p6b, t_p4b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p5b, t_p6b, t_p4b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p4b, t_p6b, t_p5b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p6b, t_p5b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p4b, t_p6b, t_p5b, t_h3b, t_h1b, t_h2b)};


  for (auto ia6 = 0; ia6 < 8; ia6++) {
    if (std::get<0>(a3_d1[ia6]) != 0) {
      for (auto ja6 = ia6 + 1; ja6 < 9; ja6++) {  // TODO: ja6 start ?
        if (a3_d1[ia6] == a3_d1[ja6]) {
          a3_d1[ja6] = std::make_tuple(0,0,0,0,0,0);
        }
      }
    }
  }

  const size_t max_dima = abuf_size1 /  (9*noab);
  const size_t max_dimb = bbuf_size1 /  (9*noab);

  //doubles 1
  // std::vector<size_t> sd_t_d1_exec(9*9*noab,-1);
  // std::vector<size_t> d1_sizes_ext(9*7*noab);

  size_t d1b = 0;

  for (auto ia6 = 0; ia6 < 9; ia6++) {
    for (Index h7b=0; h7b < noab; h7b++) {
      d1_sizes_ext[0 + (h7b + (ia6) * noab) * 7] = k_range[t_h1b];
      d1_sizes_ext[1 + (h7b + (ia6) * noab) * 7] = k_range[t_h2b];
      d1_sizes_ext[2 + (h7b + (ia6) * noab) * 7] = k_range[t_h3b];
      d1_sizes_ext[3 + (h7b + (ia6) * noab) * 7] = k_range[h7b];
      d1_sizes_ext[4 + (h7b + (ia6) * noab) * 7] = k_range[t_p4b];
      d1_sizes_ext[5 + (h7b + (ia6) * noab) * 7] = k_range[t_p5b];
      d1_sizes_ext[6 + (h7b + (ia6) * noab) * 7] = k_range[t_p6b];
    }
  }

  std::vector<bool> ia6_enabled(9*noab,false);
  
  //ia6 -- compute which variants are enabled
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];

    if (!((p4b <= p5b) && (h2b <= h3b) && p4b != 0)) {
      continue;
    }
    if (!(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] + k_spin[h1b] + k_spin[h2b] +
               k_spin[h3b] !=
           12)) {
      continue;
    }
    if(!(k_spin[p4b] + k_spin[p5b] + k_spin[p6b] ==
            k_spin[h1b] + k_spin[h2b] + k_spin[h3b])) {
      continue;
    }

    for (Index h7b=0; h7b < noab; h7b++) {
      if (!(k_spin[p4b]+k_spin[p5b] == k_spin[h1b]+k_spin[h7b])) {
        continue;
      }
      if (!(k_range[p4b] > 0 && k_range[p5b] > 0 && k_range[p6b] > 0 &&
            k_range[h1b] > 0 && k_range[h2b] > 0 && k_range[h3b] > 0)) {
        continue;
      }
      if(!(h7b <= p6b)) continue;
      ia6_enabled[ia6*noab + h7b] = true;
    } //end h7b

  } // end ia6

  //ia6 -- compute sizes and permutations
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];
    auto ref_p456_h123 = std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
    auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
    auto cur_p645_h123 = std::make_tuple(p6b, p4b, p5b, h1b, h2b, h3b);
    auto cur_p645_h213 = std::make_tuple(p6b, p4b, p5b, h2b, h1b, h3b);
    auto cur_p645_h231 = std::make_tuple(p6b, p4b, p5b, h2b, h3b, h1b);
    auto cur_p465_h123 = std::make_tuple(p4b, p6b, p5b, h1b, h2b, h3b);
    auto cur_p465_h213 = std::make_tuple(p4b, p6b, p5b, h2b, h1b, h3b);
    auto cur_p465_h231 = std::make_tuple(p4b, p6b, p5b, h2b, h3b, h1b);

    for (Index h7b=0;h7b<noab;h7b++)
    {
      if(!ia6_enabled[ia6*noab + h7b]) continue;
    
      if (ref_p456_h123 == cur_p456_h123) {
        sd_t_d1_exec[0 + (h7b + (ia6) * noab) * 9] = d1b++;
      }
      if (ref_p456_h123 == cur_p456_h213) {
        sd_t_d1_exec[1 + (h7b + (ia6) * noab) * 9] = d1b++;
      }
      if (ref_p456_h123 == cur_p456_h231) {
        sd_t_d1_exec[2 + (h7b + (ia6) * noab) * 9] = d1b++;
      }
      if (ref_p456_h123 == cur_p645_h123) {
        sd_t_d1_exec[3 + (h7b + (ia6) * noab) * 9] = d1b++;
      }
      if (ref_p456_h123 == cur_p645_h213) {
        sd_t_d1_exec[4 + (h7b + (ia6) * noab) * 9] = d1b++;
      }
      if (ref_p456_h123 == cur_p645_h231) {
        sd_t_d1_exec[5 + (h7b + (ia6) * noab) * 9] = d1b++;
      }
      if (ref_p456_h123 == cur_p465_h123) {
        sd_t_d1_exec[6 + (h7b + (ia6) * noab) * 9] = d1b++;
      }
      if (ref_p456_h123 == cur_p465_h213) {
        sd_t_d1_exec[7 + (h7b + (ia6) * noab) * 9] = d1b++;
      }
      if (ref_p456_h123 == cur_p465_h231) {
        sd_t_d1_exec[8 + (h7b + (ia6) * noab) * 9] = d1b++;
      }
    } //h7b
  }  // end ia6

  //ia6 -- get for t2
  d1b = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) {

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];

    IndexVector a_bids_minus_sidx = {
      p4b - noab,
      p5b - noab,
      h1b
    };
    auto [hit, value] = cache_d1t.log_access(a_bids_minus_sidx);
    if (hit) {
      std::copy(value.begin(), value.end(), k_abuf1.begin() + d1b * max_dima);
      d1b += value.size() / max_dima;
    } else {
      auto abuf_start = d1b * max_dima;
      for (Index h7b = 0; h7b < noab; h7b++) {
        if (!ia6_enabled[ia6 * noab + h7b]) continue;

        size_t dim_common = k_range[h7b];
        size_t dima_sort = k_range[p4b] * k_range[p5b] * k_range[h1b];
        size_t dima = dim_common * dima_sort;

        std::vector<T> k_a(dima);
        std::vector<T> k_a_sort(dima);

        // TODO
        if (h7b < h1b) {
          {
            TimerGuard tg_total{&ccsd_t_t2_GetTime};
            ccsd_t_data_per_rank += dima;
            d_t2.get({p4b - noab, p5b - noab, h7b, h1b},
                     k_a);  // h1b,h7b,p5b-noab,p4b-noab
          }
          int perm[4] = {3, 1, 0, 2};  // 3,1,0,2
          int size[4] = {(int)k_range[p4b], (int)k_range[p5b],
                         (int)k_range[h7b], (int)k_range[h1b]};

          auto plan = hptt::create_plan(perm, 4, -1.0, &k_a[0], size, NULL, 0,
                                        &k_a_sort[0], NULL, hptt::ESTIMATE, 1,
                                        NULL, true);
          plan->execute();
        }
        if (h1b <= h7b) {
          {
            TimerGuard tg_total{&ccsd_t_t2_GetTime};
            ccsd_t_data_per_rank += dima;
            d_t2.get({p4b - noab, p5b - noab, h1b, h7b},
                     k_a);  // h7b,h1b,p5b-noab,p4b-noab
          }
          int perm[4] = {2, 1, 0, 3};  // 2,1,0,3
          int size[4] = {(int)k_range[p4b], (int)k_range[p5b],
                         (int)k_range[h1b], (int)k_range[h7b]};

          auto plan = hptt::create_plan(perm, 4, 1.0, &k_a[0], size, NULL, 0,
                                        &k_a_sort[0], NULL, hptt::ESTIMATE, 1,
                                        NULL, true);
          plan->execute();
        }

        auto ref_p456_h123 =
            std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
        auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
        auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
        auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
        auto cur_p645_h123 = std::make_tuple(p6b, p4b, p5b, h1b, h2b, h3b);
        auto cur_p645_h213 = std::make_tuple(p6b, p4b, p5b, h2b, h1b, h3b);
        auto cur_p645_h231 = std::make_tuple(p6b, p4b, p5b, h2b, h3b, h1b);
        auto cur_p465_h123 = std::make_tuple(p4b, p6b, p5b, h1b, h2b, h3b);
        auto cur_p465_h213 = std::make_tuple(p4b, p6b, p5b, h2b, h1b, h3b);
        auto cur_p465_h231 = std::make_tuple(p4b, p6b, p5b, h2b, h3b, h1b);

        if (ref_p456_h123 == cur_p456_h123) {
          std::copy(k_a_sort.begin(), k_a_sort.end(),
                    k_abuf1.begin() + d1b * max_dima);
          d1b++;
        }
        if (ref_p456_h123 == cur_p456_h213) {
          std::copy(k_a_sort.begin(), k_a_sort.end(),
                    k_abuf1.begin() + d1b * max_dima);
          d1b++;
        }
        if (ref_p456_h123 == cur_p456_h231) {
          std::copy(k_a_sort.begin(), k_a_sort.end(),
                    k_abuf1.begin() + d1b * max_dima);
          d1b++;
        }
        if (ref_p456_h123 == cur_p645_h123) {
          std::copy(k_a_sort.begin(), k_a_sort.end(),
                    k_abuf1.begin() + d1b * max_dima);
          d1b++;
        }
        if (ref_p456_h123 == cur_p645_h213) {
          std::copy(k_a_sort.begin(), k_a_sort.end(),
                    k_abuf1.begin() + d1b * max_dima);
          d1b++;
        }
        if (ref_p456_h123 == cur_p645_h231) {
          std::copy(k_a_sort.begin(), k_a_sort.end(),
                    k_abuf1.begin() + d1b * max_dima);
          d1b++;
        }
        if (ref_p456_h123 == cur_p465_h123) {
          std::copy(k_a_sort.begin(), k_a_sort.end(),
                    k_abuf1.begin() + d1b * max_dima);
          d1b++;
        }
        if (ref_p456_h123 == cur_p465_h213) {
          std::copy(k_a_sort.begin(), k_a_sort.end(),
                    k_abuf1.begin() + d1b * max_dima);
          d1b++;
        }
        if (ref_p456_h123 == cur_p465_h231) {
          std::copy(k_a_sort.begin(), k_a_sort.end(),
                    k_abuf1.begin() + d1b * max_dima);
          d1b++;
        }
      }  // h7b
      auto abuf_end = d1b * max_dima;
      value.clear();
      value.insert(value.end(), k_abuf1.begin() + abuf_start,
                   k_abuf1.begin() + abuf_end);
    }  // else cache
  }  // end ia6

  //ia6 -- get for v2
  d1b = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_d1[ia6];
    IndexVector b_bids_minus_sidx = {p6b, h2b, h3b};
    auto [hit, value] = cache_d1v.log_access(b_bids_minus_sidx);
    if (hit) {
      std::copy(value.begin(), value.end(), k_bbuf1.begin() + d1b * max_dimb);
      d1b += value.size() / max_dimb;
    } else {
      auto bbuf_start = d1b * max_dimb;

      for (Index h7b = 0; h7b < noab; h7b++) {
        if (!ia6_enabled[ia6 * noab + h7b]) continue;

        size_t dim_common = k_range[h7b];
        size_t dimb_sort = k_range[p6b] * k_range[h2b] * k_range[h3b];
        size_t dimb = dim_common * dimb_sort;

        std::vector<T> k_b_sort(dimb);
        {
          TimerGuard tg_total{&ccsd_t_v2_GetTime};
          ccsd_t_data_per_rank += dimb;
          d_v2.get({h7b, p6b, h2b, h3b}, k_b_sort);  // h3b,h2b,p6b,h7b
        }

        auto ref_p456_h123 =
            std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
        auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
        auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
        auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
        auto cur_p645_h123 = std::make_tuple(p6b, p4b, p5b, h1b, h2b, h3b);
        auto cur_p645_h213 = std::make_tuple(p6b, p4b, p5b, h2b, h1b, h3b);
        auto cur_p645_h231 = std::make_tuple(p6b, p4b, p5b, h2b, h3b, h1b);
        auto cur_p465_h123 = std::make_tuple(p4b, p6b, p5b, h1b, h2b, h3b);
        auto cur_p465_h213 = std::make_tuple(p4b, p6b, p5b, h2b, h1b, h3b);
        auto cur_p465_h231 = std::make_tuple(p4b, p6b, p5b, h2b, h3b, h1b);

        if (ref_p456_h123 == cur_p456_h123) {
          std::copy(k_b_sort.begin(), k_b_sort.end(),
                    k_bbuf1.begin() + d1b * max_dimb);
          d1b++;
        }
        if (ref_p456_h123 == cur_p456_h213) {
          std::copy(k_b_sort.begin(), k_b_sort.end(),
                    k_bbuf1.begin() + d1b * max_dimb);
          d1b++;
        }
        if (ref_p456_h123 == cur_p456_h231) {
          std::copy(k_b_sort.begin(), k_b_sort.end(),
                    k_bbuf1.begin() + d1b * max_dimb);
          d1b++;
        }
        if (ref_p456_h123 == cur_p645_h123) {
          std::copy(k_b_sort.begin(), k_b_sort.end(),
                    k_bbuf1.begin() + d1b * max_dimb);
          d1b++;
        }
        if (ref_p456_h123 == cur_p645_h213) {
          std::copy(k_b_sort.begin(), k_b_sort.end(),
                    k_bbuf1.begin() + d1b * max_dimb);
          d1b++;
        }
        if (ref_p456_h123 == cur_p645_h231) {
          std::copy(k_b_sort.begin(), k_b_sort.end(),
                    k_bbuf1.begin() + d1b * max_dimb);
          d1b++;
        }
        if (ref_p456_h123 == cur_p465_h123) {
          std::copy(k_b_sort.begin(), k_b_sort.end(),
                    k_bbuf1.begin() + d1b * max_dimb);
          d1b++;
        }
        if (ref_p456_h123 == cur_p465_h213) {
          std::copy(k_b_sort.begin(), k_b_sort.end(),
                    k_bbuf1.begin() + d1b * max_dimb);
          d1b++;
        }
        if (ref_p456_h123 == cur_p465_h231) {
          std::copy(k_b_sort.begin(), k_b_sort.end(),
                    k_bbuf1.begin() + d1b * max_dimb);
          d1b++;
        }
      }  // h7b
      auto bbuf_end = d1b * max_dimb;
      value.clear();
      value.insert(value.end(), k_bbuf1.begin() + bbuf_start,
                   k_bbuf1.begin() + bbuf_end);
    }  // else cache
  }    // end ia6
       // return s1b;
}  // ccsd_t_data_s1

#endif //CCSD_T_ALL_FUSED_DOUBLES1_HPP_