#ifndef CCSD_T_ALL_FUSED_SINGLES_HPP_
#define CCSD_T_ALL_FUSED_SINGLES_HPP_

#include "tamm/tamm.hpp"
// using namespace tamm;

#define TEST_NEW_KERNEL
#define TEST_NEW_THREAD

extern double ccsdt_s1_t1_GetTime;
extern double ccsdt_s1_v2_GetTime;
extern double ccsd_t_data_per_rank;

// singles data driver
template <typename T>
void ccsd_t_data_s1(
    ExecutionContext& ec, const TiledIndexSpace& MO, const Index noab,
    const Index nvab, std::vector<int>& k_spin, std::vector<size_t>& k_offset,
    Tensor<T>& d_t1,
    Tensor<T>& d_t2,  // d_a
    Tensor<T>& d_v2,  // d_b
    std::vector<T>& k_evl_sorted, std::vector<size_t>& k_range, size_t t_h1b,
    size_t t_h2b, size_t t_h3b, size_t t_p4b, size_t t_p5b, size_t t_p6b,
    std::vector<T>& k_abufs1, std::vector<T>& k_bbufs1, 
    // 
    std::vector<int>& s1_flags, std::vector<int>& s1_sizes,
    T* T_s1_t1, T* T_s1_v2, 
    // 
    std::vector<int>& sd_t_s1_exec, std::vector<int>& s1_sizes_ext,
    // std::vector<size_t>& sd_t_s1_exec, std::vector<size_t>& s1_sizes_ext,
    bool is_restricted, LRUCache<Index,std::vector<T>>& cache_s1t, LRUCache<Index,std::vector<T>>& cache_s1v) {

  size_t abufs1_size = k_abufs1.size();
  size_t bbufs1_size = k_bbufs1.size();

  std::tuple<Index, Index, Index, Index, Index, Index> a3_s1[] = {
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h3b, t_h1b, t_h2b)};

  for (auto ia6 = 0; ia6 < 8; ia6++) {
    if (std::get<0>(a3_s1[ia6]) != 0) {
      for (auto ja6 = ia6 + 1; ja6 < 9; ja6++) {  // TODO: ja6 start ?
        if (a3_s1[ia6] == a3_s1[ja6]) {
          a3_s1[ja6] = std::make_tuple(0,0,0,0,0,0);
        }
      }
    }
  }

  const size_t s1_max_dima = abufs1_size / 9;
  const size_t s1_max_dimb = bbufs1_size / 9;

  // singles
  // std::vector<size_t> sd_t_s1_exec(9 * 9, -1);
  // std::vector<size_t> s1_sizes_ext(9 * 6);

  // size_t s1b = 0;
  int s1b = 0;

  for (auto ia6 = 0; ia6 < 9; ia6++) {
    s1_sizes_ext[0 + ia6 * 6] = (int)k_range[t_h1b];
    s1_sizes_ext[1 + ia6 * 6] = (int)k_range[t_h2b];
    s1_sizes_ext[2 + ia6 * 6] = (int)k_range[t_h3b];
    s1_sizes_ext[3 + ia6 * 6] = (int)k_range[t_p4b];
    s1_sizes_ext[4 + ia6 * 6] = (int)k_range[t_p5b];
    s1_sizes_ext[5 + ia6 * 6] = (int)k_range[t_p6b];
  }

#ifdef TEST_NEW_KERNEL
  s1_sizes[0] = k_range[t_h1b];
  s1_sizes[1] = k_range[t_h2b];
  s1_sizes[2] = k_range[t_h3b];
  s1_sizes[3] = k_range[t_p4b];
  s1_sizes[4] = k_range[t_p5b];
  s1_sizes[5] = k_range[t_p6b];
#endif

  std::vector<bool> ia6_enabled(9,false);
  
  //ia6 -- compute which variants are enabled
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];

    if (!((p5b <= p6b) && (h2b <= h3b) && p4b != 0)) {
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
    if (!(k_spin[p4b] == k_spin[h1b])) {
      continue;
    }
    if (!(k_range[p4b] > 0 && k_range[p5b] > 0 && k_range[p6b] > 0 &&
          k_range[h1b] > 0 && k_range[h2b] > 0 && k_range[h3b] > 0)) {
      continue;
    }

    ia6_enabled[ia6] = true;
  } // end ia6

  //ia6 -- compute sizes and permutations
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    if (!ia6_enabled[ia6]) {
      continue;
    }
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];
    auto ref_p456_h123 = std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
    auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
    auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    auto cur_p546_h213 = std::make_tuple(p5b, p4b, p6b, h2b, h1b, h3b);
    auto cur_p546_h231 = std::make_tuple(p5b, p4b, p6b, h2b, h3b, h1b);
    auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    auto cur_p564_h213 = std::make_tuple(p5b, p6b, p4b, h2b, h1b, h3b);
    auto cur_p564_h231 = std::make_tuple(p5b, p6b, p4b, h2b, h3b, h1b);

    if (ref_p456_h123 == cur_p456_h123) {
      sd_t_s1_exec[ia6 * 9 + 0] = s1b++;
    }
    if (ref_p456_h123 == cur_p456_h213) {
      sd_t_s1_exec[ia6 * 9 + 1] = s1b++;
    }
    if (ref_p456_h123 == cur_p456_h231) {
      sd_t_s1_exec[ia6 * 9 + 2] = s1b++;
    }
    if (ref_p456_h123 == cur_p546_h123) {
      sd_t_s1_exec[ia6 * 9 + 3] = s1b++;
    }
    if (ref_p456_h123 == cur_p546_h213) {
      sd_t_s1_exec[ia6 * 9 + 4] = s1b++;
    }
    if (ref_p456_h123 == cur_p546_h231) {
      sd_t_s1_exec[ia6 * 9 + 5] = s1b++;
    }
    if (ref_p456_h123 == cur_p564_h123) {
      sd_t_s1_exec[ia6 * 9 + 6] = s1b++;
    }
    if (ref_p456_h123 == cur_p564_h213) {
      sd_t_s1_exec[ia6 * 9 + 7] = s1b++;
    }
    if (ref_p456_h123 == cur_p564_h231) {
      sd_t_s1_exec[ia6 * 9 + 8] = s1b++;
    }
  }  // end ia6

  //ia6 -- get for t1
  s1b = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    if(!ia6_enabled[ia6]) {
      continue;
    }
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];

    size_t dim_common = 1;
    size_t dima_sort  = k_range[p4b]*k_range[h1b];
    size_t dima       = dim_common * dima_sort;

    std::vector<T> k_a(dima);
    std::vector<T> k_a_sort(dima);

    auto [hit, value] = cache_s1t.log_access({p4b - noab, h1b});

    if (hit) {
    // if (false) {
      // std::copy(value.begin(), value.end(),
      //           k_abufs1.begin() + s1b * s1_max_dima);
      // s1b += value.size() / s1_max_dima;      
      k_a_sort = value;
    } 
    else {
      // auto abuf_start = s1b * s1_max_dima;
      {
        // IndexVector bids = {p4b - noab, h1b};
        TimerGuard tg_total{&ccsdt_s1_t1_GetTime};
        ccsd_t_data_per_rank += dima;
        // printf ("[%s] s1 <-- p4b - noab, h1b // %d - %d, %d\n", __func__, p4b, noab, h1b);
        d_t1.get({p4b - noab, h1b}, k_a);
      }
      const int ndim = 2;
      int perm[ndim] = {1, 0};
      int size[ndim] = {(int)k_range[p4b], (int)k_range[h1b]};
      // create a plan (shared_ptr)
      // To-Do (JK): Do we need to transpose this?
      auto plan =
          hptt::create_plan(perm, ndim, 1, &k_a[0], size, NULL, 0, &k_a_sort[0],
                            NULL, hptt::ESTIMATE, 1, NULL, true);
      plan->execute();
      value = k_a_sort;
    }

    // printf (">> inner: s_t1[10]: %.10f (s1b=%2d) by %d\n", k_a_sort[10], s1b, omp_get_thread_num());

    auto ref_p456_h123 =
        std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
    auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
    auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    auto cur_p546_h213 = std::make_tuple(p5b, p4b, p6b, h2b, h1b, h3b);
    auto cur_p546_h231 = std::make_tuple(p5b, p4b, p6b, h2b, h3b, h1b);
    auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    auto cur_p564_h213 = std::make_tuple(p5b, p6b, p4b, h2b, h1b, h3b);
    auto cur_p564_h231 = std::make_tuple(p5b, p6b, p4b, h2b, h3b, h1b);

    if (ref_p456_h123 == cur_p456_h123) {
      std::copy(k_a_sort.begin(), k_a_sort.end(),
                // k_abufs1.begin() + s1b * s1_max_dima);
                T_s1_t1 + s1b * s1_max_dima);
      s1b++;
    }
    if (ref_p456_h123 == cur_p456_h213) {
      std::copy(k_a_sort.begin(), k_a_sort.end(),
                // k_abufs1.begin() + s1b * s1_max_dima);
                T_s1_t1 + s1b * s1_max_dima);
      s1b++;
    }
    if (ref_p456_h123 == cur_p456_h231) {
      std::copy(k_a_sort.begin(), k_a_sort.end(),
                // k_abufs1.begin() + s1b * s1_max_dima);
                T_s1_t1 + s1b * s1_max_dima);
      s1b++;
    }
    if (ref_p456_h123 == cur_p546_h123) {
      std::copy(k_a_sort.begin(), k_a_sort.end(),
                // k_abufs1.begin() + s1b * s1_max_dima);
                T_s1_t1 + s1b * s1_max_dima);
      s1b++;
    }
    if (ref_p456_h123 == cur_p546_h213) {
      std::copy(k_a_sort.begin(), k_a_sort.end(),
                // k_abufs1.begin() + s1b * s1_max_dima);
                T_s1_t1 + s1b * s1_max_dima);
      s1b++;
    }
    if (ref_p456_h123 == cur_p546_h231) {
      std::copy(k_a_sort.begin(), k_a_sort.end(),
                // k_abufs1.begin() + s1b * s1_max_dima);
                T_s1_t1 + s1b * s1_max_dima);
      s1b++;
    }
    if (ref_p456_h123 == cur_p564_h123) {
      std::copy(k_a_sort.begin(), k_a_sort.end(),
                // k_abufs1.begin() + s1b * s1_max_dima);
                T_s1_t1 + s1b * s1_max_dima);
      s1b++;
    }
    if (ref_p456_h123 == cur_p564_h213) {
      std::copy(k_a_sort.begin(), k_a_sort.end(),
                // k_abufs1.begin() + s1b * s1_max_dima);
                T_s1_t1 + s1b * s1_max_dima);
      s1b++;
    }
    if (ref_p456_h123 == cur_p564_h231) {
      std::copy(k_a_sort.begin(), k_a_sort.end(),
                // k_abufs1.begin() + s1b * s1_max_dima);
                T_s1_t1 + s1b * s1_max_dima);
      s1b++;
    }
    //   value.clear();
    //   value.insert(value.begin(), k_abufs1.begin() + abuf_start,
    //                k_abufs1.begin() + abuf_end);
    // } //else
  }  // end ia6

  //ia6 -- get for v2
  s1b = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) 
  {
    if (!ia6_enabled[ia6]) {
      continue;
    }
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];

    size_t dim_common = 1;
    size_t dimb_sort =
        k_range[p5b] * k_range[p6b] * k_range[h2b] * k_range[h3b];
    size_t dimb = dim_common * dimb_sort;

    std::vector<T> k_b_sort(dimb);
    auto [hit, value] = cache_s1v.log_access({p5b, p6b, h2b, h3b});
    if (hit) {
    // if (false) {
      // std::copy(value.begin(), value.end(),
      //           k_bbufs1.begin() + s1b * s1_max_dimb);
      // s1b += value.size() / s1_max_dimb;
      k_b_sort = value;
    } else {
      // auto bbuf_start = s1b * s1_max_dimb;
      {
        TimerGuard tg_total{&ccsdt_s1_v2_GetTime};
        ccsd_t_data_per_rank += dimb;
        d_v2.get({p5b, p6b, h2b, h3b}, k_b_sort);  // h3b,h2b,p6b,p5b
      }
      value = k_b_sort;
    }

    auto ref_p456_h123 =
        std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
    auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
    auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    auto cur_p546_h213 = std::make_tuple(p5b, p4b, p6b, h2b, h1b, h3b);
    auto cur_p546_h231 = std::make_tuple(p5b, p4b, p6b, h2b, h3b, h1b);
    auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    auto cur_p564_h213 = std::make_tuple(p5b, p6b, p4b, h2b, h1b, h3b);
    auto cur_p564_h231 = std::make_tuple(p5b, p6b, p4b, h2b, h3b, h1b);

    if (ref_p456_h123 == cur_p456_h123) {
      std::copy(k_b_sort.begin(), k_b_sort.end(),
                // k_bbufs1.begin() + s1b * s1_max_dimb);
                T_s1_v2 + s1b * s1_max_dimb);
      s1b++;
    }
    if (ref_p456_h123 == cur_p456_h213) {
      std::copy(k_b_sort.begin(), k_b_sort.end(),
                // k_bbufs1.begin() + s1b * s1_max_dimb);
                T_s1_v2 + s1b * s1_max_dimb);
      s1b++;
    }
    if (ref_p456_h123 == cur_p456_h231) {
      std::copy(k_b_sort.begin(), k_b_sort.end(),
                // k_bbufs1.begin() + s1b * s1_max_dimb);
                T_s1_v2 + s1b * s1_max_dimb);
      s1b++;
    }
    if (ref_p456_h123 == cur_p546_h123) {
      std::copy(k_b_sort.begin(), k_b_sort.end(),
                // k_bbufs1.begin() + s1b * s1_max_dimb);
                T_s1_v2 + s1b * s1_max_dimb);
      s1b++;
    }
    if (ref_p456_h123 == cur_p546_h213) {
      std::copy(k_b_sort.begin(), k_b_sort.end(),
                // k_bbufs1.begin() + s1b * s1_max_dimb);
                T_s1_v2 + s1b * s1_max_dimb);
      s1b++;
    }
    if (ref_p456_h123 == cur_p546_h231) {
      std::copy(k_b_sort.begin(), k_b_sort.end(),
                // k_bbufs1.begin() + s1b * s1_max_dimb);
                T_s1_v2 + s1b * s1_max_dimb);
      s1b++;
    }
    if (ref_p456_h123 == cur_p564_h123) {
      std::copy(k_b_sort.begin(), k_b_sort.end(),
                // k_bbufs1.begin() + s1b * s1_max_dimb);
                T_s1_v2 + s1b * s1_max_dimb);
      s1b++;
    }
    if (ref_p456_h123 == cur_p564_h213) {
      std::copy(k_b_sort.begin(), k_b_sort.end(),
                // k_bbufs1.begin() + s1b * s1_max_dimb);
                T_s1_v2 + s1b * s1_max_dimb);
      s1b++;
    }
    if (ref_p456_h123 == cur_p564_h231) {
      std::copy(k_b_sort.begin(), k_b_sort.end(),
                // k_bbufs1.begin() + s1b * s1_max_dimb);
                T_s1_v2 + s1b * s1_max_dimb);
      s1b++;
    }
    //   auto bbuf_end = s1b * s1_max_dimb;
    //   value.clear();
    //   value.insert(value.end(), k_bbufs1.begin() + bbuf_start,
    //                k_bbufs1.begin() + bbuf_end);
    //  } //else
  }  // end ia6
  // return s1b;
}  // ccsd_t_data_s1


// singles data driver
template <typename T>
void ccsd_t_data_s1_new(bool is_restricted,
    // ExecutionContext& ec, 
    // const TiledIndexSpace& MO, 
    const Index noab, const Index nvab, 
    std::vector<int>& k_spin, 
    // std::vector<size_t>& k_offset,
    Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_v2, 
    std::vector<T>& k_evl_sorted, std::vector<size_t>& k_range, 
    size_t t_h1b, size_t t_h2b, size_t t_h3b, 
    size_t t_p4b, size_t t_p5b, size_t t_p6b,
    // 
    size_t  size_T_s1_t1,       size_t  size_T_s1_v2, 
    // 
    int*    df_simple_s1_size,  int*    df_simple_s1_exec, 
    T*      df_T_s1_t1,         T*      df_T_s1_v2, 
    int*    df_num_s1_enabled, 
    // 
    LRUCache<Index,std::vector<T>>& cache_s1t, LRUCache<Index,std::vector<T>>& cache_s1v) 
{
  // printf ("[%s] ------------------------------------------------\n", __func__);
  // 
  size_t abufs1_size = size_T_s1_t1; //k_abufs1.size();
  size_t bbufs1_size = size_T_s1_v2; //k_bbufs1.size();

  std::tuple<Index, Index, Index, Index, Index, Index> a3_s1[] = {
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h3b, t_h1b, t_h2b)};

  for (auto ia6 = 0; ia6 < 8; ia6++) {
    if (std::get<0>(a3_s1[ia6]) != 0) {
      for (auto ja6 = ia6 + 1; ja6 < 9; ja6++) {  // TODO: ja6 start ?
        if (a3_s1[ia6] == a3_s1[ja6]) {
          a3_s1[ja6] = std::make_tuple(0,0,0,0,0,0);
        }
      }
    }
  }

  const size_t s1_max_dima = abufs1_size / 9;
  const size_t s1_max_dimb = bbufs1_size / 9;

  // size_t s1b = 0;
  // int s1b = 0;
#if 0
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    df_s1_size[0 + ia6 * 6] = (int)k_range[t_h1b];
    df_s1_size[1 + ia6 * 6] = (int)k_range[t_h2b];
    df_s1_size[2 + ia6 * 6] = (int)k_range[t_h3b];
    df_s1_size[3 + ia6 * 6] = (int)k_range[t_p4b];
    df_s1_size[4 + ia6 * 6] = (int)k_range[t_p5b];
    df_s1_size[5 + ia6 * 6] = (int)k_range[t_p6b];
  }
#endif

  df_simple_s1_size[0] = (int)k_range[t_h1b];
  df_simple_s1_size[1] = (int)k_range[t_h2b];
  df_simple_s1_size[2] = (int)k_range[t_h3b];
  df_simple_s1_size[3] = (int)k_range[t_p4b];
  df_simple_s1_size[4] = (int)k_range[t_p5b];
  df_simple_s1_size[5] = (int)k_range[t_p6b];

  std::vector<bool> ia6_enabled(9,false);
  
  //ia6 -- compute which variants are enabled
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];

    if (!((p5b <= p6b) && (h2b <= h3b) && p4b != 0)) {
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
    if (!(k_spin[p4b] == k_spin[h1b])) {
      continue;
    }
    if (!(k_range[p4b] > 0 && k_range[p5b] > 0 && k_range[p6b] > 0 &&
          k_range[h1b] > 0 && k_range[h2b] > 0 && k_range[h3b] > 0)) {
      continue;
    }

    ia6_enabled[ia6] = true;
  } // end ia6

  //ia6 -- compute sizes and permutations
  int idx_offset = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    if (!ia6_enabled[ia6]) {
      continue;
    }
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];
    auto ref_p456_h123 = std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
    auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
    auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    auto cur_p546_h213 = std::make_tuple(p5b, p4b, p6b, h2b, h1b, h3b);
    auto cur_p546_h231 = std::make_tuple(p5b, p4b, p6b, h2b, h3b, h1b);
    auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    auto cur_p564_h213 = std::make_tuple(p5b, p6b, p4b, h2b, h1b, h3b);
    auto cur_p564_h231 = std::make_tuple(p5b, p6b, p4b, h2b, h3b, h1b);

    if (ref_p456_h123 == cur_p456_h123) {
      // df_s1_exec[ia6 * 9 + 0] = s1b++;
      df_simple_s1_exec[0] = idx_offset;
    }
    if (ref_p456_h123 == cur_p456_h213) {
      // df_s1_exec[ia6 * 9 + 1] = s1b++;
      df_simple_s1_exec[1] = idx_offset;
    }
    if (ref_p456_h123 == cur_p456_h231) {
      // df_s1_exec[ia6 * 9 + 2] = s1b++;
      df_simple_s1_exec[2] = idx_offset;
    }
    if (ref_p456_h123 == cur_p546_h123) {
      // df_s1_exec[ia6 * 9 + 3] = s1b++;
      df_simple_s1_exec[3] = idx_offset;
    }
    if (ref_p456_h123 == cur_p546_h213) {
      // df_s1_exec[ia6 * 9 + 4] = s1b++;
      df_simple_s1_exec[4] = idx_offset;
    }
    if (ref_p456_h123 == cur_p546_h231) {
      // df_s1_exec[ia6 * 9 + 5] = s1b++;
      df_simple_s1_exec[5] = idx_offset;
    }
    if (ref_p456_h123 == cur_p564_h123) {
      // df_s1_exec[ia6 * 9 + 6] = s1b++;
      df_simple_s1_exec[6] = idx_offset;
    }
    if (ref_p456_h123 == cur_p564_h213) {
      // df_s1_exec[ia6 * 9 + 7] = s1b++;
      df_simple_s1_exec[7] = idx_offset;
    }
    if (ref_p456_h123 == cur_p564_h231) {
      // df_s1_exec[ia6 * 9 + 8] = s1b++;
      df_simple_s1_exec[8] = idx_offset;
    }

    // 
    idx_offset++;
  } // end ia6

  //ia6 -- get for t1
  // s1b = 0;
  idx_offset = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) 
  {
    if(!ia6_enabled[ia6]) {
      continue;
    }
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];

    size_t dim_common = 1;
    size_t dima_sort  = k_range[p4b]*k_range[h1b];
    size_t dima       = dim_common * dima_sort;

    std::vector<T> k_a(dima);
    std::vector<T> k_a_sort(dima);

    auto [hit, value] = cache_s1t.log_access({p4b - noab, h1b});
    if (hit) {
      k_a_sort = value;
    } 
    else {
      {
        // IndexVector bids = {p4b - noab, h1b};
        TimerGuard tg_total{&ccsdt_s1_t1_GetTime};
        ccsd_t_data_per_rank += dima;
        d_t1.get({p4b - noab, h1b}, k_a);
      }
      const int ndim = 2;
      int perm[ndim] = {1, 0};
      int size[ndim] = {(int)k_range[p4b], (int)k_range[h1b]};
      // create a plan (shared_ptr)

      // To-Do (JK): Do we need to transpose this?
      auto plan =
          hptt::create_plan(perm, ndim, 1, &k_a[0], size, NULL, 0, &k_a_sort[0],
                            NULL, hptt::ESTIMATE, 1, NULL, true);
      plan->execute();
      value = k_a_sort;
    }

    // auto ref_p456_h123 =
    //     std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    // auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    // auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
    // auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
    // auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    // auto cur_p546_h213 = std::make_tuple(p5b, p4b, p6b, h2b, h1b, h3b);
    // auto cur_p546_h231 = std::make_tuple(p5b, p4b, p6b, h2b, h3b, h1b);
    // auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    // auto cur_p564_h213 = std::make_tuple(p5b, p6b, p4b, h2b, h1b, h3b);
    // auto cur_p564_h231 = std::make_tuple(p5b, p6b, p4b, h2b, h3b, h1b);

    // 
    //  already pruned by "ia6_enabled[ia6]"
    //
    {
      // printf ("[%s] to get s1's t1 // ia6=%lu // on %d (offset: %lu)\n", __func__, ia6, idx_offset, idx_offset * s1_max_dima);
      std::copy(k_a_sort.begin(), k_a_sort.end(), df_T_s1_t1 + idx_offset * s1_max_dima);
    }

    // if (ref_p456_h123 == cur_p456_h123) {
    //   std::copy(k_a_sort.begin(), k_a_sort.end(),
    //             T_s1_t1 + s1b * s1_max_dima);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p456_h213) {
    //   std::copy(k_a_sort.begin(), k_a_sort.end(),
    //             T_s1_t1 + s1b * s1_max_dima);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p456_h231) {
    //   std::copy(k_a_sort.begin(), k_a_sort.end(),
    //             T_s1_t1 + s1b * s1_max_dima);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p546_h123) {
    //   std::copy(k_a_sort.begin(), k_a_sort.end(),
    //             T_s1_t1 + s1b * s1_max_dima);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p546_h213) {
    //   std::copy(k_a_sort.begin(), k_a_sort.end(),
    //             T_s1_t1 + s1b * s1_max_dima);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p546_h231) {
    //   std::copy(k_a_sort.begin(), k_a_sort.end(),
    //             T_s1_t1 + s1b * s1_max_dima);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p564_h123) {
    //   std::copy(k_a_sort.begin(), k_a_sort.end(),
    //             T_s1_t1 + s1b * s1_max_dima);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p564_h213) {
    //   std::copy(k_a_sort.begin(), k_a_sort.end(),
    //             T_s1_t1 + s1b * s1_max_dima);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p564_h231) {
    //   std::copy(k_a_sort.begin(), k_a_sort.end(),
    //             T_s1_t1 + s1b * s1_max_dima);
    //   s1b++;
    // }

    // 
    idx_offset++;
  } // end ia6

  //ia6 -- get for v2
  // s1b = 0;
  idx_offset = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) 
  {
    if (!ia6_enabled[ia6]) {
      continue;
    }
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];

    size_t dim_common = 1;
    size_t dimb_sort =
        k_range[p5b] * k_range[p6b] * k_range[h2b] * k_range[h3b];
    size_t dimb = dim_common * dimb_sort;

    std::vector<T> k_b_sort(dimb);
    auto [hit, value] = cache_s1v.log_access({p5b, p6b, h2b, h3b});
    if (hit) {
      k_b_sort = value;
    } else {
      {
        TimerGuard tg_total{&ccsdt_s1_v2_GetTime};
        ccsd_t_data_per_rank += dimb;
        d_v2.get({p5b, p6b, h2b, h3b}, k_b_sort);  // h3b,h2b,p6b,p5b
      }
      value = k_b_sort;
    }

    // auto ref_p456_h123 =
    //     std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    // auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    // auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
    // auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
    // auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    // auto cur_p546_h213 = std::make_tuple(p5b, p4b, p6b, h2b, h1b, h3b);
    // auto cur_p546_h231 = std::make_tuple(p5b, p4b, p6b, h2b, h3b, h1b);
    // auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    // auto cur_p564_h213 = std::make_tuple(p5b, p6b, p4b, h2b, h1b, h3b);
    // auto cur_p564_h231 = std::make_tuple(p5b, p6b, p4b, h2b, h3b, h1b);

    // 
    //  already pruned by "ia6_enabled[ia6]"
    // 
    {
      // printf ("[%s] to get s1's v2 // ia6=%lu // on %d (offset: %lu)\n", __func__, ia6, idx_offset, idx_offset * s1_max_dimb);
      std::copy(k_b_sort.begin(), k_b_sort.end(), df_T_s1_v2 + idx_offset * s1_max_dimb);
    }

    // if (ref_p456_h123 == cur_p456_h123) {
    //   std::copy(k_b_sort.begin(), k_b_sort.end(),
    //             T_s1_v2 + s1b * s1_max_dimb);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p456_h213) {
    //   std::copy(k_b_sort.begin(), k_b_sort.end(),
    //             T_s1_v2 + s1b * s1_max_dimb);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p456_h231) {
    //   std::copy(k_b_sort.begin(), k_b_sort.end(),
    //             T_s1_v2 + s1b * s1_max_dimb);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p546_h123) {
    //   std::copy(k_b_sort.begin(), k_b_sort.end(),
    //             T_s1_v2 + s1b * s1_max_dimb);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p546_h213) {
    //   std::copy(k_b_sort.begin(), k_b_sort.end(),
    //             T_s1_v2 + s1b * s1_max_dimb);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p546_h231) {
    //   std::copy(k_b_sort.begin(), k_b_sort.end(),
    //             T_s1_v2 + s1b * s1_max_dimb);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p564_h123) {
    //   std::copy(k_b_sort.begin(), k_b_sort.end(),
    //             T_s1_v2 + s1b * s1_max_dimb);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p564_h213) {
    //   std::copy(k_b_sort.begin(), k_b_sort.end(),
    //             T_s1_v2 + s1b * s1_max_dimb);
    //   s1b++;
    // }
    // if (ref_p456_h123 == cur_p564_h231) {
    //   std::copy(k_b_sort.begin(), k_b_sort.end(),
    //             T_s1_v2 + s1b * s1_max_dimb);
    //   s1b++;
    // }

    // 
    idx_offset++;
  } // end ia6

  *df_num_s1_enabled = idx_offset;
  // printf ("[%s] ------------------------------------------------ df_num_s1_enabled: %d\n", __func__, *df_num_s1_enabled);
} // ccsd_t_data_s1

// singles data driver
template <typename T>
void ccsd_t_data_s1_info_only(bool is_restricted, const Index noab, const Index nvab, 
                              std::vector<int>& k_spin, 
                              std::vector<T>& k_evl_sorted, std::vector<size_t>& k_range, 
                              size_t t_h1b, size_t t_h2b, size_t t_h3b, 
                              size_t t_p4b, size_t t_p5b, size_t t_p6b,
                              // 
                              int* df_simple_s1_size, int* df_simple_s1_exec, 
                              int* num_enabled_kernels, size_t& comm_data_elems)
{
  std::tuple<Index, Index, Index, Index, Index, Index> a3_s1[] = {
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p4b, t_p5b, t_p6b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p5b, t_p4b, t_p6b, t_h3b, t_h1b, t_h2b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h1b, t_h2b, t_h3b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h2b, t_h1b, t_h3b),
      std::make_tuple(t_p6b, t_p4b, t_p5b, t_h3b, t_h1b, t_h2b)};

  for (auto ia6 = 0; ia6 < 8; ia6++) {
    if (std::get<0>(a3_s1[ia6]) != 0) {
      for (auto ja6 = ia6 + 1; ja6 < 9; ja6++) {  // TODO: ja6 start ?
        if (a3_s1[ia6] == a3_s1[ja6]) {
          a3_s1[ja6] = std::make_tuple(0,0,0,0,0,0);
        }
      }
    }
  }

  // 
  df_simple_s1_size[0] = (int)k_range[t_h1b];
  df_simple_s1_size[1] = (int)k_range[t_h2b];
  df_simple_s1_size[2] = (int)k_range[t_h3b];
  df_simple_s1_size[3] = (int)k_range[t_p4b];
  df_simple_s1_size[4] = (int)k_range[t_p5b];
  df_simple_s1_size[5] = (int)k_range[t_p6b];

  std::vector<bool> ia6_enabled(9,false);
  
  //ia6 -- compute which variants are enabled
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];

    if (!((p5b <= p6b) && (h2b <= h3b) && p4b != 0)) {
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
    if (!(k_spin[p4b] == k_spin[h1b])) {
      continue;
    }
    if (!(k_range[p4b] > 0 && k_range[p5b] > 0 && k_range[p6b] > 0 &&
          k_range[h1b] > 0 && k_range[h2b] > 0 && k_range[h3b] > 0)) {
      continue;
    }

    ia6_enabled[ia6] = true;
  } // end ia6

  int detailed_stats[9];

  detailed_stats[0] = 0;
  detailed_stats[1] = 0;
  detailed_stats[2] = 0;
  detailed_stats[3] = 0;
  detailed_stats[4] = 0;
  detailed_stats[5] = 0;
  detailed_stats[6] = 0;
  detailed_stats[7] = 0;
  detailed_stats[8] = 0;


  //ia6 -- compute sizes and permutations
  int idx_offset = 0;
  int idx_new_offset = 0;
  for (auto ia6 = 0; ia6 < 9; ia6++) {
    if (!ia6_enabled[ia6]) {
      continue;
    }

    auto [p4b, p5b, p6b, h1b, h2b, h3b] = a3_s1[ia6];
    auto ref_p456_h123 = std::make_tuple(t_p4b, t_p5b, t_p6b, t_h1b, t_h2b, t_h3b);
    auto cur_p456_h123 = std::make_tuple(p4b, p5b, p6b, h1b, h2b, h3b);
    auto cur_p456_h213 = std::make_tuple(p4b, p5b, p6b, h2b, h1b, h3b);
    auto cur_p456_h231 = std::make_tuple(p4b, p5b, p6b, h2b, h3b, h1b);
    auto cur_p546_h123 = std::make_tuple(p5b, p4b, p6b, h1b, h2b, h3b);
    auto cur_p546_h213 = std::make_tuple(p5b, p4b, p6b, h2b, h1b, h3b);
    auto cur_p546_h231 = std::make_tuple(p5b, p4b, p6b, h2b, h3b, h1b);
    auto cur_p564_h123 = std::make_tuple(p5b, p6b, p4b, h1b, h2b, h3b);
    auto cur_p564_h213 = std::make_tuple(p5b, p6b, p4b, h2b, h1b, h3b);
    auto cur_p564_h231 = std::make_tuple(p5b, p6b, p4b, h2b, h3b, h1b);

    size_t dim_common = 1;
    size_t dima_sort  = k_range[p4b]*k_range[h1b];
    size_t dimb_sort =
        k_range[p5b] * k_range[p6b] * k_range[h2b] * k_range[h3b];

    comm_data_elems += dim_common * (dima_sort + dimb_sort);
    
    if (ref_p456_h123 == cur_p456_h123) {
      df_simple_s1_exec[0] = idx_offset;
      *num_enabled_kernels = *num_enabled_kernels + 1;
      idx_new_offset++;
      detailed_stats[0] = detailed_stats[0] + 1;
    }
    if (ref_p456_h123 == cur_p456_h213) {
      df_simple_s1_exec[1] = idx_offset;
      *num_enabled_kernels = *num_enabled_kernels + 1;
      idx_new_offset++;
      detailed_stats[1] = detailed_stats[1] + 1;
    }
    if (ref_p456_h123 == cur_p456_h231) {
      df_simple_s1_exec[2] = idx_offset;
      *num_enabled_kernels = *num_enabled_kernels + 1;
      idx_new_offset++;
      detailed_stats[2] = detailed_stats[2] + 1;
    }
    if (ref_p456_h123 == cur_p546_h123) {
      df_simple_s1_exec[3] = idx_offset;
      *num_enabled_kernels = *num_enabled_kernels + 1;
      idx_new_offset++;
      detailed_stats[3] = detailed_stats[3] + 1;
    }
    if (ref_p456_h123 == cur_p546_h213) {
      df_simple_s1_exec[4] = idx_offset;
      *num_enabled_kernels = *num_enabled_kernels + 1;
      idx_new_offset++;
      detailed_stats[4] = detailed_stats[4] + 1;
    }
    if (ref_p456_h123 == cur_p546_h231) {
      df_simple_s1_exec[5] = idx_offset;
      *num_enabled_kernels = *num_enabled_kernels + 1;
      idx_new_offset++;
      detailed_stats[5] = detailed_stats[5] + 1;
    }
    if (ref_p456_h123 == cur_p564_h123) {
      df_simple_s1_exec[6] = idx_offset;
      *num_enabled_kernels = *num_enabled_kernels + 1;
      idx_new_offset++;
      detailed_stats[6] = detailed_stats[6] + 1;
    }
    if (ref_p456_h123 == cur_p564_h213) {
      df_simple_s1_exec[7] = idx_offset;
      *num_enabled_kernels = *num_enabled_kernels + 1;
      idx_new_offset++;
      detailed_stats[7] = detailed_stats[7] + 1;
    }
    if (ref_p456_h123 == cur_p564_h231) {
      df_simple_s1_exec[8] = idx_offset;
      *num_enabled_kernels = *num_enabled_kernels + 1;
      idx_new_offset++;
      detailed_stats[8] = detailed_stats[8] + 1;
    }
    // 
    idx_offset++;
  } // end ia6

  // printf ("[%s] s1, #: %d >> %d,%d,%d,%d,%d,%d,%d,%d,%d\n", __func__, idx_new_offset,
  //     detailed_stats[0], detailed_stats[1], detailed_stats[2], 
  //     detailed_stats[3], detailed_stats[4], detailed_stats[5], 
  //     detailed_stats[6], detailed_stats[7], detailed_stats[8]);

  // printf ("[%s] s1: %d\n", __func__, idx_new_offset);
} // ccsd_t_data_s1

#endif //CCSD_T_ALL_FUSED_SINGLES_HPP_