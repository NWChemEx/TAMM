#pragma once

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)

#include <tamm/gpu_streams.hpp>

#include "tamm/mr/gpu_memory_resource.hpp"
#include "tamm/mr/per_device_resource.hpp"
// #include "tamm/mr/pinned_memory_resource.hpp"
#endif

#include <tamm/errors.hpp>

#include "tamm/mr/aligned.hpp"
#include "tamm/mr/device_memory_resource.hpp"
#include "tamm/mr/host_memory_resource.hpp"
#include "tamm/mr/new_delete_resource.hpp"
#include "tamm/mr/pool_memory_resource.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace tamm {

namespace detail {
// TAMM_ENABLE_SPRHBM = 0(default), 1
static const uint32_t tamm_enable_sprhbm = [] {
  const char* tammEnableSprHBM = std::getenv("TAMM_ENABLE_SPRHBM");
  uint32_t    usinghbm         = 0;
  if(tammEnableSprHBM != nullptr) { usinghbm = std::atoi(tammEnableSprHBM); }
  return usinghbm;
}();

// Parse a percentage-valued env var. Values outside (0,100] are rejected rather than
// silently wrapped: these feed directly into the pool size, and e.g. TAMM_GPU_POOL=-50
// would otherwise wrap through uint32_t to ~4.29e9 and produce a nonsense pool request.
static uint32_t parse_pool_percent(const char* name, uint32_t default_value) {
  const char* raw = std::getenv(name);
  if(raw == nullptr) { return default_value; }

  char*     end = nullptr;
  long long val = std::strtoll(raw, &end, 10);
  if(end == raw || *end != '\0' || val <= 0 || val > 100) {
    std::ostringstream os;
    os << "[TAMM ERROR] " << name << " must be an integer percentage in (0, 100]; got \"" << raw
       << "\".\n"
       << __FILE__ << ":L" << __LINE__;
    tamm_terminate(os.str());
  }
  return static_cast<uint32_t>(val);
}

// TAMM_GPU_POOL
static const uint32_t tamm_gpu_pool = parse_pool_percent("TAMM_GPU_POOL", 80);

// TAMM_CPU_POOL
static const uint32_t tamm_cpu_pool = parse_pool_percent("TAMM_CPU_POOL", 100);

// TAMM_RMM_DEBUG = 0(default), 1
// When set, rank 0 reports the computed pool sizes and the inputs used to derive them.
static const bool tamm_rmm_debug = [] {
  const char* tammRmmDebug = std::getenv("TAMM_RMM_DEBUG");
  return (tammRmmDebug != nullptr) && (std::atoi(tammRmmDebug) != 0);
}();

} // namespace detail

class RMMMemoryManager {
protected:
  bool invalid_state{true};
  using host_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::host_memory_resource>;
  std::unique_ptr<host_pool_mr> hostMR;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  using device_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  std::unique_ptr<device_pool_mr> deviceMR;
  // using pinned_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>;
  // std::unique_ptr<pinned_pool_mr> pinnedHostMR;
#endif

private:
  RMMMemoryManager() { initialize(); }
  // TAMM_RANKS_PER_GPU_POOL
  uint32_t tamm_rpg;

public:
  uint32_t get_rpg() { return tamm_rpg; }

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  /// Returns a RMM device pool handle
  device_pool_mr& getDeviceMemoryPool() { return *(deviceMR.get()); }
  // /// Returns a RMM pinnedHost pool handle
  // pinned_pool_mr& getPinnedMemoryPool() { return *(pinnedHostMR.get()); }
#endif

  /// Returns a RMM host pool handle
  host_pool_mr& getHostMemoryPool() { return *(hostMR.get()); }

  /// Returns the instance of device manager singleton.
  inline static RMMMemoryManager& getInstance() {
    static RMMMemoryManager d_m{};
    return d_m;
  }

  void reset() {
    hostMR.reset();
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    deviceMR.reset();
    // pinnedHostMR.reset();
#endif

    this->invalid_state = true;
  }

  void initialize() {
    if(!this->invalid_state) return;

    tamm_rpg = 1;
    // Number of user-MPI ranks is needed for efficient CPU-pool size
    int ranks_pn_ = 0;
#if defined(USE_UPCXX)
    ranks_pn_ = upcxx::local_team().rank_n();
#else
    ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_nodeid());
#endif

    // ranks_pn_ is used as a divisor when apportioning the host pool; a zero here would be
    // an integer division by zero (SIGFPE) rather than a wrapped value.
    if(ranks_pn_ <= 0) {
      std::ostringstream os;
      os << "[TAMM ERROR] Detected " << ranks_pn_ << " ranks per node; expected at least 1.\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    long max_host_bytes{0};

    // Currently these checks are limited to CUDA & HIP.
    // Since accessing system APIs would be pretty expensive,
    // these checks can be done only by the master rank.
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    int world_rank_    = 0;
    int ngpus_per_node = 0;
#if defined(USE_UPCXX)
    world_rank_ = upcxx::rank_me();
#else
    world_rank_ = GA_Nodeid();
#endif // USE_UPCXX

    if(world_rank_ == 0) { tamm::getHardwareGPUCount(&ngpus_per_node); }

    // Only ngpus_per_node is rank-0 data. tamm_rpg is derived identically on every rank from
    // ngpus_per_node below, so broadcasting it here (before it is computed) was a no-op.
#if defined(USE_UPCXX)
    upcxx::broadcast(&ngpus_per_node, 0).wait();
#else
    MPI_Bcast(&ngpus_per_node, 1, MPI_INT, 0, GA_MPI_Comm());
#endif

    if(ngpus_per_node <= 0) {
      std::ostringstream os;
      os << "[TAMM ERROR] Detected " << ngpus_per_node << " GPUs per node; expected at least 1.\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    if(ranks_pn_ > ngpus_per_node) {
      if(ranks_pn_ % ngpus_per_node != 0) {
        std::ostringstream os;
        os << "[TAMM ERROR] Num_ranks_per_node (" << ranks_pn_
           << ") is not a multiple of num_gpus_per_node (" << ngpus_per_node << ")\n"
           << __FILE__ << ":L" << __LINE__;
        tamm_terminate(os.str());
      }
      tamm_rpg = ranks_pn_ / ngpus_per_node;
    }

#endif // USE_CUDA, USE_HIP

#if defined(__APPLE__)
    size_t cpu_mem_per_node;
    size_t size_mpn = sizeof(cpu_mem_per_node);
    // TODO: query for freeram, not total
    sysctlbyname("hw.memsize", &(cpu_mem_per_node), &size_mpn, nullptr, 0);
    max_host_bytes = 0.5 * cpu_mem_per_node;
    // Use only "tamm_cpu_pool" percent of the remaining memory
    max_host_bytes *= (detail::tamm_cpu_pool / 100.0);
#elif defined(TAMM_DISABLE_LIBNUMA)
    struct sysinfo cpumeminfo_;
    sysinfo(&cpumeminfo_);
    // 50% allocation was reserved for the GA distributed arrays followed by the
    // memory pool creation
    max_host_bytes = 0.5 * cpumeminfo_.freeram * cpumeminfo_.mem_unit;
    // Use only "tamm_cpu_pool" percent of the remaining memory
    max_host_bytes *= (detail::tamm_cpu_pool / 100.0);
#else
    // Set the CPU memory-pool
    if(numa_available() == -1) {
      std::ostringstream os;
      os << "[TAMM ERROR] numa APIs are not available!\n" << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    numa_set_bind_policy(1);
    numa_set_strict(1);
    int numNumaNodes = numa_num_task_nodes();

    // for ranks_pn_=1, there is no need to check the mapping to numa-nodes (mostly used for CI)
    // for ranks_pn_ > numNumaNodes, it has to be divisble by the number of numa-domains in the
    // system
    if(ranks_pn_ >= numNumaNodes && ranks_pn_ > 1) {
      if((ranks_pn_ % numNumaNodes) != 0) {
        std::ostringstream os;
        os << "[TAMM ERROR] Number of user MPI ranks(" << ranks_pn_
           << ") is not a multiple of number of numa-nodes(" << numNumaNodes << ")! \n"
           << __FILE__ << ":L" << __LINE__;
        tamm_terminate(os.str());
      }
    }
    struct bitmask* numaNodes = numa_get_mems_allowed();
    numa_bind(numaNodes);
    numa_bitmask_free(numaNodes);

    int numa_id = numa_node_of_cpu(sched_getcpu());
    /* long numa_total_size = */ numa_node_size(numa_id, &max_host_bytes);
    max_host_bytes *= 0.40; // reserve 40% only of the free numa-node memory (reserving rest of
                            // GA, non-pool allocations)

    if(numNumaNodes > 1) { // please the systems with just 1 Numa partitions
      // Identify the NUMA distance for faster numa-regions
      std::map<int, int> numadist_id;
      for(int j = 0; j < numNumaNodes; j++) {
        if(numa_id != j) { numadist_id[j] = numa_distance(numa_id, j); }
      }
      int  val    = numadist_id.begin()->second;
      auto result = std::all_of(
        std::next(numadist_id.begin()), numadist_id.end(),
        [val](typename std::map<int, int>::const_reference t) { return t.second == val; });
      if(!result) { // There are some faster NUMA domains available than the defaults (only for
                    // Aurora)
        auto it =
          std::min_element(numadist_id.begin(), numadist_id.end(),
                           [](const auto& l, const auto& r) { return l.second < r.second; });

        numNumaNodes /= 2; // This is done for the Aurora nodes only

        if(detail::tamm_enable_sprhbm) {
          numa_id = it->first;
          numa_set_preferred(numa_id);
          /* numa_total_size = */ numa_node_size(numa_id, &max_host_bytes);
          max_host_bytes *=
            0.94; // One can use full HBM memory capacity, since the DDR is left for GA
        }
      }
    } // numNumaNodes > 1

    max_host_bytes *=
      (detail::tamm_cpu_pool / 100.0); // Use only "tamm_cpu_pool" percent of the left-overs
    max_host_bytes /= ((numNumaNodes > 1)
                         ? ((ranks_pn_ >= numNumaNodes) ? (ranks_pn_ / numNumaNodes) : 1)
                         : ranks_pn_);
#endif

    // Validate the host size before reserving anything from the GPU, so a bad host-memory
    // query does not leave an orphaned device reservation behind when we terminate.
    // max_host_bytes is a signed long fed by numa_node_size()/sysinfo(), either of which can
    // report failure as a negative value; casting that to size_t would wrap to a huge request.
    if(max_host_bytes <= 0) {
      std::ostringstream os;
      os << "[TAMM ERROR] Computed a non-positive CPU memory-pool size (" << max_host_bytes
         << " bytes).\n"
         << "  The available-memory query failed, or TAMM_CPU_POOL is too small.\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    size_t const max_host_bytes_aligned = rmm::detail::align_down(
      static_cast<size_t>(max_host_bytes), rmm::detail::RMM_ALLOCATION_ALIGNMENT);

    if(max_host_bytes_aligned == 0) {
      std::ostringstream os;
      os << "[TAMM ERROR] CPU memory-pool size rounds down to zero (" << max_host_bytes
         << " bytes before alignment).\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    size_t free{}, total{};
    gpuMemGetInfo(&free, &total);
    size_t max_device_bytes{0};
    max_device_bytes = ((detail::tamm_gpu_pool / 100.0) * free) / tamm_rpg;
    // The pool is fixed-size and never grows, so hand it an aligned size; otherwise the
    // trailing partial-alignment bytes are unusable and silently shrink the usable pool.
    max_device_bytes =
      rmm::detail::align_down(max_device_bytes, rmm::detail::RMM_ALLOCATION_ALIGNMENT);

    if(detail::tamm_rmm_debug && world_rank_ == 0) {
      std::cout << "[TAMM RMM] device pool: " << max_device_bytes << " B ("
                << (max_device_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB) per rank | "
                << "gpu free=" << free << " B, total=" << total << " B | "
                << "ranks/node=" << ranks_pn_ << ", gpus/node=" << ngpus_per_node
                << ", ranks-per-gpu=" << tamm_rpg << ", TAMM_GPU_POOL=" << detail::tamm_gpu_pool
                << "%" << std::endl;
    }

    deviceMR = std::make_unique<device_pool_mr>(new rmm::mr::gpu_memory_resource, max_device_bytes);

    // size_t max_pinned_host_bytes{0};
    // max_pinned_host_bytes = 0.18 * free;
    // pinnedHostMR =
    //   std::make_unique<pinned_pool_mr>(new rmm::mr::pinned_memory_resource,
    //   max_pinned_host_bytes);
#endif

    if(detail::tamm_rmm_debug) {
      int host_dbg_rank = 0;
#if defined(USE_UPCXX)
      host_dbg_rank = upcxx::rank_me();
#else
      host_dbg_rank = GA_Nodeid();
#endif
      if(host_dbg_rank == 0) {
        std::cout << "[TAMM RMM] host pool  : " << max_host_bytes_aligned << " B ("
                  << (max_host_bytes_aligned / (1024.0 * 1024.0 * 1024.0)) << " GiB) per rank | "
                  << "ranks/node=" << ranks_pn_ << ", TAMM_CPU_POOL=" << detail::tamm_cpu_pool
                  << "%" << std::endl;
      }
    }

    hostMR =
      std::make_unique<host_pool_mr>(new rmm::mr::new_delete_resource, max_host_bytes_aligned);

    // after setting up the pool: change the invalid_state to FALSE
    invalid_state = false;
  }

  RMMMemoryManager(const RMMMemoryManager&)            = delete;
  RMMMemoryManager& operator=(const RMMMemoryManager&) = delete;
  RMMMemoryManager(RMMMemoryManager&&)                 = delete;
  RMMMemoryManager& operator=(RMMMemoryManager&&)      = delete;
};

static inline uint32_t ranks_per_gpu_pool() { return RMMMemoryManager::getInstance().get_rpg(); }

// The reset pool & reinitialize only is being used for the (T) segement of cannonical
static inline void reset_rmm_pool() { RMMMemoryManager::getInstance().reset(); }

static inline void reinitialize_rmm_pool() { RMMMemoryManager::getInstance().initialize(); }

} // namespace tamm
