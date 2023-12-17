#pragma once

#include <sys/sysinfo.h>

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#else
#include <ga/ga.h>
#endif

#include <tamm/gpu_streams.hpp>

#include "tamm/mr/gpu_memory_resource.hpp"
#include "tamm/mr/per_device_resource.hpp"
#include "tamm/mr/pinned_memory_resource.hpp"
#endif

#include <tamm/errors.hpp>

#include "tamm/mr/device_memory_resource.hpp"
#include "tamm/mr/host_memory_resource.hpp"
#include "tamm/mr/new_delete_resource.hpp"
#include "tamm/mr/pool_memory_resource.hpp"

namespace tamm {

namespace detail {
// TAMM_GPU_POOL
static const uint32_t tamm_gpu_pool = [] {
  uint32_t usinggpupool = 80;
  if(const char* tammGpupoolsize = std::getenv("TAMM_GPU_POOL")) {
    usinggpupool = std::atoi(tammGpupoolsize);
  }
  return usinggpupool;
}();

// TAMM_CPU_POOL
static const uint32_t tamm_cpu_pool = [] {
  uint32_t usingcpupool = 80;
  if(const char* tammCpupoolsize = std::getenv("TAMM_CPU_POOL")) {
    usingcpupool = std::atoi(tammCpupoolsize);
  }
  return usingcpupool;
}();

// TAMM_RANKS_PER_GPU_POOL
static const uint32_t tamm_rpg = [] {
  uint32_t usingrpg = 1;
  if(const char* tammrpg = std::getenv("TAMM_RANKS_PER_GPU_POOL")) {
    usingrpg = std::atoi(tammrpg);
  }
  return usingrpg;
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
#endif
  // #if defined(USE_CUDA) || defined(USE_HIP)
  //   using pinned_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>;
  //   std::unique_ptr<pinned_pool_mr> pinnedHostMR;
  // #endif

private:
  RMMMemoryManager() { initialize(); }

public:
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  /// Returns a RMM device pool handle
  device_pool_mr& getDeviceMemoryPool() { return *(deviceMR.get()); }
#endif

  // #if defined(USE_CUDA) || defined(USE_HIP)
  //   /// Returns a RMM pinnedHost pool handle
  //   pinned_pool_mr& getPinnedMemoryPool() { return *(pinnedHostMR.get()); }
  // #elif defined(USE_DPCPP)
  //   /// Returns a RMM pinnedHost pool handle
  //   host_pool_mr& getPinnedMemoryPool() { return *(hostMR.get()); }
  // #endif

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
// #elif defined(USE_CUDA) || defined(USE_HIP)
//     pinnedHostMR.reset();
#endif

    this->invalid_state = true;
  }

  void initialize() {
    if(this->invalid_state) {
      // Set the CPU memory-pool
      size_t max_host_bytes{0};
      // Number of user-MPI ranks is needed for efficient CPU-pool size
      int ranks_pn_ = 0;
#if defined(USE_UPCXX)
      ranks_pn_ = upcxx::local_team().rank_n();
#else
      ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_nodeid());
#endif

      struct sysinfo cpumeminfo_;
      sysinfo(&cpumeminfo_);
      // 50% allocation was reserved for the GA distributed arrays followed by the
      // memory pool creation
      max_host_bytes = 0.5 * cpumeminfo_.freeram * cpumeminfo_.mem_unit;
      // Use only "tamm_cpu_pool" percent of the free memory let
      max_host_bytes *= (detail::tamm_cpu_pool / 100.0);

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
      size_t free{}, total{};
      gpuMemGetInfo(&free, &total);
      size_t max_device_bytes{0};
      max_device_bytes = ((detail::tamm_gpu_pool / 100.0) * free) / detail::tamm_rpg;

#ifdef USE_MEMKIND
      max_host_bytes = 128849018880; // 120 GB (on both socket os Aurora)
#endif

      deviceMR =
        std::make_unique<device_pool_mr>(new rmm::mr::gpu_memory_resource, max_device_bytes);
      // #if defined(USE_CUDA) || defined(USE_HIP)
      //       size_t max_pinned_host_bytes{0};
      //       max_pinned_host_bytes = 0.18 * free;
      //       pinnedHostMR = std::make_unique<pinned_pool_mr>(new rmm::mr::pinned_memory_resource,
      //                                                       max_pinned_host_bytes);
      // #endif
#endif
      max_host_bytes /= ranks_pn_;
      hostMR = std::make_unique<host_pool_mr>(new rmm::mr::new_delete_resource, max_host_bytes);

      // after setting up the pool: change the invalid_state to FALSE
      invalid_state = false;
    }
  }

  RMMMemoryManager(const RMMMemoryManager&)            = delete;
  RMMMemoryManager& operator=(const RMMMemoryManager&) = delete;
  RMMMemoryManager(RMMMemoryManager&&)                 = delete;
  RMMMemoryManager& operator=(RMMMemoryManager&&)      = delete;
};

// The reset pool & reinitialize only is being used for the (T) segement of cannonical
static inline void reset_rmm_pool() { RMMMemoryManager::getInstance().reset(); }

static inline void reinitialize_rmm_pool() { RMMMemoryManager::getInstance().initialize(); }

} // namespace tamm
