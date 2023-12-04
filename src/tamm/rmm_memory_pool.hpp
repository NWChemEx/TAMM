#pragma once

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include <tamm/gpu_streams.hpp>

#include "tamm/mr/gpu_memory_resource.hpp"
#include "tamm/mr/per_device_resource.hpp"
#include "tamm/mr/pinned_memory_resource.hpp"
#else
#include <sys/sysinfo.h>
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
  const char* tammGpupoolsize = std::getenv("TAMM_GPU_POOL");
  uint32_t    usinggpupool    = 80;
  if(tammGpupoolsize != nullptr) { usinggpupool = std::atoi(tammGpupoolsize); }
  return usinggpupool;
}();

// TAMM_CPU_POOL
static const uint32_t tamm_cpu_pool = [] {
  const char* tammCpupoolsize = std::getenv("TAMM_CPU_POOL");
  uint32_t    usingcpupool    = 36;
  if(tammCpupoolsize != nullptr) { usingcpupool = std::atoi(tammCpupoolsize); }
  return usingcpupool;
}();

// TAMM_RANKS_PER_GPU_POOL
static const uint32_t tamm_rpg = [] {
  const char* tammrpg  = std::getenv("TAMM_RANKS_PER_GPU_POOL");
  uint32_t    usingrpg = 1;
  if(tammrpg != nullptr) { usingrpg = std::atoi(tammrpg); }
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
      size_t max_host_bytes{0};

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
      size_t free{}, total{};
      gpuMemGetInfo(&free, &total);

      size_t max_device_bytes{0};
      // Allocate 35% of total free memory on GPU
      // Similarly allocate the same size for the CPU pool too
      // For the host-pinned memory allcoate 15% of the free memory reported
      max_device_bytes = ((detail::tamm_gpu_pool / 100.0) * free) / detail::tamm_rpg;
#ifdef USE_MEMKIND
      // Idea is to allocate 0.15 * 64Gb=~9Gb per rank. Such that 6 ranks from
      // 1 Aurora socket maps to 54Gb of HBM out of 64Gb capacity per socket.
      max_host_bytes = 0.15 * free;
#else
      max_host_bytes = ((detail::tamm_cpu_pool / 100.0) * free) / detail::tamm_rpg;
#endif

      deviceMR =
        std::make_unique<device_pool_mr>(new rmm::mr::gpu_memory_resource, max_device_bytes);
      hostMR = std::make_unique<host_pool_mr>(new rmm::mr::new_delete_resource, max_host_bytes);
      // #if defined(USE_CUDA) || defined(USE_HIP)
      //       size_t max_pinned_host_bytes{0};
      //       max_pinned_host_bytes = 0.18 * free;
      //       pinnedHostMR = std::make_unique<pinned_pool_mr>(new rmm::mr::pinned_memory_resource,
      //                                                       max_pinned_host_bytes);
      // #endif

#else // CPU-only
      struct sysinfo cpumeminfo_;
      sysinfo(&cpumeminfo_);
      max_host_bytes = cpumeminfo_.freeram * cpumeminfo_.mem_unit;

      max_host_bytes *= 0.05;
      hostMR = std::make_unique<host_pool_mr>(new rmm::mr::new_delete_resource, max_host_bytes);
#endif

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
