#pragma once

#include <tamm/gpu_streams.hpp>
#include <tamm/errors.hpp>

#include "tamm/mr/device_memory_resource.hpp"
#include "tamm/mr/gpu_memory_resource.hpp"
#include "tamm/mr/host_memory_resource.hpp"
#include "tamm/mr/new_delete_resource.hpp"
#include "tamm/mr/per_device_resource.hpp"
#include "tamm/mr/pinned_memory_resource.hpp"
#include "tamm/mr/pool_memory_resource.hpp"

namespace tamm {

// TAMM_GPU_POOLSIZE (sets poolsize in GB for GPU)
static const uint32_t tamm_gpu_poolsize = [] {
  const char* tammGpupoolsize = std::getenv("TAMM_GPU_POOLSIZE");
  uint32_t    usingGpupoolSize   = 0;
  if(tammGpupoolsize) { usingGpupoolSize = std::atoi(tammGpupoolsize); }
  return usingGpupoolSize;
}();

class RMMMemoryManager {
protected:
  using device_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  using host_pool_mr   = rmm::mr::pool_memory_resource<rmm::mr::host_memory_resource>;
  std::unique_ptr<device_pool_mr> deviceMR;
  std::unique_ptr<host_pool_mr>   hostMR;
  // std::unique_ptr<host_pool_mr>   pinnedHostMR;

private:
  RMMMemoryManager() {
    size_t free{}, total{};
    gpuMemGetInfo(&free, &total);

    size_t max_device_bytes{0};
    size_t max_host_bytes{0};
    // size_t max_pinned_host_bytes{0};
    if(tamm_gpu_poolsize) {
      // sets the GPU & CPU pool size as requested by the env variable TAMM_GPU_POOLSIZE
      EXPECTS(tamm_gpu_poolsize < free);

      max_device_bytes = tamm_gpu_poolsize;
      max_host_bytes   = tamm_gpu_poolsize;
    } else {
      // Allocate 80% of total free memory on GPU
      // Similarly allocate the same size for the CPU pool too
      // For the host-pinned memory allcoate 5% of the free memory reported
      max_device_bytes      = 0.80 * free;
      max_host_bytes        = 0.80 * free;
      // max_pinned_host_bytes = 0.05 * free;
    }

    deviceMR = std::make_unique<device_pool_mr>(new rmm::mr::gpu_memory_resource, max_device_bytes);
    hostMR   = std::make_unique<host_pool_mr>(new rmm::mr::new_delete_resource, max_host_bytes);
    // pinnedHostMR =
    //   std::make_unique<host_pool_mr>(new rmm::mr::pinned_memory_resource, max_pinned_host_bytes);
  }

public:
  /// Returns a RMM device pool handle
  device_pool_mr& getDeviceMemoryPool() { return *(deviceMR.get()); }
  /// Returns a RMM host pool handle
  host_pool_mr& getHostMemoryPool() { return *(hostMR.get()); }
  // /// Returns a RMM pinnedHost pool handle
  // host_pool_mr& getPinnedMemoryPool() { return *(pinnedHostMR.get()); }

  /// Returns the instance of device manager singleton.
  inline static RMMMemoryManager& getInstance() {
    static RMMMemoryManager d_m{};
    return d_m;
  }

  RMMMemoryManager(const RMMMemoryManager&)            = delete;
  RMMMemoryManager& operator=(const RMMMemoryManager&) = delete;
  RMMMemoryManager(RMMMemoryManager&&)                 = delete;
  RMMMemoryManager& operator=(RMMMemoryManager&&)      = delete;
};

} // namespace tamm
