#pragma once

#include <tamm/gpu_streams.hpp>

#include "tamm/mr/device_memory_resource.hpp"
#include "tamm/mr/gpu_memory_resource.hpp"
#include "tamm/mr/per_device_resource.hpp"
#include "tamm/mr/pool_memory_resource.hpp"

namespace tamm {

class RMMMemoryManager {
protected:
    using device_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
    std::unique_ptr<device_pool_mr> deviceMR;

private:
    RMMMemoryManager() {
	size_t free{}, total{};
	gpuMemGetInfo(&free, &total);

	// Allocate 45% of total free memory on GPU
	// Similarly allocate the same size for the CPU pool too
	// For the host-pinned memory allcoate 5% of the free memory reported
	// Motivation: When 2 GA progress-ranks are used per GPU
	// the GPU might furnish the memory-pools apporiately for each rank
	size_t max_device_bytes = 0.40 * free;
	size_t max_host_bytes = 0.40 * free;
	size_t max_pinned_host_bytes = 0.05 * free;

	deviceMR = std::make_unique<device_pool_mr>( rmm::mr::get_per_device_resource(0),
						     max_device_bytes );
    }

public:
  /// Returns a RMM or default pool handle
  device_pool_mr& getMemoryPool() {
      return *(deviceMR.get());
  }

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
