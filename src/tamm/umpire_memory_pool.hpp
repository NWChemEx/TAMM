#pragma once

#include <tamm/gpu_streams.hpp>

#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

// Don't use the thread-safe allocators from Umpire for now.
// #define ENABLE_UMPIRE_THREADSAFE 0

namespace tamm {

class UmpireMemoryManager {
protected:
    umpire::Allocator host_allocator;
    umpire::Allocator device_allocator;
    umpire::Allocator pinned_host_allocator;
    bool initialized{false};

private:
    UmpireMemoryManager() {
	if(!initialized) {
	    size_t free{}, total{};
	    gpuMemGetInfo(&free, &total);

	    // Allocate 45% of total free memory on GPU
	    // Similarly allocate the same size for the CPU pool too
	    // For the host-pinned memory allcoate 5% of the free memory reported
	    // Motivation: When 2 GA progress-ranks are used per GPU
	    // the GPU might furnish the memory-pools apporiately for each rank
	    size_t initial_device_bytes = 0.45 * free;
	    size_t initial_host_bytes = 0.45 * free;
	    size_t initial_pinned_host_bytes = 0.05 * free;

	    auto pooled_host_allocator =
		umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>(
		    "HOST_pool", umpire::ResourceManager::getInstance().getAllocator("HOST"), initial_host_bytes);
	    auto pooled_device_allocator =
		umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>(
		    "DEVICE_pool", umpire::ResourceManager::getInstance().getAllocator("DEVICE"), initial_device_bytes);
	    auto pooled_pinned_host_allocator =
		umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>(
		    "PINNED_pool", pinned_host_allocator, initial_pinned_host_bytes);

#ifdef ENABLE_UMPIRE_THREADSAFE
	    auto thread_safe_pooled_host_allocator =
		umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
		    "HOST_thread_safe_pool", pooled_host_allocator);
	    auto thread_safe_pooled_device_allocator =
		umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
		    "DEVICE_thread_safe_pool", pooled_device_allocator);
	    auto thread_safe_pooled_pinned_host_allocator =
		umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
		    "PINNED_thread_safe_pool", pooled_pinned_host_allocator);

	    this->host_allocator = thread_safe_pooled_host_allocator;
	    this->device_allocator = thread_safe_pooled_device_allocator;
	    this->pinned_host_allocator = thread_safe_pooled_pinned_host_allocator;
#else
	    this->host_allocator = pooled_host_allocator;
	    this->device_allocator = pooled_device_allocator;
	    this->pinned_host_allocator = pooled_pinned_host_allocator;
#endif

	    initialized = true;
	}
    }

public:
    umpire::Allocator& getUmpirePinnedHostAllocator() {
	return pinned_host_allocator;
    }
    umpire::Allocator& getUmpireDeviceAllocator() {
	return device_allocator;
    }
    umpire::Allocator& getUmpireHostAllocator() {
	return host_allocator;
    }
    
  /// Returns the instance of device manager singleton.
  inline static UmpireMemoryManager& getInstance() {
    static UmpireMemoryManager d_m{};
    return d_m;
  }

  UmpireMemoryManager(const UmpireMemoryManager&)            = delete;
  UmpireMemoryManager& operator=(const UmpireMemoryManager&) = delete;
  UmpireMemoryManager(UmpireMemoryManager&&)                 = delete;
  UmpireMemoryManager& operator=(UmpireMemoryManager&&)      = delete;

}; // class UmpireMemoryManager

}

// namespace tamm {
// namespace memory {
// namespace internal {

// // void initializeUmpirePinnedHostAllocator(std::size_t initial_bytes);
// // void initializeUmpireHostAllocator(std::size_t initial_bytes);
// // void initializeUmpireDeviceAllocator(std::size_t initial_bytes);

// static umpire::Allocator& getUmpirePinnedHostAllocator() {
//   static auto pinned_host_allocator = umpire::ResourceManager::getInstance().getAllocator("PINNED");
//   return pinned_host_allocator;
// }
// static umpire::Allocator& getUmpireDeviceAllocator() {
//   static auto device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
//   return device_allocator;
// }
// static umpire::Allocator& getUmpireHostAllocator() {
//   static auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
//   return host_allocator;
// }

// static void initializeUmpirePinnedHostAllocator(std::size_t initial_bytes) {
//   static bool initialized = false;

//   if(!initialized) {
//     auto pooled_pinned_host_allocator =
//       umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>(
//         "PINNED_pool", umpire::ResourceManager::getInstance().getAllocator("PINNED"), initial_bytes);

// #ifdef ENABLE_UMPIRE_THREADSAFE
//     auto thread_safe_pooled_pinned_host_allocator =
//       umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
//         "PINNED_thread_safe_pool", pooled_pinned_host_allocator);
//     memory::internal::getUmpirePinnedHostAllocator() = thread_safe_pooled_pinned_host_allocator;
// #else
//     memory::internal::getUmpirePinnedHostAllocator() = pooled_pinned_host_allocator;
// #endif

//     initialized = true;
//   }
// }

// static void initializeUmpireDeviceAllocator(std::size_t initial_bytes) {
//   static bool initialized = false;

//   if(!initialized) {
//     auto device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
//     auto pooled_device_allocator =
//       umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>(
//         "DEVICE_pool", device_allocator, initial_bytes);

// #ifdef ENABLE_UMPIRE_THREADSAFE
//     auto thread_safe_pooled_device_allocator =
//       umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
//         "DEVICE_thread_safe_pool", pooled_device_allocator);
//     memory::internal::getUmpireDeviceAllocator() = thread_safe_pooled_device_allocator;
// #else
//     memory::internal::getUmpireDeviceAllocator()     = pooled_device_allocator;
// #endif

//     initialized = true;
//   }
// }

// static void initializeUmpireHostAllocator(std::size_t initial_bytes) {
//   static bool initialized = false;

//   if(!initialized) {
//     auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
//     auto pooled_host_allocator =
//       umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>(
//         "HOST_pool", host_allocator, initial_bytes);

// #ifdef ENABLE_UMPIRE_THREADSAFE
//     auto thread_safe_pooled_host_allocator =
//       umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::ThreadSafeAllocator>(
//         "HOST_thread_safe_pool", pooled_host_allocator);
//     memory::internal::getUmpireHostAllocator() = thread_safe_pooled_host_allocator;
// #else
//     memory::internal::getUmpireHostAllocator()       = pooled_host_allocator;
// #endif

//     initialized = true;
//   }
// }

// } // namespace internal
// } // namespace memory
// } // namespace tamm
