#pragma once

#include <umpire/umpire.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

// Don't use the thread-safe allocators from Umpire for now.
//#define ENABLE_UMPIRE_THREADSAFE 0

namespace tamm {
namespace memory {
namespace internal {

umpire::Allocator& getUmpirePinnedHostAllocator() {
    static auto pinned_host_allocator = umpire::ResourceManager::getInstance().getAllocator("PINNED");
    return pinned_host_allocator;
}
umpire::Allocator& getUmpireDeviceAllocator() {
    static auto device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
    return device_allocator;
}
umpire::Allocator& getUmpireHostAllocator() {
    static auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
    return host_allocator;
}


void initializeUmpirePinnedHostAllocator(std::size_t initial_bytes) {
    static bool initialized = false;

    if (!initialized) {
        auto pinned_host_allocator = umpire::ResourceManager::getInstance().getAllocator("PINNED");
        auto pooled_pinned_host_allocator =
            umpire::ResourceManager::getInstance().makeAllocator<umpire::strategy::QuickPool>("PINNED_pool",
                                                                                              pinned_host_allocator,
                                                                                              initial_bytes);

        #ifdef ENABLE_UMPIRE_THREADSAFE
        auto thread_safe_pooled_pinned_host_allocator =
            umpire::ResourceManager::getInstance()
            .makeAllocator<umpire::strategy::ThreadSafeAllocator>("PINNED_thread_safe_pool",
                                                                  pooled_pinned_host_allocator);
        memory::internal::getUmpirePinnedHostAllocator() = thread_safe_pooled_pinned_host_allocator;
        #else
        memory::internal::getUmpirePinnedHostAllocator() = pooled_pinned_host_allocator;
        #endif

        initialized = true;
    }
}

void initializeUmpireDeviceAllocator(std::size_t initial_bytes) {
    static bool initialized = false;

    if (!initialized) {
        auto device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
        auto pooled_device_allocator =
            umpire::ResourceManager::getInstance()
            .makeAllocator<umpire::strategy::QuickPool>("DEVICE_pool", device_allocator, initial_bytes);

        #ifdef ENABLE_UMPIRE_THREADSAFE
        auto thread_safe_pooled_device_allocator =
            umpire::ResourceManager::getInstance()
            .makeAllocator<umpire::strategy::ThreadSafeAllocator>("DEVICE_thread_safe_pool",
                                                                  pooled_device_allocator);
        memory::internal::getUmpireDeviceAllocator() = thread_safe_pooled_device_allocator;
        #else
        memory::internal::getUmpireDeviceAllocator() = pooled_device_allocator;
        #endif

        initialized = true;
    }
}


void initializeUmpireHostAllocator(std::size_t initial_bytes) {
    static bool initialized = false;

    if (!initialized) {
        auto host_allocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
        auto pooled_host_allocator =
            umpire::ResourceManager::getInstance()
            .makeAllocator<umpire::strategy::QuickPool>("HOST_pool", host_allocator, initial_bytes);

        #ifdef ENABLE_UMPIRE_THREADSAFE
        auto thread_safe_pooled_host_allocator =
            umpire::ResourceManager::getInstance()
            .makeAllocator<umpire::strategy::ThreadSafeAllocator>("HOST_thread_safe_pool",
                                                                  pooled_host_allocator);
        memory::internal::getUmpireHostAllocator() = thread_safe_pooled_host_allocator;
        #else
        memory::internal::getUmpireHostAllocator() = pooled_host_allocator;
        #endif

        initialized = true;
    }
}

void finalizeUmpireHostAllocator() {}
void finalizeUmpirePinnedHostAllocator() {}
void finalizeUmpireDeviceAllocator() {}
}
}
}
