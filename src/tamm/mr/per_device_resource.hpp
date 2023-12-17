#pragma once

#include "device_memory_resource.hpp"
#include "gpu_memory_resource.hpp"

// Macros used for defining symbol visibility, only GLIBC is supported
#if(defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define RMM_EXPORT __attribute__((visibility("default")))
#define RMM_HIDDEN __attribute__((visibility("hidden")))
#else
#define RMM_EXPORT
#define RMM_HIDDEN
#endif

#include <map>

/**
 * @file per_device_resource.hpp
 * @brief Management of per-device `device_memory_resource`s
 *
 * One might wish to construct a `device_memory_resource` and use it for (de)allocation
 * without explicit dependency injection, i.e., passing a reference to that object to all places it
 * is to be used. Instead, one might want to set their resource as a "default" and have it be used
 * in all places where another resource has not been explicitly specified. In applications with
 * multiple GPUs in the same process, it may also be necessary to maintain independent default
 * resources for each device. To this end, the `set_per_device_resource` and
 * `get_per_device_resource` functions enable mapping a CUDA device id to a `device_memory_resource`
 * pointer.
 *
 * For example, given a pointer, `mr`, to a `device_memory_resource` object, calling
 * `set_per_device_resource(cuda_device_id{0}, mr)` will establish a mapping between CUDA device 0
 * and `mr` such that all future calls to `get_per_device_resource(cuda_device_id{0})` will return
 * the same pointer, `mr`. In this way, all places that use the resource returned from
 * `get_per_device_resource` for (de)allocation will use the user provided resource, `mr`.
 *
 * @note `device_memory_resource`s make CUDA API calls without setting the current CUDA device.
 * Therefore a memory resource should only be used with the current CUDA device set to the device
 * that was active when the memory resource was created. Calling `set_per_device_resource(id, mr)`
 * is only valid if `id` refers to the CUDA device that was active when `mr` was created.
 *
 * If no resource was explicitly set for a given device specified by `id`, then
 * `get_per_device_resource(id)` will return a pointer to a `gpu_memory_resource`.
 *
 * To fetch and modify the resource for the current CUDA device, `get_current_device_resource()` and
 * `set_current_device_resource()` will automatically use the current CUDA device id from
 * `cudaGetDevice()`.
 *
 * Creating a device_memory_resource for each device requires care to set the current device
 * before creating each resource, and to maintain the lifetime of the resources as long as they
 * are set as per-device resources. Here is an example loop that creates `unique_ptr`s to
 * pool_memory_resource objects for each device and sets them as the per-device resource for that
 * device.
 *
 * @code{c++}
 * std::vector<unique_ptr<pool_memory_resource>> per_device_pools;
 * for(int i = 0; i < N; ++i) {
 *   cudaSetDevice(i);
 *   per_device_pools.push_back(std::make_unique<pool_memory_resource>());
 *   set_per_device_resource(cuda_device_id{i}, &per_device_pools.back());
 * }
 * @endcode
 */

namespace tamm::rmm::mr {

namespace detail {

/**
 * @brief Returns a pointer to the initial resource.
 *
 * Returns a global instance of a `gpu_memory_resource` as a function local static.
 *
 * @return Pointer to the static gpu_memory_resource used as the initial, default resource
 */
inline device_memory_resource* initial_resource() {
  static gpu_memory_resource mr{};
  return &mr;
}

// Must have default visibility, see: https://github.com/rapidsai/rmm/issues/826
RMM_EXPORT inline auto& get_map() {
  static std::map<int, device_memory_resource*> device_id_to_resource;
  return device_id_to_resource;
}

} // namespace detail

/**
 * @brief Get the resource for the specified device.
 *
 * Returns a pointer to the `device_memory_resource` for the specified device. The initial
 * resource is a `gpu_memory_resource`.
 *
 * `id.value()` must be in the range `[0, cudaGetDeviceCount())`, otherwise behavior is undefined.
 *
 * This function is thread-safe with respect to concurrent calls to `set_per_device_resource`,
 * `get_per_device_resource`, `get_current_device_resource`, and `set_current_device_resource`.
 * Concurrent calls to any of these functions will result in a valid state, but the order of
 * execution is undefined.
 *
 * @note The returned `device_memory_resource` should only be used when CUDA device `id` is the
 * current device  (e.g. set using `cudaSetDevice()`). The behavior of a device_memory_resource is
 * undefined if used while the active CUDA device is a different device from the one that was active
 * when the device_memory_resource was created.
 *
 * @param id The id of the target device
 * @return Pointer to the current `device_memory_resource` for device `id`
 */
inline device_memory_resource* get_per_device_resource(int device_id) {
  auto& map = detail::get_map();
  // If a resource was never set for `id`, set to the initial resource
  auto const found = map.find(device_id);
  return (found == map.end()) ? (map[device_id] = detail::initial_resource()) : found->second;
}
} // namespace tamm::rmm::mr
