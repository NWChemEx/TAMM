#pragma once

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

#include <sycl/sycl.hpp>

class device_ext: public sycl::device {
public:
  device_ext(): sycl::device(), _ctx(*this) {}
  ~device_ext() { std::lock_guard<std::mutex> lock(m_mutex); }
  device_ext(const sycl::device& base): sycl::device(base), _ctx(*this) {}

private:
  sycl::context      _ctx;
  mutable std::mutex m_mutex;
};

static inline int get_tid() { return syscall(SYS_gettid); }

class dev_mgr {
public:
  int current_device() {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto                        it = _thread2dev_map.find(get_tid());
    if(it != _thread2dev_map.end()) {
      check_id(it->second);
      return it->second;
    }
    printf("WARNING: no SYCL device found in the map, returning DEFAULT_DEVICE_ID\n");
    return DEFAULT_DEVICE_ID;
  }
  device_ext* get_sycl_device(int id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    return (_devs[id].first).get();
  }
  sycl::context* get_sycl_context(int id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    return (_devs[id].second).get();
  }

  void select_device(int id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    _thread2dev_map[get_tid()] = id;
  }
  int device_count() { return _devs.size(); }

  /// Returns the instance of device manager singleton.
  static dev_mgr& instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr&)            = delete;
  dev_mgr& operator=(const dev_mgr&) = delete;
  dev_mgr(dev_mgr&&)                 = delete;
  dev_mgr& operator=(dev_mgr&&)      = delete;

private:
  mutable std::mutex m_mutex;

  dev_mgr() {
    std::vector<sycl::device> sycl_all_devs =
      sycl::device::get_devices(sycl::info::device_type::gpu);
    for(auto& dev: sycl_all_devs) {
      if(dev.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
        auto subDevicesDomainNuma =
          dev.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
            sycl::info::partition_affinity_domain::numa);
        for(auto& tile: subDevicesDomainNuma) {
          _devs.push_back(std::make_pair(std::make_shared<device_ext>(tile),
                                         std::make_shared<sycl::context>(tile)));
        }
      }
      else {
        _devs.push_back(
          std::make_pair(std::make_shared<device_ext>(dev), std::make_shared<sycl::context>(dev)));
      }
    }
  }

  void check_id(int id) const {
    if(id >= _devs.size()) { throw std::runtime_error("invalid device id"); }
  }

  std::vector<std::pair<std::shared_ptr<device_ext>, std::shared_ptr<sycl::context>>> _devs;
  /// DEFAULT_DEVICE_ID is used, if current_device() can not find current
  /// thread id in _thread2dev_map, which means default device should be used
  /// for the current thread.
  const int DEFAULT_DEVICE_ID = 0;
  /// thread-id to device-id map.
  std::map<int, int> _thread2dev_map;
};

/// Util function to get the current device (in int).
static inline void syclGetDevice(int* id) { *id = dev_mgr::instance().current_device(); }

/// Util function to get the current sycl::device by id.
static inline device_ext* sycl_get_device(int id) {
  return dev_mgr::instance().get_sycl_device(id);
}
/// Util function to get the current sycl::context by id.
static inline sycl::context* sycl_get_context(int id) {
  return dev_mgr::instance().get_sycl_context(id);
}

/// Util function to set a device by id. (to _thread2dev_map)
static inline void syclSetDevice(int id) { dev_mgr::instance().select_device(id); }

/// Util function to get number of GPU devices (default: explicit scaling)
static inline void syclGetDeviceCount(int* id) { *id = dev_mgr::instance().device_count(); }
