#include <cuda_runtime.h>

#include <cstddef>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <new>

namespace tamm {

  class GPUPooledStorageManager {
  protected:
    std::mutex mutex_;
    // used memory
    size_t used_memory_ = 0;
    // percentage of reserved memory
    int reserve_;
    // memory pool
    std::unordered_map<size_t, std::vector<void*>> memory_pool_;

  private:
    GPUPooledStorageManager() {
      reserve_ = 5;
    }
    ~GPUPooledStorageManager() {
      for (auto&& i : memory_pool_) {
	for (auto&& j : i.second) {
	  cudaError_t err = cudaFree(j);
	  used_memory_ -= i.first;
	}
      }
      memory_pool_.clear();
    }

  public:
    void* allocate(size_t size) {
      std::lock_guard<std::mutex> lock(mutex_);

      auto&& reuse_it = memory_pool_.find(size);
      if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	if (size > free - total*reserve_/100) ReleaseAll();

	void* ret = nullptr;
	cudaError_t e = cudaMalloc(&ret, size);
	used_memory_ += size;
	return ret;
      } else {
	auto&& reuse_pool = reuse_it->second;
	auto ret = reuse_pool.back();
	reuse_pool.pop_back();
	return ret;
      }      
    }
    void deallocate(void* ptr, size_t size) {
      std::lock_guard<std::mutex> lock(mutex_);

      auto&& reuse_pool = memory_pool_[size];
      reuse_pool.push_back(ptr);
    }

    void GPUPooledStorageManager::ReleaseAll() {
      for (auto&& i : memory_pool_) {
	for (auto&& j : i.second) {
	  DirectFree(j, i.first);
	}
      }
      memory_pool_.clear();
    }

    /// Returns the instance of device manager singleton.
    inline static GPUPooledStorageManager& getInstance() {
      static GPUPooledStorageManager d_m{};
      return d_m;
    }

    GPUPooledStorageManager(const GPUPooledStorageManager&)            = delete;
    GPUPooledStorageManager& operator=(const GPUPooledStorageManager&) = delete;
    GPUPooledStorageManager(GPUPooledStorageManager&&)                 = delete;
    GPUPooledStorageManager& operator=(GPUPooledStorageManager&&)      = delete;
    
  };  // class GPUPooledStorageManager

}  // namespace tamm
