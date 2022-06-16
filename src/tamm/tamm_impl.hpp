#pragma once

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
// headers related to device/stream pool
#include "tamm/gpu_streams.hpp"

// headers related to memory pool
#include "tamm/mr/device_memory_resource.hpp"
#include "tamm/mr/gpu_memory_resource.hpp"
#include "tamm/mr/per_device_resource.hpp"
#include "tamm/mr/pool_memory_resource.hpp"
#endif

namespace tamm {

// tamm as singleton object
class TAMM {
protected:
  using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  std::unique_ptr<pool_mr> mr_;

private:
  TAMM() {
#if !defined(USE_TALSH) && (defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
    int ngpus_{};
    tamm::getDeviceCount(&ngpus_);
    std::cout << "number of ngpu_ from execution_context.cpp: " << ngpus_ << std::endl;

    // When TALSH is not defined, only 1 MPI-rank should see 1 GPU device
    if(ngpus_ >= 1 || ngpus_ == 0) {
      std::string terminate_msg =
        "[NO-TALSH] Multiple/No GPUs detected per MPI rank, ngpus_(" + std::to_string(ngpus_) +
        ")!. Please consider using `tamm_gpu_bind.sh` script or job manager to bind...";

      // // NOTE: tamm_terminate needs to be called only after tamm_initialize()
      // tamm_terminate(terminate_msg);
    }

    // GPU Stream Pool as singleton object
    auto&        pool                  = tamm::GPUStreamPool::getInstance();
    auto&        dev_stream            = pool.getStream();
    unsigned int default_gpu_device_id = 0; // per MPI-rank
    pool.set_device(default_gpu_device_id);

    mr_ = std::make_unique<pool_mr>(rmm::mr::get_current_device_resource(), dev_stream);
#endif // !defined(USE_TALSH)
  }

public:
  pool_mr* get_memory_pool() { return mr_.get(); }

  /// Returns the instance of TAMM singleton.
  inline static TAMM& getInstance() {
    static TAMM m_tamm_{};
    return m_tamm_;
  }

  TAMM(const TAMM&)            = delete;
  TAMM& operator=(const TAMM&) = delete;
  TAMM(TAMM&&)                 = delete;
  TAMM& operator=(TAMM&&)      = delete;
};

} // namespace tamm
