#include "ga/ga.h"
#include <mpi.h>

#include "distribution.hpp"
#include "execution_context.hpp"
#include "labeled_tensor.hpp"
#include "memory_manager.hpp"
#include "proc_group.hpp"
#include "rmm_memory_pool.hpp"
#include "runtime_engine.hpp"

#include <nlohmann/json.hpp>

#include <ctime>

namespace tamm {
ExecutionContext::ExecutionContext(ProcGroup pg, DistributionKind default_dist_kind,
                                   MemoryManagerKind default_memory_manager_kind,
                                   RuntimeEngine*    re):
  pg_{pg},
  distribution_kind_{default_dist_kind},
  memory_manager_kind_{default_memory_manager_kind},
  ac_{IndexedAC{nullptr, 0}} {
  if(re == nullptr) { re_.reset(runtime_ptr()); }
  else {
    re_.reset(re, [](auto) {});
  }

#if defined(USE_UPCXX)
  pg_self_ = ProcGroup{team_self};

#else
  pg_self_ = ProcGroup{MPI_COMM_SELF, ProcGroup::self_ga_pgroup()};
#endif

#if defined(USE_UPCXX)
  ranks_pn_ = upcxx::local_team().rank_n();
#else
  ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_proc_nodeid(pg.rank().value()));
#endif
  nnodes_ = pg.size().value() / ranks_pn_;

#if defined(__APPLE__)
  {
    size_t size_mpn = sizeof(minfo_.cpu_mem_per_node);
    sysctlbyname("hw.memsize", &(minfo_.cpu_mem_per_node), &size_mpn, nullptr, 0);
  }
#else
  {
    struct sysinfo cpumeminfo_;
    sysinfo(&cpumeminfo_);
    minfo_.cpu_mem_per_node = cpumeminfo_.totalram * cpumeminfo_.mem_unit;
  }
#endif
  minfo_.cpu_name = getHostName();
  minfo_.cpu_mem_per_node /= (1024 * 1024 * 1024.0); // GiB
  minfo_.total_cpu_mem = minfo_.cpu_mem_per_node * nnodes_;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  has_gpu_ = true;
  exhw_    = ExecutionHW::GPU;
  gpus_pn_ = ranks_pn_ / ranks_per_gpu_pool();
  {
    size_t free_{};
    minfo_.gpu_name = getDeviceName() + ", " + getRuntimeVersion();
    gpuMemGetInfo(&free_, &minfo_.gpu_mem_per_device);
    minfo_.gpu_mem_per_device /= (1024 * 1024 * 1024.0); // GiB
    minfo_.gpu_mem_per_node = minfo_.gpu_mem_per_device * gpus_pn_;
    minfo_.total_gpu_mem    = minfo_.gpu_mem_per_device * nnodes_ * gpus_pn_;
  }
#endif
}

ExecutionContext::ExecutionContext(ProcGroup pg, Distribution* default_distribution,
                                   MemoryManager* default_memory_manager, RuntimeEngine* re):
  ExecutionContext{
    pg, default_distribution != nullptr ? default_distribution->kind() : DistributionKind::invalid,
    default_memory_manager != nullptr ? default_memory_manager->kind() : MemoryManagerKind::invalid,
    re} {}

void ExecutionContext::set_distribution(Distribution* distribution) {
  if(distribution) { distribution_kind_ = distribution->kind(); }
  else { distribution_kind_ = DistributionKind::invalid; }
}

void ExecutionContext::set_re(RuntimeEngine* re) { re_.reset(re); }

namespace {
// The CPU model without the padding some CPUs report.
std::string trim_trailing_spaces(std::string s) {
  s.erase(s.find_last_not_of(" \t") + 1);
  return s;
}

// Get current local time in ISO 8601 with UTC offset
std::string local_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm           local{};
  localtime_r(&now, &local);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &local);
  return buf;
}

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
// The GPU runtime ("CUDA", "ROCm" or "SYCL") and its version, from getRuntimeVersion(), which
// returns
// "<CUDA|ROCM|SYCL> v<version>". For DPC++ the version is the device driver's.
std::pair<std::string, std::string> gpu_runtime() {
#if defined(USE_CUDA)
  const std::string name = "CUDA";
#elif defined(USE_HIP)
  const std::string name = "ROCm";
#else
  const std::string name = "SYCL";
#endif
  std::string version = getRuntimeVersion();
  if(const auto sep = version.find(" v"); sep != std::string::npos) version.erase(0, sep + 2);
  return {name, version};
}
#endif
} // namespace

void ExecutionContext::print_execution_environment() const {
  if(pg_.rank() != 0) return;

  // "<label>:" padded so values start at column 26, as in tamm_build_config().
  auto print_field = [](const std::string& label, const auto& value) {
    std::string padded = label + ":";
    if(padded.size() < 26) padded.append(26 - padded.size(), ' ');
    std::cout << padded << value << std::endl;
  };

  std::cout << "Execution environment" << std::endl;
  std::cout << "---------------------" << std::endl;
  print_field("Date", local_timestamp());
  print_field("Nodes", nnodes_);
  std::cout << "MPI ranks:" << std::endl;
  print_field("  Per node", ranks_pn_);
  print_field("  Total", nnodes_ * ranks_pn_);

  std::cout << std::endl << "CPU:" << std::endl;
  print_field("  Model", trim_trailing_spaces(minfo_.cpu_name));
  print_field("  Memory per node", std::to_string(minfo_.cpu_mem_per_node) + " GiB");
  print_field("  Memory total", std::to_string(minfo_.total_cpu_mem) + " GiB");

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  const auto [runtime, version] = gpu_runtime();
  std::cout << std::endl << "GPU:" << std::endl;
  print_field("  Model", getDeviceName());
  print_field("  " + runtime + (runtime == "SYCL" ? " driver" : " runtime"), version);
  print_field("  Devices per node", gpus_pn_);
  print_field("  Devices total", nnodes_ * gpus_pn_);
  print_field("  Memory per device", std::to_string(minfo_.gpu_mem_per_device) + " GiB");
  print_field("  Memory per node", std::to_string(minfo_.gpu_mem_per_node) + " GiB");
  print_field("  Memory total", std::to_string(minfo_.total_gpu_mem) + " GiB");
#endif
}

std::string ExecutionContext::execution_environment_json() const {
  nlohmann::ordered_json env;
  env["date"]                       = local_timestamp();
  env["nodes"]                      = nnodes_;
  env["mpi_ranks"]["per_node"]      = ranks_pn_;
  env["mpi_ranks"]["total"]         = nnodes_ * ranks_pn_;
  env["cpu"]["model"]               = trim_trailing_spaces(minfo_.cpu_name);
  env["cpu"]["memory_per_node_gib"] = minfo_.cpu_mem_per_node;
  env["cpu"]["memory_total_gib"]    = minfo_.total_cpu_mem;
  env["gpu"]                        = nullptr;
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  const auto [runtime, version]       = gpu_runtime();
  env["gpu"]["model"]                 = getDeviceName();
  env["gpu"]["runtime"]               = runtime + " " + version;
  env["gpu"]["devices_per_node"]      = gpus_pn_;
  env["gpu"]["devices_total"]         = nnodes_ * gpus_pn_;
  env["gpu"]["memory_per_device_gib"] = minfo_.gpu_mem_per_device;
  env["gpu"]["memory_per_node_gib"]   = minfo_.gpu_mem_per_node;
  env["gpu"]["memory_total_gib"]      = minfo_.total_gpu_mem;
#endif
  return env.dump();
}

} // namespace tamm
