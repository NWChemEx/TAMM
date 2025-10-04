#include <tamm/tamm.hpp>
#include <tamm/tamm_git.hpp>

using namespace tamm;
using namespace tamm::kernels;

template<typename T>
struct GEMMTestCase {
  int contraction_size;
  int output_a_size;
  int output_b_size;
  int total_output_size;
  int batch_size;
  int reduction_a_size;
  int reduction_b_size;

  // Derived GEMM dimensions
  int         M, N, K, B, AR, BR;
  std::string test_name;

  void compute_gemm_dimensions() {
    M  = output_a_size;
    N  = output_b_size;
    K  = contraction_size;
    B  = batch_size;
    AR = reduction_a_size;
    BR = reduction_b_size;

    // Create descriptive test name
    std::ostringstream oss;
    oss << "GEMM_" << get_type_name() << "_" << M << "x" << N << "x" << K << "_B" << B << "_AR"
        << AR << "_BR" << BR;
    test_name = oss.str();
  }

  // Check for potential overflow in size calculations
  bool validate_dimensions() const {
    // Check for negative or zero dimensions
    if(M <= 0 || N <= 0 || K <= 0 || B <= 0 || AR <= 0 || BR <= 0) { return false; }

    // Check for potential overflow in buffer size calculations
    const size_t max_size = std::numeric_limits<size_t>::max() / sizeof(T);

    // Check A buffer: AR * B * M * K
    if(static_cast<size_t>(AR) > max_size / B || static_cast<size_t>(AR) * B > max_size / M ||
       static_cast<size_t>(AR) * B * M > max_size / K) {
      return false;
    }

    // Check B buffer: BR * B * K * N
    if(static_cast<size_t>(BR) > max_size / B || static_cast<size_t>(BR) * B > max_size / K ||
       static_cast<size_t>(BR) * B * K > max_size / N) {
      return false;
    }

    // Check C buffer: B * M * N
    if(static_cast<size_t>(B) > max_size / M || static_cast<size_t>(B) * M > max_size / N) {
      return false;
    }

    return true;
  }

  size_t get_a_buffer_size() const {
    if(!validate_dimensions()) { throw std::runtime_error("Invalid dimensions for buffer A"); }
    return static_cast<size_t>(AR) * B * M * K;
  }

  size_t get_b_buffer_size() const {
    if(!validate_dimensions()) { throw std::runtime_error("Invalid dimensions for buffer B"); }
    return static_cast<size_t>(BR) * B * K * N;
  }

  size_t get_c_buffer_size() const {
    if(!validate_dimensions()) { throw std::runtime_error("Invalid dimensions for buffer C"); }
    return static_cast<size_t>(B) * M * N;
  }

  double calculate_flops() const {
    // Each GEMM operation: 2*M*N*K FLOPs
    // Total operations: AR * BR * B * (2*M*N*K)
    return static_cast<double>(AR) * BR * B * 2.0 * M * N * K;
  }

  double calculate_data_size_mb() const {
    try {
      size_t total_elements = get_a_buffer_size() + get_b_buffer_size() + get_c_buffer_size();
      return (total_elements * sizeof(T)) / (1024.0 * 1024.0);
    } catch(const std::exception&) { return 0.0; }
  }

  std::string get_type_name() const {
    if constexpr(std::is_same_v<T, float>) return "float";
    else if constexpr(std::is_same_v<T, double>) return "double";
    else if constexpr(std::is_same_v<T, int>) return "int";
    else if constexpr(std::is_same_v<T, long>) return "long";
    else return "unknown";
  }
};

// Utility functions
std::vector<std::string> split(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream        ss(str);
  std::string              token;
  while(std::getline(ss, token, delimiter)) {
    // Trim whitespace
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    if(!token.empty()) { // Only add non-empty tokens
      tokens.push_back(token);
    }
  }
  return tokens;
}

template<typename T>
std::vector<GEMMTestCase<T>> read_test_cases(const std::string& filename) {
  std::vector<GEMMTestCase<T>> test_cases;
  std::ifstream                file(filename);
  std::string                  line;

  if(!file.is_open()) { throw std::runtime_error("Could not open file: " + filename); }

  // Skip header line
  if(std::getline(file, line)) { std::cout << "Header: " << line << std::endl; }

  int line_number = 1; // Track line numbers for better error reporting
  while(std::getline(file, line)) {
    line_number++;
    if(line.empty()) continue;

    auto parts = split(line, ',');
    if(parts.size() != 7) {
      std::cerr << "Skipping invalid line " << line_number << " (expected 7 columns): " << line
                << std::endl;
      continue;
    }

    try {
      GEMMTestCase<T> test_case;

      // Parse with validation
      test_case.contraction_size  = std::stoi(parts[0]);
      test_case.output_a_size     = std::stoi(parts[1]);
      test_case.output_b_size     = std::stoi(parts[2]);
      test_case.total_output_size = std::stoi(parts[3]);
      test_case.batch_size        = std::stoi(parts[4]);
      test_case.reduction_a_size  = std::stoi(parts[5]);
      test_case.reduction_b_size  = std::stoi(parts[6]);

      test_case.compute_gemm_dimensions();

      // Validate dimensions and check for overflow
      if(test_case.validate_dimensions()) { test_cases.push_back(test_case); }
      else {
        std::cerr << "Skipping line " << line_number
                  << " due to invalid or overflow-prone dimensions: " << line << std::endl;
      }

    } catch(const std::exception& e) {
      std::cerr << "Error parsing line " << line_number << ": " << line << " - " << e.what()
                << std::endl;
    }
  }

  return test_cases;
}

template<typename T>
void generate_random_data(T* data, size_t size) {
  if(!data || size == 0) return;

  std::random_device rd;
  std::mt19937       gen(rd());

  if constexpr(std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dis(static_cast<T>(-1.0), static_cast<T>(1.0));
    for(size_t i = 0; i < size; ++i) { data[i] = dis(gen); }
  }
  else if constexpr(std::is_integral_v<T>) {
    // Handle integral types
    if constexpr(sizeof(T) == 1) {
      std::uniform_int_distribution<int> dis(-100, 100);
      for(size_t i = 0; i < size; ++i) { data[i] = static_cast<T>(dis(gen)); }
    }
    else {
      std::uniform_int_distribution<T> dis(-100, 100);
      for(size_t i = 0; i < size; ++i) { data[i] = dis(gen); }
    }
  }
  else {
    // Fallback for unknown types
    std::fill(data, data + size, T{});
  }
}

template<typename T>
void run_gemm_benchmark(const GEMMTestCase<T>& test_case, ExecutionHW hw, gpuStream_t& thandle,
                        int num_iterations = 100) {
  std::cout << "Testing: " << test_case.test_name << std::endl;
  std::cout << "  Matrix dimensions: A(" << test_case.M << "x" << test_case.K << ") × B("
            << test_case.K << "x" << test_case.N << ") = C(" << test_case.M << "x" << test_case.N
            << ")" << std::endl;
  std::cout << "  Batch size: " << test_case.B << std::endl;
  std::cout << "  Reduction dimensions: AR=" << test_case.AR << ", BR=" << test_case.BR
            << std::endl;
  std::cout << "  Data type: " << test_case.get_type_name() << " (" << sizeof(T)
            << " bytes per element)" << std::endl;

  try {
    // Validate dimensions before proceeding
    if(!test_case.validate_dimensions()) {
      throw std::runtime_error("Invalid test case dimensions");
    }

    size_t asize = test_case.get_a_buffer_size();
    size_t bsize = test_case.get_b_buffer_size();
    size_t csize = test_case.get_c_buffer_size();

    std::cout << "  Buffer sizes: A=" << asize << ", B=" << bsize << ", C=" << csize << std::endl;

    std::cout << "  Allocating buffers..." << std::endl;

    // Allocate host buffers using TAMM framework
    T* ainter_buf = new T[asize];
    T* binter_buf = new T[bsize];
    T* cinter_buf = new T[csize];

    allocate_host_buffers(hw, ainter_buf, asize);
    allocate_host_buffers(hw, binter_buf, bsize);
    allocate_host_buffers(hw, cinter_buf, csize);

    // Generate random test data
    std::cout << "  Generating test data..." << std::endl;
    generate_random_data(ainter_buf, asize);
    generate_random_data(binter_buf, bsize);
    generate_random_data(cinter_buf, csize);

    // Device buffer pointers (allocated by framework)
    T* ainter_buf_dev = nullptr;
    T* binter_buf_dev = nullptr;
    T* cinter_buf_dev = nullptr;

    std::cout << "  Allocating and copying A buffer to GPU..." << std::endl;
    allocate_device_buffers(hw, ainter_buf_dev, asize);
    std::cout << "  Allocating and copying B buffer to GPU..." << std::endl;
    allocate_device_buffers(hw, binter_buf_dev, bsize);

    // Device buffer cleanup helper
    auto cleanup_device_buffers = [&]() {
      try {
        if(ainter_buf_dev) {
          free_device_buffers(hw, ainter_buf_dev, asize);
          ainter_buf_dev = nullptr;
        }
        if(binter_buf_dev) {
          free_device_buffers(hw, binter_buf_dev, bsize);
          binter_buf_dev = nullptr;
        }
        if(cinter_buf_dev) {
          free_device_buffers(hw, cinter_buf_dev, csize);
          cinter_buf_dev = nullptr;
        }
      } catch(const std::exception& e) {
        std::cerr << "Warning: Failed to cleanup device buffers: " << e.what() << std::endl;
      }
    };

    // Copy data to GPU using framework function
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    if(hw == ExecutionHW::GPU) {
      try {
        std::cout << "  Copying data to GPU..." << std::endl;
        std::cout << "    Allocating and copying A and B buffers to GPU..." << std::endl;
        copy_data_to_gpu(hw, thandle, ainter_buf, asize, ainter_buf_dev, binter_buf, bsize,
                         binter_buf_dev);

        std::cout << "    Allocating and copying C buffer to GPU..." << std::endl;
        allocate_device_buffers(hw, cinter_buf_dev, csize);
        copy_data_to_gpu(hw, thandle, cinter_buf, csize, cinter_buf_dev);
        std::cout << "    Data copy to GPU completed." << std::endl;
      } catch(const std::exception& e) {
        cleanup_device_buffers();
        throw std::runtime_error("Failed to copy data to GPU: " + std::string(e.what()));
      }
    }
#endif

    // GEMM parameters
    T alpha = static_cast<T>(1.0);
    T beta  = static_cast<T>(0.0);

    std::cout << "  Running warmup iterations..." << std::endl;

    try {
      // Warmup runs
      for(int i = 0; i < 5; ++i) {
        gemm_wrapper<T, T, T, T>(hw, thandle, test_case.AR, test_case.BR, test_case.B, test_case.M,
                                 test_case.N, test_case.K, alpha, beta, ainter_buf, ainter_buf_dev,
                                 binter_buf, binter_buf_dev, cinter_buf, cinter_buf_dev);
      }

      // Synchronize before timing
      if(hw == ExecutionHW::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        gpuStreamSynchronize(thandle);
#endif
      }

      std::cout << "  Running " << num_iterations << " timing iterations..." << std::endl;

      // Benchmark runs
      auto start_time = std::chrono::high_resolution_clock::now();

      for(int i = 0; i < num_iterations; ++i) {
        gemm_wrapper<T, T, T, T>(hw, thandle, test_case.AR, test_case.BR, test_case.B, test_case.M,
                                 test_case.N, test_case.K, alpha, beta, ainter_buf, ainter_buf_dev,
                                 binter_buf, binter_buf_dev, cinter_buf, cinter_buf_dev);
      }

      // Synchronize after timing
      if(hw == ExecutionHW::GPU) {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        gpuStreamSynchronize(thandle);
#endif
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

      // Calculate performance metrics
      double avg_time_us = static_cast<double>(duration.count()) / num_iterations;
      double avg_time_ms = avg_time_us / 1000.0;

      double total_flops = test_case.calculate_flops();
      double gflops      = 0.0;
      if(avg_time_us > 0.0) {
        gflops = total_flops / (avg_time_us * 1000.0); // GFLOPS
      }

      double data_size_mb    = test_case.calculate_data_size_mb();
      double throughput_gb_s = 0.0;
      if(avg_time_ms > 0.0) {
        throughput_gb_s = (data_size_mb * 2) / (avg_time_ms / 1000.0) / 1024.0; // Read + Write
      }

      // Output results
      std::cout << std::fixed << std::setprecision(6);
      std::cout << "  Total FLOPs: " << std::scientific << std::setprecision(3) << total_flops
                << std::endl;
      std::cout << "  Data size: " << std::fixed << std::setprecision(3) << data_size_mb << " MB"
                << std::endl;
      std::cout << "  Iterations: " << num_iterations << std::endl;
      std::cout << "  Average time: " << std::setprecision(6) << avg_time_ms << " ms" << std::endl;
      std::cout << "  Performance: " << std::setprecision(3) << gflops << " GFLOPS" << std::endl;
      std::cout << "  Throughput: " << std::setprecision(3) << throughput_gb_s << " GB/s"
                << std::endl;
      std::cout << "  Hardware: " << (hw == ExecutionHW::GPU ? "GPU" : "CPU") << std::endl;
      std::cout << "  Status: SUCCESS" << std::endl;

    } catch(const std::exception& e) {
      std::cout << "  Status: FAILED during GEMM execution (" << e.what() << ")" << std::endl;
    }

    // Cleanup device buffers
    cleanup_device_buffers();

  } catch(const std::exception& e) {
    std::cout << "  Status: FAILED (" << e.what() << ")" << std::endl;
  }

  std::cout << "----------------------------------------" << std::endl;
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  try {
    tamm::initialize(argc, argv);

    if(argc < 2 || argc > 3) {
      std::cerr << "Usage: mpirun -n 2 " << argv[0] << " <csv_file> [num_iterations]" << std::endl;
      std::cerr << std::endl;
      std::cerr << "CSV file format:" << std::endl;
      std::cerr << "contraction_size,output_a_size,output_b_size,total_output_size,batch_size,"
                   "reduction_a_size,reduction_b_size"
                << std::endl;
      std::cerr << "256,128,128,16384,32,4,4" << std::endl;
      std::cerr << "512,256,256,65536,16,2,2" << std::endl;
      tamm::finalize();
      return 1;
    }

    std::string filename       = argv[1];
    int         num_iterations = 100; // default

    if(argc == 3) {
      try {
        num_iterations = std::stoi(argv[2]);
        if(num_iterations <= 0) {
          throw std::runtime_error("Number of iterations must be positive");
        }
      } catch(const std::exception& e) {
        std::cerr << "Invalid number of iterations: " << e.what() << std::endl;
        tamm::finalize();
        return 1;
      }
    }

    // Read test cases
    auto test_cases = read_test_cases<double>(filename);

    if(test_cases.empty()) {
      std::cerr << "No valid test cases found in file: " << filename << std::endl;
      tamm::finalize();
      return 1;
    }

    std::cout << "Loaded " << test_cases.size() << " GEMM test cases (double precision)"
              << std::endl
              << std::endl;

    // Initialize GPU stream (if using GPU)
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    gpuStream_t thandle = tamm::GPUStreamPool::getInstance().getStream();
#else
    gpuStream_t thandle{};
#endif

    // Run tests
    std::cout << "=== GPU GEMM TESTS (DOUBLE PRECISION) ===" << std::endl;
    for(const auto& test_case: test_cases) {
      run_gemm_benchmark(test_case, ExecutionHW::GPU, thandle, num_iterations);
    }
    std::cout << std::endl;

    // Print summary
    std::cout << "=== BENCHMARK SUMMARY ===" << std::endl;
    std::cout << "Completed " << test_cases.size() << " test cases" << std::endl;
    std::cout << "Data type: double precision (8 bytes)" << std::endl;
    std::cout << "Iterations per test: " << num_iterations << std::endl;
    std::cout << "Hardware tested: GPU" << std::endl;

    tamm::finalize();

  } catch(const std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    try {
      tamm::finalize();
    } catch(...) { std::cerr << "Failed to finalize TAMM" << std::endl; }
    return 1;
  }

  return 0;
}