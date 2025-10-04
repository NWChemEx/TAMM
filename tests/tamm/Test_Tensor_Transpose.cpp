#include <tamm/tamm.hpp>
#include <tamm/tamm_git.hpp>

using namespace tamm;
using namespace tamm::kernels;

struct TransposeTestCase {
  SizeVec     input_dims;
  IntLabelVec input_labels;
  IntLabelVec output_labels;
  std::string transpose_string;

  size_t total_elements() const {
    size_t total = 1;
    for(auto dim: input_dims) { total *= dim.value(); }
    return total;
  }

  SizeVec compute_output_dims() const {
    if(input_dims.empty() || output_labels.empty()) { return SizeVec{}; }

    // Validate output_labels indices
    auto max_label = *std::max_element(output_labels.begin(), output_labels.end());
    if(max_label >= static_cast<int>(input_dims.size())) {
      throw std::runtime_error("Invalid output label index: " + std::to_string(max_label));
    }

    SizeVec output_dims(input_dims.size());
    for(size_t i = 0; i < input_labels.size(); ++i) {
      if(i >= output_labels.size()) break; // Safety check
      int output_pos = output_labels[i];
      if(output_pos >= 0 && output_pos < static_cast<int>(output_dims.size())) {
        output_dims[output_pos] = input_dims[i];
      }
    }
    return output_dims;
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
    if(!token.empty()) { // Added: only add non-empty tokens
      tokens.push_back(token);
    }
  }
  return tokens;
}

std::vector<TransposeTestCase> read_test_cases(const std::string& filename) {
  std::vector<TransposeTestCase> test_cases;
  std::ifstream                  file(filename);
  std::string                    line;

  if(!file.is_open()) { throw std::runtime_error("Could not open file: " + filename); }

  // Skip header line
  if(std::getline(file, line)) { std::cout << "Header: " << line << std::endl; }

  while(std::getline(file, line)) {
    if(line.empty()) continue;

    auto parts = split(line, ',');
    if(parts.size() != 9) {
      std::cerr << "Skipping invalid line (expected 9 columns): " << line << std::endl;
      continue;
    }

    TransposeTestCase test_case;

    // Parse block sizes (dimensions)
    SizeVec     block_sizes;
    IntLabelVec permutation_map;

    try {
      for(int i = 0; i < 4; ++i) {
        size_t block_size = std::stoull(parts[i]);
        int    perm_idx   = std::stoi(parts[i + 4]);

        if(block_size > 1 && perm_idx >= 0) { // Valid dimension
          block_sizes.push_back(block_size);
          permutation_map.push_back(perm_idx);
        }
      }
    } catch(const std::exception& e) {
      std::cerr << "Skipping line due to parsing error: " << line << " (" << e.what() << ")"
                << std::endl;
      continue;
    }

    test_case.input_dims       = block_sizes;
    test_case.transpose_string = parts[8];

    // Create input labels (sequential: 0, 1, 2, ...)
    test_case.input_labels.resize(block_sizes.size());
    std::iota(test_case.input_labels.begin(), test_case.input_labels.end(), 0);

    // Create output labels based on permutation map
    test_case.output_labels = permutation_map;

    // Validate that we have a valid test case
    if(!test_case.input_dims.empty() &&
       test_case.input_dims.size() == test_case.output_labels.size()) {
      // Validate the permutation map
      try {
        test_case.compute_output_dims(); // This will throw if invalid
        test_cases.push_back(test_case);
      } catch(const std::exception& e) {
        std::cerr << "Skipping invalid test case: " << test_case.transpose_string << " ("
                  << e.what() << ")" << std::endl;
      }
    }
    else { std::cerr << "Skipping invalid test case: " << test_case.transpose_string << std::endl; }
  }

  return test_cases;
}

template<typename T>
void generate_random_data(T* data, size_t size) {
  std::random_device rd;
  std::mt19937       gen(rd());

  if constexpr(std::is_integral_v<T>) {
    // Handle different integral types appropriately
    if constexpr(sizeof(T) == 1) {
      std::uniform_int_distribution<int> dis(0, 100);
      for(size_t i = 0; i < size; ++i) { data[i] = static_cast<T>(dis(gen)); }
    }
    else {
      std::uniform_int_distribution<T> dis(0, 100);
      for(size_t i = 0; i < size; ++i) { data[i] = dis(gen); }
    }
  }
  else {
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for(size_t i = 0; i < size; ++i) { data[i] = dis(gen); }
  }
}

template<typename T>
void run_transpose_benchmark(const TransposeTestCase& test_case, ExecutionHW hw,
                             gpuStream_t& thandle, int num_iterations = 100) {
  size_t total_elements = test_case.total_elements();

  if(total_elements == 0) {
    std::cerr << "Skipping test case with zero elements" << std::endl;
    return;
  }

  SizeVec output_dims;
  try {
    output_dims = test_case.compute_output_dims();
  } catch(const std::exception& e) {
    std::cerr << "Skipping test case due to invalid dimensions: " << e.what() << std::endl;
    return;
  }

  double data_size_mb = (total_elements * sizeof(T)) / (1024.0 * 1024.0);

  // Use smart pointers for automatic cleanup
  T* input_buf      = new T[total_elements];
  T* output_buf     = new T[total_elements];
  T* output_buf_dev = nullptr;

  // Generate random test data
  generate_random_data(input_buf, total_elements);

  std::cout << "Testing (" << typeid(T).name() << "): " << test_case.transpose_string << std::endl;
  std::cout << "  Dimensions: [";
  for(size_t i = 0; i < test_case.input_dims.size(); ++i) {
    if(i > 0) std::cout << ",";
    std::cout << test_case.input_dims[i];
  }
  std::cout << "] -> [";
  for(size_t i = 0; i < output_dims.size(); ++i) {
    if(i > 0) std::cout << ",";
    std::cout << output_dims[i];
  }
  std::cout << "]" << std::endl;

  std::cout << "  Permutation: [";
  for(size_t i = 0; i < test_case.input_labels.size(); ++i) {
    if(i > 0) std::cout << ",";
    std::cout << test_case.input_labels[i];
  }
  std::cout << "] -> [";
  for(size_t i = 0; i < test_case.output_labels.size(); ++i) {
    if(i > 0) std::cout << ",";
    std::cout << test_case.output_labels[i];
  }
  std::cout << "]" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  Elements: " << total_elements << std::endl;
  std::cout << "  Data size: " << data_size_mb << " MB" << std::endl;
  std::cout << "  Element size: " << sizeof(T) << " bytes" << std::endl;

  try {
    // Warmup runs
    for(int i = 0; i < 5; ++i) {
      transpose_tensor(hw, thandle, output_buf, output_dims, test_case.output_labels, input_buf,
                       total_elements, test_case.input_dims, test_case.input_labels,
                       output_buf_dev);
    }

    // Benchmark runs
    auto start_time = std::chrono::high_resolution_clock::now();

    bool used_gpu = false;
    for(int i = 0; i < num_iterations; ++i) {
      used_gpu = transpose_tensor(hw, thandle, output_buf, output_dims, test_case.output_labels,
                                  input_buf, total_elements, test_case.input_dims,
                                  test_case.input_labels, output_buf_dev);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    double avg_time_us = static_cast<double>(duration.count()) / num_iterations;
    double avg_time_ms = avg_time_us / 1000.0;

    // Calculate throughput (avoid division by zero)
    double throughput_gb_s = 0.0;
    if(avg_time_ms > 0.0) {
      throughput_gb_s = (data_size_mb * 2) / (avg_time_ms / 1000.0) / 1024.0; // Read + Write
    }

    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << "  Average time: " << std::setprecision(6) << avg_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(3) << throughput_gb_s << " GB/s"
              << std::endl;
    std::cout << "  Hardware: " << (hw == ExecutionHW::GPU ? "GPU" : "CPU") << std::endl;
    std::cout << std::endl;

  } catch(const std::exception& e) {
    std::cerr << "Error during transpose operation: " << e.what() << std::endl;
  }

  // GPU buffer cleanup (if the function exists)
  if(output_buf_dev) {
    try {
      free_device_buffers(hw, output_buf_dev, total_elements);
    } catch(...) { std::cerr << "Warning: Failed to free GPU buffer" << std::endl; }
  }
}

template<typename T>
void run_all_tests(const std::vector<TransposeTestCase>& test_cases, int num_iterations) {
  // Initialize GPU stream (if using GPU)
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  gpuStream_t thandle = tamm::GPUStreamPool::getInstance().getStream();
#else
  gpuStream_t thandle{};
#endif

  std::cout << "\n========== TESTING WITH TYPE: " << typeid(T).name() << " ==========" << std::endl;
  std::cout << "=== GPU TESTS ===" << std::endl;

  for(const auto& test_case: test_cases) {
    run_transpose_benchmark<T>(test_case, ExecutionHW::GPU, thandle, num_iterations);
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  try {
    tamm::initialize(argc, argv);

    if(argc < 2 || argc > 4) {
      std::cerr << "Usage: mpirun -n 2 " << argv[0] << " <csv_file> [num_iterations] [data_type]"
                << std::endl;
      std::cerr << "  data_type options: float, double, all (default: double)" << std::endl;
      return 1;
    }

    int         num_iterations = 100;      // default
    std::string data_type      = "double"; // default

    if(argc >= 3) {
      num_iterations = std::stoi(argv[2]);
      if(num_iterations <= 0) {
        std::cerr << "Number of iterations must be positive" << std::endl;
        return 1;
      }
    }
    if(argc == 4) { data_type = argv[3]; }

    // Read test cases
    auto test_cases = read_test_cases(argv[1]);
    if(test_cases.empty()) {
      std::cerr << "No valid test cases found in file: " << argv[1] << std::endl;
      return 1;
    }

    std::cout << "Loaded " << test_cases.size() << " test cases" << std::endl;
    std::cout << "Iterations per test: " << num_iterations << std::endl;
    std::cout << "Data type(s): " << data_type << std::endl;

    // Run tests based on specified data type
    if(data_type == "float") { run_all_tests<float>(test_cases, num_iterations); }
    else if(data_type == "double") { run_all_tests<double>(test_cases, num_iterations); }
    else if(data_type == "all") {
      run_all_tests<float>(test_cases, num_iterations);
      run_all_tests<double>(test_cases, num_iterations);
    }
    else {
      std::cerr << "Unknown data type: " << data_type << std::endl;
      std::cerr << "Supported types: float, double, all" << std::endl;
      return 1;
    }

    tamm::finalize();

  } catch(const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    try {
      tamm::finalize();
    } catch(...) {}
    return 1;
  }

  return 0;
}