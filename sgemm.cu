#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <tuple>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  // Usage:
  // ./sgemm <kernel>
  //     -> runs the original square-size sweep
  //
  // ./sgemm <kernel> <M> <N> <K>
  //     -> runs one arbitrary GEMM:
  //        A[M x K] * B[K x N] = C[M x N]

  if (argc != 2 && argc != 5) {
    std::cerr << "Usage:\n"
              << "  ./sgemm <kernel>\n"
              << "  ./sgemm <kernel> <M> <N> <K>\n"
              << "Kernel range: 0-12 (0 = NVIDIA cuBLAS)\n";
    exit(EXIT_FAILURE);
  }

  // Get kernel number
  int kernel_num = std::stoi(argv[1]);

  if (kernel_num < 0 || kernel_num > 11) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // ------------------------------------------------------------
  // Build list of test matrix dimensions
  // ------------------------------------------------------------

  std::vector<std::tuple<int, int, int>> TESTS;

  if (argc == 5) {
    int M = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);
    int K = std::stoi(argv[4]);

    if (M <= 0 || N <= 0 || K <= 0) {
      std::cerr << "M, N and K must all be positive integers." << std::endl;
      exit(EXIT_FAILURE);
    }

    TESTS.push_back({M, N, K});

  } else {
    // Original square benchmark sweep
    std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

    for (int size : SIZE) {
      TESTS.push_back({size, size, size});
    }
  }

  // ------------------------------------------------------------
  // Select GPU
  // ------------------------------------------------------------

  int deviceIdx = 0;

  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }

  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // Uncomment if you want detailed GPU information
  // CudaDeviceInfo();

  // ------------------------------------------------------------
  // Create cuBLAS handle
  // ------------------------------------------------------------

  cublasHandle_t handle;

  if (cublasCreate(&handle)) {
    std::cerr << "Create cuBLAS handle error." << std::endl;
    exit(EXIT_FAILURE);
  }

  // ------------------------------------------------------------
  // CUDA events used for GPU timing
  // ------------------------------------------------------------

  float elapsed_time;

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // ------------------------------------------------------------
  // Determine maximum memory needed across all tests
  //
  // A = M x K
  // B = K x N
  // C = M x N
  // ------------------------------------------------------------

  size_t maxA = 0;
  size_t maxB = 0;
  size_t maxC = 0;

  for (const auto &[M, N, K] : TESTS) {
    maxA = std::max(maxA, static_cast<size_t>(M) * K);
    maxB = std::max(maxB, static_cast<size_t>(K) * N);
    maxC = std::max(maxC, static_cast<size_t>(M) * N);
  }

  std::cout << "Maximum allocated elements:"
            << " A=" << maxA
            << ", B=" << maxB
            << ", C=" << maxC << std::endl;

  float alpha = 0.5f;
  float beta = 3.0f; // C = alpha * A*B + beta * C

  // ------------------------------------------------------------
  // Host matrices
  // ------------------------------------------------------------

  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  float *C_ref = nullptr;

  A = static_cast<float *>(malloc(sizeof(float) * maxA));
  B = static_cast<float *>(malloc(sizeof(float) * maxB));
  C = static_cast<float *>(malloc(sizeof(float) * maxC));
  C_ref = static_cast<float *>(malloc(sizeof(float) * maxC));

  if (!A || !B || !C || !C_ref) {
    std::cerr << "Host memory allocation failed." << std::endl;
    exit(EXIT_FAILURE);
  }

  randomize_matrix(A, static_cast<int>(maxA));
  randomize_matrix(B, static_cast<int>(maxB));
  randomize_matrix(C, static_cast<int>(maxC));

  // C_ref must initially contain the same values as C
  copy_matrix(C, C_ref, static_cast<int>(maxC));

  // ------------------------------------------------------------
  // Device matrices
  // ------------------------------------------------------------

  float *dA = nullptr;
  float *dB = nullptr;
  float *dC = nullptr;
  float *dC_ref = nullptr;

  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dA),
                       sizeof(float) * maxA));

  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dB),
                       sizeof(float) * maxB));

  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dC),
                       sizeof(float) * maxC));

  cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dC_ref),
                       sizeof(float) * maxC));

  cudaCheck(cudaMemcpy(dA, A,
                       sizeof(float) * maxA,
                       cudaMemcpyHostToDevice));

  cudaCheck(cudaMemcpy(dB, B,
                       sizeof(float) * maxB,
                       cudaMemcpyHostToDevice));

  cudaCheck(cudaMemcpy(dC, C,
                       sizeof(float) * maxC,
                       cudaMemcpyHostToDevice));

  cudaCheck(cudaMemcpy(dC_ref, C_ref,
                       sizeof(float) * maxC,
                       cudaMemcpyHostToDevice));

  // ------------------------------------------------------------
  // Benchmark
  // ------------------------------------------------------------

  const int repeat_times = 50;

  for (const auto &[M, N, K] : TESTS) {
    long m = M;
    long n = N;
    long k = K;

    std::cout << "\n----------------------------------------\n";
    std::cout << "Dimensions: "
              << "M=" << m
              << ", N=" << n
              << ", K=" << k
              << ", alpha=" << alpha
              << ", beta=" << beta
              << std::endl;

    // ----------------------------------------------------------
    // Correctness verification
    // ----------------------------------------------------------

    if (kernel_num != 0) {
      // dC and dC_ref must start from exactly the same C matrix
      cudaCheck(cudaMemcpy(dC_ref, dC,
                           sizeof(float) * m * n,
                           cudaMemcpyDeviceToDevice));

      // Reference result using cuBLAS
      run_kernel(0,
                 m, n, k,
                 alpha,
                 dA,
                 dB,
                 beta,
                 dC_ref,
                 handle);

      // Our kernel
      run_kernel(kernel_num,
                 m, n, k,
                 alpha,
                 dA,
                 dB,
                 beta,
                 dC,
                 handle);

      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError());

      cudaCheck(cudaMemcpy(C,
                           dC,
                           sizeof(float) * m * n,
                           cudaMemcpyDeviceToHost));

      cudaCheck(cudaMemcpy(C_ref,
                           dC_ref,
                           sizeof(float) * m * n,
                           cudaMemcpyDeviceToHost));

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout
            << "Failed to pass correctness verification against NVIDIA cuBLAS."
            << std::endl;

        if (m <= 128 && n <= 128 && k <= 128) {
          std::cout << "Logging faulty output into "
                    << errLogFile << "\n";

          std::ofstream fs;
          fs.open(errLogFile);

          fs << "A (" << m << " x " << k << "):\n";
          print_matrix(A, m, k, fs);

          fs << "B (" << k << " x " << n << "):\n";
          print_matrix(B, k, n, fs);

          fs << "C (" << m << " x " << n << "):\n";
          print_matrix(C, m, n, fs);

          fs << "Reference (" << m << " x " << n << "):\n";
          print_matrix(C_ref, m, n, fs);

          fs.close();
        }

        exit(EXIT_FAILURE);
      }

      std::cout << "Correctness verification: PASSED" << std::endl;

      // Restore dC to the reference result so both are synchronized
      cudaCheck(cudaMemcpy(dC,
                           dC_ref,
                           sizeof(float) * m * n,
                           cudaMemcpyDeviceToDevice));
    }

    // ----------------------------------------------------------
    // Performance benchmark
    // ----------------------------------------------------------

    cudaEventRecord(beg);

    for (int j = 0; j < repeat_times; j++) {
      // We intentionally don't reset C between repetitions.
      // Both runtime and FLOP/s are measured consistently.
      run_kernel(kernel_num,
                 m, n, k,
                 alpha,
                 dA,
                 dB,
                 beta,
                 dC,
                 handle);
    }

    cudaEventRecord(end);

    cudaEventSynchronize(end);

    cudaEventElapsedTime(&elapsed_time, beg, end);

    // cudaEventElapsedTime gives milliseconds
    elapsed_time /= 1000.0f;

    const long long flops =
        2LL * static_cast<long long>(m) *
        static_cast<long long>(n) *
        static_cast<long long>(k);

    const double average_time =
        elapsed_time / repeat_times;

    const double gflops =
        (repeat_times *
         static_cast<double>(flops) *
         1e-9) /
        elapsed_time;

    printf(
        "Average elapsed time: (%7.6f) s, "
        "performance: (%7.1f) GFLOPS. "
        "M: (%ld), N: (%ld), K: (%ld).\n",
        average_time,
        gflops,
        m,
        n,
        k);

    fflush(stdout);

    // Reset C after benchmarking so the next test starts cleanly
    cudaCheck(cudaMemcpy(dC,
                         C,
                         sizeof(float) * m * n,
                         cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(dC_ref,
                         C,
                         sizeof(float) * m * n,
                         cudaMemcpyHostToDevice));
  }

  // ------------------------------------------------------------
  // Cleanup
  // ------------------------------------------------------------

  free(A);
  free(B);
  free(C);
  free(C_ref);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);

  cudaEventDestroy(beg);
  cudaEventDestroy(end);

  cublasDestroy(handle);

  return 0;
}