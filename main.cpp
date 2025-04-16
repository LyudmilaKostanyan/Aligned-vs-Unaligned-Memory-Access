#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

#if defined(_WIN32)
#include <intrin.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

constexpr size_t SIZE = 100'000'000;
constexpr size_t MAX_OFFSET = 32;

size_t get_cache_line_size() {
#if defined(_WIN32)
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return ((cpuInfo[1] >> 8) & 0xFF) * 8;
#elif defined(_SC_LEVEL1_DCACHE_LINESIZE)
    long line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (line_size > 0) return static_cast<size_t>(line_size);
#endif
    return 64;
}

void print_result(const std::string& label, double sum, double time_ms) {
    std::cout << std::left
              << std::setw(20) << label
              << std::setw(20) << std::fixed << std::setprecision(0) << sum
              << std::setw(15) << std::fixed << std::setprecision(4) << time_ms
              << "\n";
}

double aligned_sum(const double* data) {
    __m256d sum = _mm256_setzero_pd();
    for (size_t i = 0; i < SIZE; i += 4) {
        __m256d v = _mm256_load_pd(data + i);
        sum = _mm256_add_pd(sum, v);
    }
    double result[4];
    _mm256_storeu_pd(result, sum);
    return result[0] + result[1] + result[2] + result[3];
}

double unaligned_sum(const double* data) {
    __m256d sum = _mm256_setzero_pd();
    for (size_t i = 0; i < SIZE; i += 4) {
        __m256d v = _mm256_loadu_pd(data + i);
        sum = _mm256_add_pd(sum, v);
    }
    double result[4];
    _mm256_storeu_pd(result, sum);
    return result[0] + result[1] + result[2] + result[3];
}

double measure_time(double (*func)(const double*), const double* data, double& sum_out) {
    auto start = std::chrono::high_resolution_clock::now();
    sum_out = func(data);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    size_t ALIGNMENT = get_cache_line_size();
    std::cout << "\nDetected cache line size: " << ALIGNMENT << " bytes\n";

    if (ALIGNMENT < 32) ALIGNMENT = 32;

#if defined(_WIN32)
    void* raw = _aligned_malloc(SIZE * sizeof(double) + MAX_OFFSET, ALIGNMENT);
    if (!raw) {
        std::cerr << "Failed to allocate aligned memory\n";
        return 1;
    }
#else
    void* raw = nullptr;
    if (posix_memalign(&raw, ALIGNMENT, SIZE * sizeof(double) + MAX_OFFSET) != 0) {
        std::cerr << "Failed to allocate aligned memory\n";
        return 1;
    }
#endif

    double* aligned_data = static_cast<double*>(raw);
    std::fill(aligned_data, aligned_data + SIZE, 1.0);

    double aligned_sum_result = 0.0;
    double time_aligned = measure_time(aligned_sum, aligned_data, aligned_sum_result);

    std::cout << "Array size (iterations): " << SIZE << "\n\n";
    std::cout << std::left
              << std::setw(20) << "Access Type"
              << std::setw(20) << "Sum"
              << std::setw(15) << "Time (ms)"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    print_result("Aligned", aligned_sum_result, time_aligned);

    std::vector<size_t> offsets = {1, 2, 4, 8, 16, 24};
    for (size_t offset : offsets) {
        const double* unaligned_data = reinterpret_cast<const double*>(reinterpret_cast<const char*>(aligned_data) + offset);

        std::vector<double> temp(SIZE, 1.0);
        std::memcpy((void*)unaligned_data, temp.data(), SIZE * sizeof(double));

        double sum_result = 0.0;
        double time_taken = measure_time(unaligned_sum, unaligned_data, sum_result);

        std::string label = "Unaligned +" + std::to_string(offset);
        print_result(label, sum_result, time_taken);
    }

#if defined(_WIN32)
    _aligned_free(raw);
#else
    free(raw);
#endif

    return 0;
}
