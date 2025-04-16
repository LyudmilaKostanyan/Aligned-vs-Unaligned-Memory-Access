#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include "kaizen.h"

#if defined(__AVX__)
    #include <immintrin.h>
    #define USE_AVX 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define USE_NEON 1
#else
    #define USE_SCALAR 1
#endif

#if defined(_WIN32) && defined(_MSC_VER)
#include <intrin.h>
#endif

size_t SIZE = 500'000'000;
constexpr size_t MAX_OFFSET = 32;

size_t get_cache_line_size() {
#if defined(_WIN32) && defined(_MSC_VER)
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return ((cpuInfo[1] >> 8) & 0xFF) * 8;
#elif defined(_SC_LEVEL1_DCACHE_LINESIZE)
    long line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    return line_size > 0 ? static_cast<size_t>(line_size) : 64;
#else
    return 64;
#endif
}

void print_result(const std::string& label, double sum, double time_ms) {
    std::cout << std::left
              << std::setw(20) << label
              << std::setw(20) << std::fixed << std::setprecision(0) << sum
              << std::setw(15) << std::fixed << std::setprecision(4) << time_ms
              << "\n";
}

void flush_cpu_cache(size_t flush_size = 10 * 1024 * 1024) {
    std::vector<char> trash(flush_size, 1);
    volatile char sink = 0;
    for (size_t i = 0; i < trash.size(); i += 64) {
        sink ^= trash[i];
    }
}

#if USE_AVX
double simd_sum(const double* data, bool aligned) {
    __m256d sum = _mm256_setzero_pd();
    for (size_t i = 0; i < SIZE; i += 4) {
        __m256d v = aligned ? _mm256_load_pd(data + i) : _mm256_loadu_pd(data + i);
        sum = _mm256_add_pd(sum, v);
    }
    double result[4];
    _mm256_storeu_pd(result, sum);
    return result[0] + result[1] + result[2] + result[3];
}
#elif USE_NEON
double simd_sum(const double* data, bool /*aligned*/) {
    float64x2_t sum0 = vdupq_n_f64(0.0), sum1 = vdupq_n_f64(0.0);
    for (size_t i = 0; i < SIZE; i += 4) {
        float64x2_t v0 = vld1q_f64(data + i);
        float64x2_t v1 = vld1q_f64(data + i + 2);
        sum0 = vaddq_f64(sum0, v0);
        sum1 = vaddq_f64(sum1, v1);
    }
    float64x2_t total = vaddq_f64(sum0, sum1);
    return vgetq_lane_f64(total, 0) + vgetq_lane_f64(total, 1);
}
#else
double simd_sum(const double* data, bool /*aligned*/) {
    double sum = 0.0;
    for (size_t i = 0; i < SIZE; ++i)
        sum += data[i];
    return sum;
}
#endif

double measure_time(double (*func)(const double*, bool), const double* data, bool aligned, double& sum_out) {
    zen::timer timer;
    timer.start();
    sum_out = func(data, aligned);
    timer.stop();
    return timer.duration<zen::timer::usec>().count() / 1000.0;
}

void parse_args(int argc, char** argv) {
    for (int i = 1; i < argc - 1; ++i) {
        std::string arg = argv[i];
        if (arg == "--n") {
            size_t parsed = std::stoull(argv[i + 1]);
            if (parsed > 0)
                SIZE = parsed;
        }
    }
}

int main(int argc, char** argv) {
    parse_args(argc, argv);

#if USE_AVX
    std::cout << "\n[INFO] Using AVX intrinsics (x86_64)\n";
#elif USE_NEON
    std::cout << "\n[INFO] Using NEON intrinsics (ARM/Apple Silicon)\n";
#else
    std::cout << "\n[INFO] Using scalar fallback\n";
#endif

    std::cout << "Detected cache line size: " << get_cache_line_size() << " bytes\n";

#if defined(_WIN32)
    void* raw = _aligned_malloc(SIZE * sizeof(double) + MAX_OFFSET, 32);
    if (!raw) return 1;
#else
    void* raw = nullptr;
    if (posix_memalign(&raw, 32, SIZE * sizeof(double) + MAX_OFFSET) != 0) return 1;
#endif

    double* aligned_data = static_cast<double*>(raw);

    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (size_t i = 0; i < SIZE; ++i) {
        aligned_data[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }

    std::cout << "Array size: " << SIZE << "\n\n";
    std::cout << std::left
              << std::setw(20) << "Access Type"
              << std::setw(20) << "Sum"
              << std::setw(15) << "Time (ms)"
              << "\n";
    std::cout << std::string(55, '-') << "\n";

    double sum_result = 0.0;

    flush_cpu_cache();
    double time_taken = measure_time(simd_sum, aligned_data, true, sum_result);
    print_result("Aligned", sum_result, time_taken);

    std::vector<size_t> offsets = {1, 2, 4, 8, 16, 24};
    for (size_t offset : offsets) {
#if defined(_WIN32)
        void* raw_unaligned = _aligned_malloc(SIZE * sizeof(double) + MAX_OFFSET, 32);
        if (!raw_unaligned) continue;
#else
        void* raw_unaligned = nullptr;
        if (posix_memalign(&raw_unaligned, 32, SIZE * sizeof(double) + MAX_OFFSET) != 0) continue;
#endif
        double* base = static_cast<double*>(raw_unaligned);
        double* unaligned_data = reinterpret_cast<double*>(
            reinterpret_cast<char*>(base) + offset);

        std::memcpy(unaligned_data, aligned_data, SIZE * sizeof(double));

        flush_cpu_cache();
        sum_result = 0.0;
        time_taken = measure_time(simd_sum, unaligned_data, false, sum_result);
        print_result("Unaligned +" + std::to_string(offset), sum_result, time_taken);

#if defined(_WIN32)
        _aligned_free(raw_unaligned);
#else
        free(raw_unaligned);
#endif
    }

#if defined(_WIN32)
    _aligned_free(raw);
#else
    free(raw);
#endif

    return 0;
}
