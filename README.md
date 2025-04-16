# Aligned vs. Unaligned Memory Access

## Overview

This project benchmarks the performance impact of **aligned** vs. **unaligned** memory access when summing large arrays using SIMD instructions. It supports **AVX (x86_64)** and **NEON (ARM/Apple Silicon)** with a fallback to scalar summation.

By comparing the access patterns, this project reveals how misaligned data can affect performance due to memory alignment, cache-line boundaries, and instruction-level differences.

---

## Problem Description

When using SIMD instructions like AVX or NEON, aligned memory access is generally faster because it can be fetched efficiently in a single CPU instruction without crossing cache-line or memory boundaries. Unaligned memory access may lead to:

- Split loads across cache lines
- Additional microcode penalties
- Higher latency in memory fetch
- Reduced throughput in SIMD pipelines

This project:
- Allocates aligned memory
- Applies manual byte offsets to simulate unaligned access
- Measures time to sum all elements of a large array under various misalignment offsets

---

## Explanation of Concepts

### Aligned Access
Memory access starts at an address that is a multiple of the alignment requirement (e.g., 32 bytes for AVX). SIMD instructions like `_mm256_load_pd` work optimally here.

### Unaligned Access
Memory starts at a non-aligned offset (e.g., +1, +2, +8 bytes). This can cause extra memory loads or cross cache-line boundaries. SIMD uses `_mm256_loadu_pd` or equivalent.

### Cache Flushing
Before each test, a fake memory sweep flushes CPU caches to simulate cold memory access. This ensures a fair and realistic performance comparison between aligned and unaligned memory reads.

---

## Example Output

```
[INFO] Using AVX intrinsics (x86_64)
Detected cache line size: 64 bytes
Array size: 500000000

Access Type         Sum                 Time (ms)
-------------------------------------------------------
Aligned             250012504           572.2540
Unaligned +1        250012504           636.5310
Unaligned +2        250012504           700.6000
Unaligned +4        250012504           775.7850
Unaligned +8        250012504           1826.7120
Unaligned +16       250012504           1919.2790
Unaligned +24       250012504           2559.3850
```

### Explanation of Output
- The **Sum** remains consistent, confirming correctness.
- The **Time (ms)** increases as the memory offset deviates more from the aligned boundary.
- Unaligned accesses like `+8`, `+16`, `+24` show significant slowdowns due to crossing multiple cache lines or banks.

---

## How to Compile and Run

### 1. Clone the Repository
```bash
git clone https://github.com/LyudmilaKostanyan/Aligned-vs-Unaligned-Memory-Access.git
cd Aligned-vs-Unaligned-Memory-Access
```

### 2. Build the Project
Make sure you have CMake and a C++ compiler installed.

```bash
cmake -S . -B build
cmake --build build
```

### 3. Run the Program

#### With Default Array Size
```bash
./build/main
```

#### With Custom Size (e.g., 500 million elements)
```bash
./build/main --n 500000000
```

#### For Windows (if executable is named differently)
```bash
./build/main.exe --n 500000000
```

---

## Parameters

| Flag     | Description                              | Example              |
|----------|------------------------------------------|----------------------|
| `--n`    | Number of elements to generate and sum   | `--n 500000000`      |

This parameter controls the size of the array to benchmark. Larger sizes increase memory pressure and help highlight caching effects.
