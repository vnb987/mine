#pragma once
#include "params.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <cassert>
#include <stdexcept>

namespace onionpir {

// ============================================================================
// Hierarchical Data Structure
//
// Level 0 (Kernel):  DevicePoly    -- raw polynomial on GPU (N uint64_t coeffs)
// Level 1 (Crypto):  RnsPoly       -- RNS representation (one DevicePoly per modulus)
// Level 2 (Arith):   RlweCt / RgswCt -- ciphertexts composed of RnsPoly
// Level 3 (Eval):    EvalKey       -- evaluation keys (collections of RgswCt)
// Level 4 (Module):  PirDb / PirQuery / PirResponse
//
// Each level only depends on the level directly below it.
// Changes at one level are isolated from other levels.
// ============================================================================

// ---- Helpers ---------------------------------------------------------------

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
                cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

// ============================================================================
// Level 0 (Kernel): DevicePoly
// A single polynomial of degree < N stored on GPU.
// Coefficients are uint64_t, stored in NTT-evaluation form by default.
// ============================================================================
struct DevicePoly {
    uint64_t* d_coeffs = nullptr; // GPU pointer, N elements
    bool owns_memory = false;

    DevicePoly() = default;

    // Allocate N coefficients on GPU
    void allocate() {
        if (d_coeffs && owns_memory) free();
        CUDA_CHECK(cudaMalloc(&d_coeffs, N * sizeof(uint64_t)));
        owns_memory = true;
    }

    // Point to existing GPU memory (non-owning view)
    void view(uint64_t* ptr) {
        if (d_coeffs && owns_memory) free();
        d_coeffs = ptr;
        owns_memory = false;
    }

    void free() {
        if (d_coeffs && owns_memory) {
            cudaFree(d_coeffs);
        }
        d_coeffs = nullptr;
        owns_memory = false;
    }

    void copy_from_host(const uint64_t* h_data, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_coeffs, h_data, N * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host(uint64_t* h_data, cudaStream_t stream = 0) const {
        CUDA_CHECK(cudaMemcpyAsync(h_data, d_coeffs, N * sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost, stream));
    }

    // Zero out
    void zero(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemsetAsync(d_coeffs, 0, N * sizeof(uint64_t), stream));
    }

    ~DevicePoly() { free(); }

    // Move semantics
    DevicePoly(DevicePoly&& o) noexcept : d_coeffs(o.d_coeffs), owns_memory(o.owns_memory) {
        o.d_coeffs = nullptr; o.owns_memory = false;
    }
    DevicePoly& operator=(DevicePoly&& o) noexcept {
        if (this != &o) { free(); d_coeffs = o.d_coeffs; owns_memory = o.owns_memory;
            o.d_coeffs = nullptr; o.owns_memory = false; }
        return *this;
    }
    DevicePoly(const DevicePoly&) = delete;
    DevicePoly& operator=(const DevicePoly&) = delete;
};

// Bulk allocation: contiguous GPU memory for multiple polynomials
struct DevicePolyArray {
    uint64_t* d_data = nullptr;  // contiguous GPU memory
    uint32_t count = 0;
    bool owns_memory = false;

    void allocate(uint32_t n) {
        if (d_data && owns_memory) free();
        count = n;
        CUDA_CHECK(cudaMalloc(&d_data, (size_t)n * N * sizeof(uint64_t)));
        owns_memory = true;
    }

    void view(uint64_t* ptr, uint32_t n) {
        if (d_data && owns_memory) free();
        d_data = ptr;
        count = n;
        owns_memory = false;
    }

    // Get pointer to i-th polynomial
    __host__ __device__ uint64_t* poly(uint32_t i) {
        return d_data + (size_t)i * N;
    }
    __host__ __device__ const uint64_t* poly(uint32_t i) const {
        return d_data + (size_t)i * N;
    }

    void zero(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemsetAsync(d_data, 0, (size_t)count * N * sizeof(uint64_t), stream));
    }

    void free() {
        if (d_data && owns_memory) cudaFree(d_data);
        d_data = nullptr; count = 0; owns_memory = false;
    }

    ~DevicePolyArray() { free(); }

    DevicePolyArray() = default;
    DevicePolyArray(DevicePolyArray&& o) noexcept
        : d_data(o.d_data), count(o.count), owns_memory(o.owns_memory) {
        o.d_data = nullptr; o.count = 0; o.owns_memory = false;
    }
    DevicePolyArray& operator=(DevicePolyArray&& o) noexcept {
        if (this != &o) { free(); d_data = o.d_data; count = o.count; owns_memory = o.owns_memory;
            o.d_data = nullptr; o.count = 0; o.owns_memory = false; }
        return *this;
    }
    DevicePolyArray(const DevicePolyArray&) = delete;
    DevicePolyArray& operator=(const DevicePolyArray&) = delete;
};

// ============================================================================
// Level 1 (Crypto): RnsPoly
// Polynomial in RNS representation: one DevicePoly per modulus.
// Stores polynomial mod (q0, q1, ..., q_{k-1}) independently.
// ============================================================================
struct RnsPoly {
    DevicePolyArray polys;       // num_moduli polynomials stored contiguously
    uint32_t num_moduli = 0;

    void allocate(uint32_t nmod) {
        num_moduli = nmod;
        polys.allocate(nmod);
    }

    // Get the polynomial component for modulus index i
    uint64_t* component(uint32_t i) { return polys.poly(i); }
    const uint64_t* component(uint32_t i) const { return polys.poly(i); }

    // Raw data pointer (all components contiguous)
    uint64_t* data() { return polys.d_data; }
    const uint64_t* data() const { return polys.d_data; }

    // Total number of uint64_t elements
    size_t total_elements() const { return (size_t)num_moduli * N; }

    void zero(cudaStream_t stream = 0) { polys.zero(stream); }

    RnsPoly() = default;
    RnsPoly(RnsPoly&&) = default;
    RnsPoly& operator=(RnsPoly&&) = default;
};

// ============================================================================
// Level 2 (Arith): RlweCt, RgswCt
// ============================================================================

// RLWE ciphertext: (a, b) where b = a*s + e + m (in NTT domain)
// Stored as two RnsPoly
struct RlweCt {
    RnsPoly a;  // "a" component
    RnsPoly b;  // "b" component
    uint32_t num_moduli = 0;

    void allocate(uint32_t nmod) {
        num_moduli = nmod;
        a.allocate(nmod);
        b.allocate(nmod);
    }

    void zero(cudaStream_t stream = 0) { a.zero(stream); b.zero(stream); }

    RlweCt() = default;
    RlweCt(RlweCt&&) = default;
    RlweCt& operator=(RlweCt&&) = default;
};

// RGSW ciphertext: a (2 * decomp_levels) x 2 matrix of RLWE-like rows
// Row structure: [ RLWE(m * B^i), RLWE(m * s * B^i) ] for each decomposition level
// Stored as a flat array of RnsPoly for GPU efficiency
struct RgswCt {
    // Total rows = 2 * decomp_levels (top half encrypts m*B^i, bottom half m*s*B^i)
    // Each row has 2 polynomials (a, b components)
    // Total polynomials = 2 * decomp_levels * 2
    DevicePolyArray data;
    uint32_t decomp_levels = 0;
    uint32_t num_moduli = 0;

    // Total poly count = 2 * decomp_levels * 2 * num_moduli
    void allocate(uint32_t levels, uint32_t nmod) {
        decomp_levels = levels;
        num_moduli = nmod;
        uint32_t total = 2 * levels * 2 * nmod;
        data.allocate(total);
    }

    // Access row r (0..2*decomp_levels-1), column c (0=a, 1=b), modulus m
    uint64_t* component(uint32_t r, uint32_t c, uint32_t m) {
        uint32_t idx = (r * 2 + c) * num_moduli + m;
        return data.poly(idx);
    }
    const uint64_t* component(uint32_t r, uint32_t c, uint32_t m) const {
        uint32_t idx = (r * 2 + c) * num_moduli + m;
        return data.poly(idx);
    }

    void zero(cudaStream_t stream = 0) { data.zero(stream); }

    RgswCt() = default;
    RgswCt(RgswCt&&) = default;
    RgswCt& operator=(RgswCt&&) = default;
};

// ============================================================================
// Level 3 (Eval): EvalKey
// Evaluation keys for automorphisms and key-switching
// ============================================================================

// Key-switching key: encrypts s' under s for modulus raising
struct KSwitchKey {
    // decomp_levels RLWE ciphertexts encrypting s'*B^i under s
    DevicePolyArray data;  // decomp_levels * 2 * num_moduli polynomials
    uint32_t decomp_levels = 0;
    uint32_t num_moduli = 0;

    void allocate(uint32_t levels, uint32_t nmod) {
        decomp_levels = levels;
        num_moduli = nmod;
        data.allocate(levels * 2 * nmod);
    }

    uint64_t* component(uint32_t level, uint32_t ab, uint32_t mod) {
        return data.poly((level * 2 + ab) * num_moduli + mod);
    }
    const uint64_t* component(uint32_t level, uint32_t ab, uint32_t mod) const {
        return data.poly((level * 2 + ab) * num_moduli + mod);
    }

    KSwitchKey() = default;
    KSwitchKey(KSwitchKey&&) = default;
    KSwitchKey& operator=(KSwitchKey&&) = default;
};

// Automorphism key: RGSW ciphertext for Galois element
struct AutoKey {
    RgswCt rgsw;
    uint32_t galois_elt = 0;  // the Galois element (e.g., 3, 5, ...)
};

// Full evaluation key set for PIR
struct EvalKeySet {
    std::vector<AutoKey> auto_keys;   // automorphism keys
    KSwitchKey ks_key;                // key-switching key (if needed)

    EvalKeySet() = default;
    EvalKeySet(EvalKeySet&&) = default;
    EvalKeySet& operator=(EvalKeySet&&) = default;
};

// ============================================================================
// Level 4 (Module): PirDb, PirQuery, PirResponse
// ============================================================================

// A chunk of the database encoded as NTT polynomials on GPU
struct DbChunk {
    DevicePolyArray data;      // polynomials in NTT form
    uint32_t num_polys = 0;
    uint32_t chunk_id = 0;

    void allocate(uint32_t n) {
        num_polys = n;
        data.allocate(n);
    }

    DbChunk() = default;
    DbChunk(DbChunk&&) = default;
    DbChunk& operator=(DbChunk&&) = default;
};

// PIR Database: potentially much larger than GPU memory
// Organized as chunks that can be streamed
struct PirDb {
    std::vector<DbChunk> chunks;           // DB chunks on GPU (or to be streamed)
    size_t num_entries = 0;                 // total DB entries
    size_t entry_bytes = DB_ENTRY_BYTES;    // bytes per entry
    uint32_t dim1_size = 0;                 // first dimension size
    uint32_t dim2_size = 0;                 // second dimension size
    uint32_t polys_per_entry = 0;           // polynomials needed per entry
    uint32_t num_moduli = 0;
    bool is_ntt = false;                    // whether data is in NTT form

    PirDb() = default;
    PirDb(PirDb&&) = default;
    PirDb& operator=(PirDb&&) = default;
};

// PIR Query: client's encrypted query
struct PirQuery {
    // First dimension: RLWE ciphertext for SimplePIR (matrix-vector product)
    RlweCt first_dim_ct;
    // Subsequent dimensions: RGSW ciphertexts (one per dimension, encoding index bits)
    std::vector<RgswCt> dim_cts;
    uint32_t query_id = 0;

    PirQuery() = default;
    PirQuery(PirQuery&&) = default;
    PirQuery& operator=(PirQuery&&) = default;
};

// PIR Response: server's answer
struct PirResponse {
    RlweCt ct;          // encrypted answer
    uint32_t query_id = 0;

    PirResponse() = default;
    PirResponse(PirResponse&&) = default;
    PirResponse& operator=(PirResponse&&) = default;
};

// Batch of queries for amortized processing
struct PirBatch {
    std::vector<PirQuery> queries;
    std::vector<PirResponse> responses;
    uint32_t batch_size = 0;

    PirBatch() = default;
    PirBatch(PirBatch&&) = default;
    PirBatch& operator=(PirBatch&&) = default;
};

// GPU stream pool for async operations
struct StreamPool {
    std::vector<cudaStream_t> streams;
    uint32_t count = 0;

    void create(uint32_t n) {
        count = n;
        streams.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
    }

    cudaStream_t get(uint32_t i) const { return streams[i % count]; }

    void sync_all() {
        for (auto& s : streams) cudaStreamSynchronize(s);
    }

    void destroy() {
        for (auto& s : streams) cudaStreamDestroy(s);
        streams.clear(); count = 0;
    }

    ~StreamPool() { destroy(); }
};

} // namespace onionpir
