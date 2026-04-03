#pragma once
#include <cstdint>
#include <cstddef>

namespace onionpir {

// Ring polynomial degree N = X^N + 1
constexpr uint32_t LOG_N = 12;
constexpr uint32_t N = 1u << LOG_N; // 4096

// NTT-friendly primes: q_i ≡ 1 (mod 2N)
// Each prime < 2^54 so that product of two fits in 128 bits
constexpr int MAX_RNS_MODULI = 4;

struct NTTParams {
    uint64_t q;          // modulus
    uint64_t psi;        // primitive 2N-th root of unity mod q
    uint64_t psi_inv;    // inverse of psi mod q
    uint64_t n_inv;      // inverse of N mod q
    uint64_t barrett_cr;  // Barrett reduction constant: floor(2^128 / q)
    uint64_t barrett_cq;
};

// Default RNS moduli (q ≡ 1 mod 2N)
// q0 = 36028797014376449  (close to 2^55, ≡ 1 mod 8192)
// q1 = 36028797015949313
// q2 = 36028797017522177  (auxiliary for key-switching)
constexpr uint64_t DEFAULT_MODULI[] = {
    0x80000000080001ULL,   // 2^55 + 2^19 + 1 = 36028797019234305 -- ≡1 mod 8192
    0x7FFFFFFFBA0001ULL,   // another NTT-friendly prime
    0x7FFFFFFFAA0001ULL,   // auxiliary
};
constexpr int DEFAULT_NUM_MODULI = 2;  // ciphertext moduli count
constexpr int AUX_MODULI = 1;          // auxiliary for key-switching

// Plaintext modulus
constexpr uint64_t PLAIN_MOD = (1ULL << 20); // 2^20 for encoding DB entries

// RGSW / Gadget decomposition parameters
constexpr int DECOMP_BASE_LOG = 18;          // log2(B_g)
constexpr uint64_t DECOMP_BASE = 1ULL << DECOMP_BASE_LOG;
// Number of decomposition levels per modulus: ceil(log_q / log_B)
constexpr int DECOMP_LEVELS = 4;  // ceil(55/18) ≈ 4

// PIR parameters
constexpr int PIR_NUM_DIM = 2;             // 2-dimensional PIR
constexpr int FIRST_DIM_SIZE = N;          // first dimension = N (SimplePIR style)

// GPU configuration
constexpr int CUDA_THREADS_PER_BLOCK = 256;
constexpr int MAX_BATCH_SIZE = 256;        // max queries in a batch
constexpr size_t GPU_MEM_POOL_SIZE = 4ULL * 1024 * 1024 * 1024; // 4GB pool

// Database constants
constexpr size_t DB_ENTRY_BYTES = 256;     // bytes per DB entry (configurable)
constexpr size_t DB_CHUNK_SIZE = 1ULL * 1024 * 1024 * 1024; // 1GB chunk for streaming

} // namespace onionpir
