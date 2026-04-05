#pragma once
#include "../common/types.cuh"
#include "../kernel/ntt.cuh"
#include "../kernel/crt.cuh"
#include "random.cuh"

namespace onionpir {
namespace crypto {

// ============================================================================
// Cryptographic Context
// Central object holding all parameters and precomputed tables needed for
// SimplePIR-based HE operations. Created once, shared across all operations.
// ============================================================================

struct CryptoContext {
    // RNS moduli
    uint64_t moduli[MAX_RNS_MODULI];
    uint64_t* d_moduli = nullptr;   // moduli array on GPU
    uint32_t num_moduli = 0;

    // NTT tables (one per modulus)
    kernel::NTTContext ntt_ctx;

    // CRT context
    kernel::CRTContext crt_ctx;

    // Random number generator
    RandomContext rng;

    // Plaintext modulus
    uint64_t plain_mod = PLAIN_MOD;

    // RGSW parameters
    uint32_t decomp_levels = DECOMP_LEVELS;
    uint64_t decomp_base = DECOMP_BASE;
    uint32_t decomp_base_log = DECOMP_BASE_LOG;

    // Precomputed: gadget vector powers [B^0, B^1, ..., B^{l-1}] mod each q_i
    uint64_t* d_gadget_powers = nullptr;  // decomp_levels * num_moduli values on GPU

    // Precomputed: plain_mod^{-1} mod q_i for decryption scaling
    uint64_t* d_plain_mod_inv = nullptr;  // num_moduli values on GPU

    // Delta = floor(q / t) for encoding
    uint64_t* d_delta = nullptr;  // num_moduli values on GPU

    // Initialize with default or custom parameters
    void init(const uint64_t* qs = nullptr, uint32_t nmod = 0, uint64_t seed = 42);

    // Clean up all GPU resources
    void destroy();

    ~CryptoContext() { destroy(); }

    // Non-copyable
    CryptoContext() = default;
    CryptoContext(const CryptoContext&) = delete;
    CryptoContext& operator=(const CryptoContext&) = delete;
};

} // namespace crypto
} // namespace onionpir
