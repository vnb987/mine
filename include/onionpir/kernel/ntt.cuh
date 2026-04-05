#pragma once
#include "../common/types.cuh"

namespace onionpir {
namespace kernel {

// ============================================================================
// NTT (Number Theoretic Transform) CUDA Kernels
// Forward NTT: coefficient form -> evaluation form
// Inverse NTT: evaluation form -> coefficient form
//
// Uses Cooley-Tukey butterfly for forward, Gentleman-Sande for inverse.
// All operations mod a single prime q.
// ============================================================================

// Host-side NTT tables stored on GPU
struct NTTTable {
    uint64_t* d_psi_powers = nullptr;      // powers of psi (twiddle factors), N elements
    uint64_t* d_psi_inv_powers = nullptr;  // powers of psi^{-1}, N elements
    uint64_t q = 0;
    uint64_t n_inv = 0;                    // N^{-1} mod q
    bool initialized = false;

    void init(const NTTParams& params);
    void destroy();
    ~NTTTable() { destroy(); }
};

// Multi-modulus NTT tables
struct NTTContext {
    NTTTable tables[MAX_RNS_MODULI];
    uint64_t moduli[MAX_RNS_MODULI];
    uint32_t num_moduli = 0;

    void init(const uint64_t* qs, uint32_t count);
    void destroy();
};

// ---- Device kernels (called from host wrappers) ----------------------------

// Forward NTT on a single polynomial, in-place
// data: N uint64_t coefficients
// psi_powers: precomputed twiddle factors
// q: modulus
__global__ void ntt_forward_kernel(uint64_t* data, const uint64_t* psi_powers,
                                    uint64_t q);

// Inverse NTT on a single polynomial, in-place
__global__ void ntt_inverse_kernel(uint64_t* data, const uint64_t* psi_inv_powers,
                                    uint64_t q, uint64_t n_inv);

// Batch NTT: apply forward NTT to `count` polynomials
// Each polynomial has N coefficients, stored contiguously
__global__ void ntt_forward_batch_kernel(uint64_t* data, uint32_t count,
                                          const uint64_t* psi_powers, uint64_t q);

__global__ void ntt_inverse_batch_kernel(uint64_t* data, uint32_t count,
                                          const uint64_t* psi_inv_powers,
                                          uint64_t q, uint64_t n_inv);

// ---- Host-side wrappers ----------------------------------------------------

// Forward NTT on DevicePoly
void ntt_forward(DevicePoly& poly, const NTTTable& table, cudaStream_t stream = 0);

// Inverse NTT on DevicePoly
void ntt_inverse(DevicePoly& poly, const NTTTable& table, cudaStream_t stream = 0);

// Forward NTT on DevicePolyArray (batch)
void ntt_forward_batch(DevicePolyArray& arr, const NTTTable& table,
                        cudaStream_t stream = 0);

void ntt_inverse_batch(DevicePolyArray& arr, const NTTTable& table,
                        cudaStream_t stream = 0);

// Forward NTT on RnsPoly (all moduli)
void ntt_forward_rns(RnsPoly& poly, const NTTContext& ctx, cudaStream_t stream = 0);

void ntt_inverse_rns(RnsPoly& poly, const NTTContext& ctx, cudaStream_t stream = 0);

} // namespace kernel
} // namespace onionpir
