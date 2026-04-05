#pragma once
#include "../common/types.cuh"

namespace onionpir {
namespace kernel {

// ============================================================================
// Element-wise CUDA Kernels
// Basic modular arithmetic operations on polynomial coefficients.
// All operate on arrays of N uint64_t values with a given modulus.
// ============================================================================

// ---- Modular arithmetic device functions -----------------------------------

// Barrett reduction: compute x mod q using precomputed constants
__device__ __forceinline__ uint64_t barrett_reduce(uint64_t x, uint64_t q);

// Modular addition: (a + b) mod q
__device__ __forceinline__ uint64_t mod_add(uint64_t a, uint64_t b, uint64_t q);

// Modular subtraction: (a - b) mod q
__device__ __forceinline__ uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t q);

// Modular multiplication: (a * b) mod q using 128-bit intermediate
__device__ __forceinline__ uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t q);

// ---- Element-wise kernels --------------------------------------------------

// c[i] = (a[i] + b[i]) mod q
__global__ void ewise_add_kernel(const uint64_t* a, const uint64_t* b,
                                  uint64_t* c, uint64_t q, uint32_t len);

// c[i] = (a[i] - b[i]) mod q
__global__ void ewise_sub_kernel(const uint64_t* a, const uint64_t* b,
                                  uint64_t* c, uint64_t q, uint32_t len);

// c[i] = (a[i] * b[i]) mod q  (Hadamard / pointwise multiply in NTT domain)
__global__ void ewise_mul_kernel(const uint64_t* a, const uint64_t* b,
                                  uint64_t* c, uint64_t q, uint32_t len);

// c[i] = (a[i] * scalar) mod q
__global__ void ewise_scalar_mul_kernel(const uint64_t* a, uint64_t scalar,
                                         uint64_t* c, uint64_t q, uint32_t len);

// c[i] += (a[i] * b[i]) mod q  (fused multiply-accumulate)
__global__ void ewise_mulacc_kernel(const uint64_t* a, const uint64_t* b,
                                     uint64_t* c, uint64_t q, uint32_t len);

// c[i] = -a[i] mod q
__global__ void ewise_negate_kernel(const uint64_t* a, uint64_t* c,
                                     uint64_t q, uint32_t len);

// Apply automorphism sigma_k: c[i] = a[k*i mod 2N] (with sign adjustment)
// Operates on coefficient form; index permutation for NTT form differs
__global__ void ewise_automorphism_kernel(const uint64_t* a, uint64_t* c,
                                           uint32_t galois_elt, uint64_t q,
                                           uint32_t n);

// Round and scale: c[i] = round(a[i] * t / q) mod t
// Used in decryption to extract plaintext
__global__ void ewise_round_scale_kernel(const uint64_t* a, uint64_t* c,
                                          uint64_t q, uint64_t t, uint32_t len);

// ---- Batch element-wise kernels (operate on arrays of polynomials) ---------

// Add arrays of polynomials: out[poly_idx] = a[poly_idx] + b[poly_idx]
__global__ void ewise_add_batch_kernel(const uint64_t* a, const uint64_t* b,
                                        uint64_t* c, uint64_t q,
                                        uint32_t poly_count);

__global__ void ewise_mul_batch_kernel(const uint64_t* a, const uint64_t* b,
                                        uint64_t* c, uint64_t q,
                                        uint32_t poly_count);

__global__ void ewise_mulacc_batch_kernel(const uint64_t* a, const uint64_t* b,
                                           uint64_t* c, uint64_t q,
                                           uint32_t poly_count);

// ---- Host wrappers ---------------------------------------------------------

// Single polynomial operations
void ewise_add(const DevicePoly& a, const DevicePoly& b, DevicePoly& c,
               uint64_t q, cudaStream_t stream = 0);

void ewise_sub(const DevicePoly& a, const DevicePoly& b, DevicePoly& c,
               uint64_t q, cudaStream_t stream = 0);

void ewise_mul(const DevicePoly& a, const DevicePoly& b, DevicePoly& c,
               uint64_t q, cudaStream_t stream = 0);

void ewise_scalar_mul(const DevicePoly& a, uint64_t scalar, DevicePoly& c,
                       uint64_t q, cudaStream_t stream = 0);

void ewise_mulacc(const DevicePoly& a, const DevicePoly& b, DevicePoly& c,
                   uint64_t q, cudaStream_t stream = 0);

void ewise_negate(const DevicePoly& a, DevicePoly& c,
                   uint64_t q, cudaStream_t stream = 0);

// RNS-level operations (apply to all modulus components)
void ewise_add_rns(const RnsPoly& a, const RnsPoly& b, RnsPoly& c,
                    const uint64_t* moduli, cudaStream_t stream = 0);

void ewise_sub_rns(const RnsPoly& a, const RnsPoly& b, RnsPoly& c,
                    const uint64_t* moduli, cudaStream_t stream = 0);

void ewise_mul_rns(const RnsPoly& a, const RnsPoly& b, RnsPoly& c,
                    const uint64_t* moduli, cudaStream_t stream = 0);

void ewise_mulacc_rns(const RnsPoly& a, const RnsPoly& b, RnsPoly& c,
                       const uint64_t* moduli, cudaStream_t stream = 0);

// RLWE ciphertext addition
void rlwe_add(const RlweCt& a, const RlweCt& b, RlweCt& c,
               const uint64_t* moduli, cudaStream_t stream = 0);

void rlwe_sub(const RlweCt& a, const RlweCt& b, RlweCt& c,
               const uint64_t* moduli, cudaStream_t stream = 0);

} // namespace kernel
} // namespace onionpir
