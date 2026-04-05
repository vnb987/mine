#pragma once
#include "../common/types.cuh"

namespace onionpir {
namespace kernel {

// ============================================================================
// CRT (Chinese Remainder Theorem) CUDA Kernels
// Converts between RNS representation and single big-integer representation.
// Used for operations that need full coefficient values (e.g., rounding, scaling).
// ============================================================================

// CRT reconstruction context
struct CRTContext {
    // For k moduli q0, q1, ..., q_{k-1}:
    // Q = product of all q_i
    // Q_i = Q / q_i
    // Q_i_inv = Q_i^{-1} mod q_i
    uint64_t* d_q_i_inv = nullptr;     // Q_i^{-1} mod q_i for each i, on GPU
    uint64_t* d_moduli = nullptr;       // moduli array on GPU
    uint64_t* d_q_hat = nullptr;        // Q/q_i mod output_mod for each i
    uint32_t num_moduli = 0;

    void init(const uint64_t* moduli, uint32_t count);
    void destroy();
    ~CRTContext() { destroy(); }
};

// ---- Device kernels --------------------------------------------------------

// CRT compose: RNS -> single value mod target
// input: k * N values (k polynomials of N coefficients each)
// output: N values (single polynomial in target modulus)
__global__ void crt_compose_kernel(const uint64_t* input, uint64_t* output,
                                    const uint64_t* q_i_inv,
                                    const uint64_t* moduli,
                                    const uint64_t* q_hat,
                                    uint32_t num_moduli,
                                    uint64_t target_mod);

// CRT decompose (inverse CRT): single value -> RNS
// input: N values in big representation
// output: k * N values (one poly per modulus)
__global__ void crt_decompose_kernel(const uint64_t* input, uint64_t* output,
                                      const uint64_t* moduli,
                                      uint32_t num_moduli);

// Approximate CRT compose for fast base conversion (used in key-switching)
// Outputs centered representation to avoid full big-integer arithmetic
__global__ void crt_compose_approx_kernel(const uint64_t* input, uint64_t* output,
                                           const uint64_t* q_i_inv,
                                           const uint64_t* moduli,
                                           uint32_t num_moduli,
                                           uint64_t target_mod);

// ---- Host wrappers ---------------------------------------------------------

// Compose RnsPoly into a single-modulus polynomial
void crt_compose(const RnsPoly& input, DevicePoly& output,
                  const CRTContext& ctx, uint64_t target_mod,
                  cudaStream_t stream = 0);

// Decompose a single polynomial into RNS form
void crt_decompose(const DevicePoly& input, RnsPoly& output,
                    const CRTContext& ctx, cudaStream_t stream = 0);

// Fast approximate base conversion for key-switching
void crt_base_convert(const RnsPoly& input, RnsPoly& output,
                       const CRTContext& src_ctx, const CRTContext& dst_ctx,
                       cudaStream_t stream = 0);

} // namespace kernel
} // namespace onionpir
