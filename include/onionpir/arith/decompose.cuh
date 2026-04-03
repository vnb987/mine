#pragma once
#include "../common/types.cuh"
#include "../crypto/context.cuh"

namespace onionpir {
namespace arith {

// ============================================================================
// Gadget / Bit Decomposition
// Decomposes a polynomial into base-B digits for RGSW external product.
//
// Given polynomial a(X) with coefficients in Z_q, compute
// a_0(X), a_1(X), ..., a_{l-1}(X) such that
// a(X) = sum_j a_j(X) * B^j
// where coefficients of a_j are in [0, B).
//
// This is the inverse of the gadget multiplication.
// ============================================================================

// ---- Device kernels --------------------------------------------------------

// Decompose a single polynomial: coefficient-wise base-B decomposition
// input: N coefficients mod q
// output: decomp_levels * N coefficients (each in [0, B))
// Works in coefficient form (requires iNTT before, NTT after)
__global__ void gadget_decompose_kernel(const uint64_t* input, uint64_t* output,
                                         uint32_t decomp_levels,
                                         uint32_t base_log, uint64_t base,
                                         uint64_t q, uint32_t n);

// Signed/balanced decomposition: digits in [-B/2, B/2)
// Reduces noise growth in external product
__global__ void gadget_decompose_signed_kernel(const uint64_t* input, uint64_t* output,
                                                uint32_t decomp_levels,
                                                uint32_t base_log, uint64_t base,
                                                uint64_t q, uint32_t n);

// ---- Host wrappers ---------------------------------------------------------

// Decompose an RnsPoly into decomp_levels RnsPolys
// Input must be in coefficient form
// Output polynomials are in coefficient form, need NTT afterwards
struct DecompOutput {
    DevicePolyArray data;  // decomp_levels * num_moduli polynomials
    uint32_t decomp_levels = 0;
    uint32_t num_moduli = 0;

    void allocate(uint32_t levels, uint32_t nmod) {
        decomp_levels = levels;
        num_moduli = nmod;
        data.allocate(levels * nmod);
    }

    // Get decomposed polynomial for level j, modulus m
    uint64_t* component(uint32_t j, uint32_t m) {
        return data.poly(j * num_moduli + m);
    }
    const uint64_t* component(uint32_t j, uint32_t m) const {
        return data.poly(j * num_moduli + m);
    }
};

// Decompose a polynomial (in coeff form) into base-B digits
void gadget_decompose(const RnsPoly& input, DecompOutput& output,
                       const crypto::CryptoContext& ctx,
                       cudaStream_t stream = 0);

// Decompose with signed/balanced representation
void gadget_decompose_signed(const RnsPoly& input, DecompOutput& output,
                              const crypto::CryptoContext& ctx,
                              cudaStream_t stream = 0);

// Decompose and convert to NTT form in one step
void gadget_decompose_ntt(const RnsPoly& input, DecompOutput& output,
                           const crypto::CryptoContext& ctx,
                           cudaStream_t stream = 0);

} // namespace arith
} // namespace onionpir
