#include "onionpir/arith/decompose.cuh"
#include "onionpir/kernel/ntt.cuh"

namespace onionpir {
namespace arith {

// ============================================================================
// Gadget Decomposition Kernels
// ============================================================================

// Unsigned decomposition: extract base-B digits
__global__ void gadget_decompose_kernel(const uint64_t* input, uint64_t* output,
                                         uint32_t decomp_levels,
                                         uint32_t base_log, uint64_t base,
                                         uint64_t q, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t val = input[idx];
    for (uint32_t j = 0; j < decomp_levels; j++) {
        // Extract j-th digit in base B
        output[(size_t)j * n + idx] = val & (base - 1);  // base is power of 2
        val >>= base_log;
    }
}

// Signed/balanced decomposition: digits in [-B/2, B/2)
// This reduces noise in the external product
__global__ void gadget_decompose_signed_kernel(const uint64_t* input, uint64_t* output,
                                                uint32_t decomp_levels,
                                                uint32_t base_log, uint64_t base,
                                                uint64_t q, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t val = input[idx];
    uint64_t half_base = base >> 1;
    uint64_t mask = base - 1;

    for (uint32_t j = 0; j < decomp_levels; j++) {
        uint64_t digit = val & mask;

        if (digit >= half_base) {
            // Signed: digit - base, stored as q - (base - digit)
            output[(size_t)j * n + idx] = q - (base - digit);
            val >>= base_log;
            val += 1; // carry
        } else {
            output[(size_t)j * n + idx] = digit;
            val >>= base_log;
        }
    }
}

// ============================================================================
// Host wrappers
// ============================================================================

void gadget_decompose(const RnsPoly& input, DecompOutput& output,
                       const crypto::CryptoContext& ctx,
                       cudaStream_t stream) {
    uint32_t levels = ctx.decomp_levels;
    uint32_t nmod = ctx.num_moduli;
    output.allocate(levels, nmod);

    uint32_t blocks = (N + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    for (uint32_t m = 0; m < nmod; m++) {
        gadget_decompose_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            input.component(m), output.component(0, m),
            levels, ctx.decomp_base_log, ctx.decomp_base, ctx.moduli[m], N);
    }
}

void gadget_decompose_signed(const RnsPoly& input, DecompOutput& output,
                              const crypto::CryptoContext& ctx,
                              cudaStream_t stream) {
    uint32_t levels = ctx.decomp_levels;
    uint32_t nmod = ctx.num_moduli;
    output.allocate(levels, nmod);

    uint32_t blocks = (N + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    for (uint32_t m = 0; m < nmod; m++) {
        gadget_decompose_signed_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            input.component(m), output.component(0, m),
            levels, ctx.decomp_base_log, ctx.decomp_base, ctx.moduli[m], N);
    }
}

void gadget_decompose_ntt(const RnsPoly& input, DecompOutput& output,
                           const crypto::CryptoContext& ctx,
                           cudaStream_t stream) {
    // First decompose in coefficient form
    gadget_decompose_signed(input, output, ctx, stream);

    // Then NTT each decomposed polynomial
    for (uint32_t j = 0; j < ctx.decomp_levels; j++) {
        for (uint32_t m = 0; m < ctx.num_moduli; m++) {
            DevicePoly dp;
            dp.view(output.component(j, m));
            kernel::ntt_forward(dp, ctx.ntt_ctx.tables[m], stream);
        }
    }
}

} // namespace arith
} // namespace onionpir
