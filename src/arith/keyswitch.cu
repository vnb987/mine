#include "onionpir/arith/keyswitch.cuh"
#include "onionpir/kernel/ntt.cuh"
#include "onionpir/kernel/elementwise.cuh"

namespace onionpir {
namespace arith {

// ============================================================================
// Key-Switching Implementation
//
// Input: RLWE ct = (a, b) under key s'
// Output: RLWE ct' = (a', b') under key s
//
// Using key-switching key KSK = {RLWE_s(s'*B^j)} for j = 0..l-1
//
// Algorithm:
// 1. Decompose a into base-B digits: a = sum_j a_j * B^j
// 2. For each level j:
//      ct' += a_j * KSK[j]
// 3. ct'.b += b  (the b component passes through directly)
// ============================================================================

void key_switch(const RlweCt& ct_in, RlweCt& ct_out,
                 const KSwitchKey& ksk, const crypto::CryptoContext& ctx,
                 cudaStream_t stream) {
    uint32_t levels = ctx.decomp_levels;
    uint32_t nmod = ctx.num_moduli;
    uint32_t blocks = (N + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    // Convert a to coefficient form for decomposition
    RnsPoly a_coeff;
    a_coeff.allocate(nmod);
    for (uint32_t m = 0; m < nmod; m++) {
        CUDA_CHECK(cudaMemcpyAsync(a_coeff.component(m), ct_in.a.component(m),
                                    N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    }
    kernel::ntt_inverse_rns(a_coeff, ctx.ntt_ctx, stream);

    // Decompose a into digits
    DecompOutput a_decomp;
    gadget_decompose_ntt(a_coeff, a_decomp, ctx, stream);

    // Initialize output
    ct_out.allocate(nmod);
    ct_out.zero(stream);

    // Accumulate: ct_out += sum_j a_j * KSK[j]
    for (uint32_t j = 0; j < levels; j++) {
        for (uint32_t m = 0; m < nmod; m++) {
            // ct_out.a += a_j * ksk[j].a
            kernel::ewise_mulacc_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
                a_decomp.component(j, m), ksk.component(j, 0, m),
                ct_out.a.component(m), ctx.moduli[m], N);
            // ct_out.b += a_j * ksk[j].b
            kernel::ewise_mulacc_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
                a_decomp.component(j, m), ksk.component(j, 1, m),
                ct_out.b.component(m), ctx.moduli[m], N);
        }
    }

    // Add original b component
    kernel::ewise_add_rns(ct_out.b, ct_in.b, ct_out.b, ctx.moduli, stream);
}

void key_switch_inplace(RlweCt& ct, const KSwitchKey& ksk,
                         const crypto::CryptoContext& ctx,
                         cudaStream_t stream) {
    RlweCt temp;
    key_switch(ct, temp, ksk, ctx, stream);

    // Copy back
    for (uint32_t m = 0; m < ctx.num_moduli; m++) {
        CUDA_CHECK(cudaMemcpyAsync(ct.a.component(m), temp.a.component(m),
                                    N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ct.b.component(m), temp.b.component(m),
                                    N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    }
}

void key_switch_batch(const RlweCt* ct_in, RlweCt* ct_out, uint32_t count,
                       const KSwitchKey& ksk, const crypto::CryptoContext& ctx,
                       cudaStream_t stream) {
    for (uint32_t i = 0; i < count; i++) {
        key_switch(ct_in[i], ct_out[i], ksk, ctx, stream);
    }
}

} // namespace arith
} // namespace onionpir
