#include "onionpir/arith/external_product.cuh"
#include "onionpir/kernel/ntt.cuh"
#include "onionpir/kernel/elementwise.cuh"

namespace onionpir {
namespace arith {

// ============================================================================
// External Product: RLWE ⊗ RGSW -> RLWE
//
// Algorithm:
// 1. Convert RLWE ct = (a, b) from NTT to coefficient form
// 2. Gadget-decompose a -> (a_0, ..., a_{l-1}) and b -> (b_0, ..., b_{l-1})
// 3. For each level j:
//      - Multiply a_j by RGSW row j (top half, encrypts m*B^j)
//      - Multiply b_j by RGSW row l+j (bottom half, encrypts m*s*B^j)
//      - Accumulate into output
// 4. Output is in NTT domain
// ============================================================================

void external_product(const RlweCt& ct_in, const RgswCt& rgsw,
                       RlweCt& ct_out, const crypto::CryptoContext& ctx,
                       cudaStream_t stream) {
    uint32_t levels = ctx.decomp_levels;
    uint32_t nmod = ctx.num_moduli;
    uint32_t blocks = (N + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    // Step 1: Convert input RLWE to coefficient form for decomposition
    RnsPoly a_coeff, b_coeff;
    a_coeff.allocate(nmod);
    b_coeff.allocate(nmod);

    // Copy and inverse NTT
    for (uint32_t m = 0; m < nmod; m++) {
        CUDA_CHECK(cudaMemcpyAsync(a_coeff.component(m), ct_in.a.component(m),
                                    N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(b_coeff.component(m), ct_in.b.component(m),
                                    N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    }
    kernel::ntt_inverse_rns(a_coeff, ctx.ntt_ctx, stream);
    kernel::ntt_inverse_rns(b_coeff, ctx.ntt_ctx, stream);

    // Step 2: Gadget-decompose both a and b, then NTT
    DecompOutput a_decomp, b_decomp;
    gadget_decompose_ntt(a_coeff, a_decomp, ctx, stream);
    gadget_decompose_ntt(b_coeff, b_decomp, ctx, stream);

    // Step 3: Initialize output to zero
    ct_out.allocate(nmod);
    ct_out.zero(stream);

    // Step 4: Accumulate: ct_out += sum_j (a_j * RGSW[j] + b_j * RGSW[l+j])
    for (uint32_t j = 0; j < levels; j++) {
        for (uint32_t m = 0; m < nmod; m++) {
            // a_j contributes to output via top-half RGSW rows
            // RGSW row j: (rgsw_a, rgsw_b) where rgsw_b encrypts m*B^j
            // ct_out.a += a_j * rgsw[j].a
            kernel::ewise_mulacc_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
                a_decomp.component(j, m), rgsw.component(j, 0, m),
                ct_out.a.component(m), ctx.moduli[m], N);
            // ct_out.b += a_j * rgsw[j].b
            kernel::ewise_mulacc_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
                a_decomp.component(j, m), rgsw.component(j, 1, m),
                ct_out.b.component(m), ctx.moduli[m], N);

            // b_j contributes via bottom-half RGSW rows
            // ct_out.a += b_j * rgsw[l+j].a
            kernel::ewise_mulacc_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
                b_decomp.component(j, m), rgsw.component(levels + j, 0, m),
                ct_out.a.component(m), ctx.moduli[m], N);
            // ct_out.b += b_j * rgsw[l+j].b
            kernel::ewise_mulacc_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
                b_decomp.component(j, m), rgsw.component(levels + j, 1, m),
                ct_out.b.component(m), ctx.moduli[m], N);
        }
    }
}

void external_product_batch(const RlweCt* ct_in, uint32_t count,
                             const RgswCt& rgsw, RlweCt* ct_out,
                             const crypto::CryptoContext& ctx,
                             cudaStream_t stream) {
    // Process each RLWE ciphertext independently
    // Could be parallelized with multiple streams for better GPU utilization
    for (uint32_t i = 0; i < count; i++) {
        external_product(ct_in[i], rgsw, ct_out[i], ctx, stream);
    }
}

void external_product_acc(RlweCt& ct_acc, const RlweCt& ct_in,
                           const RgswCt& rgsw,
                           const crypto::CryptoContext& ctx,
                           cudaStream_t stream) {
    RlweCt temp;
    external_product(ct_in, rgsw, temp, ctx, stream);

    // Accumulate into ct_acc
    kernel::rlwe_add(ct_acc, temp, ct_acc, ctx.moduli, stream);
}

} // namespace arith
} // namespace onionpir
