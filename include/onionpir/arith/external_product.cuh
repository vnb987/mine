#pragma once
#include "../common/types.cuh"
#include "../crypto/context.cuh"
#include "decompose.cuh"

namespace onionpir {
namespace arith {

// ============================================================================
// External Product: RLWE ⊗ RGSW -> RLWE
//
// Given RLWE ciphertext ct = (a, b) and RGSW ciphertext C encrypting m:
// 1. Decompose ct into digits: (a_0, ..., a_{l-1}, b_0, ..., b_{l-1})
// 2. Multiply each digit by the corresponding RGSW row
// 3. Sum to get output RLWE ciphertext encrypting ct_msg * m
//
// This is THE core operation for PIR: it selects DB entries based on
// encrypted query bits.
// ============================================================================

// External product: RLWE ⊗ RGSW -> RLWE
// ct_in: input RLWE ciphertext (in NTT domain)
// rgsw: RGSW ciphertext (in NTT domain)
// ct_out: output RLWE ciphertext (in NTT domain)
void external_product(const RlweCt& ct_in, const RgswCt& rgsw,
                       RlweCt& ct_out, const crypto::CryptoContext& ctx,
                       cudaStream_t stream = 0);

// Batch external product: process multiple RLWE × same RGSW
// Useful when same query bit selects across multiple ciphertext rows
void external_product_batch(const RlweCt* ct_in, uint32_t count,
                             const RgswCt& rgsw, RlweCt* ct_out,
                             const crypto::CryptoContext& ctx,
                             cudaStream_t stream = 0);

// Fused external product + accumulate: ct_acc += ct_in ⊗ rgsw
void external_product_acc(RlweCt& ct_acc, const RlweCt& ct_in,
                           const RgswCt& rgsw,
                           const crypto::CryptoContext& ctx,
                           cudaStream_t stream = 0);

} // namespace arith
} // namespace onionpir
