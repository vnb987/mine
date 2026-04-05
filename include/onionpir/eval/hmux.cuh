#pragma once
#include "../common/types.cuh"
#include "../crypto/context.cuh"
#include "../arith/external_product.cuh"

namespace onionpir {
namespace eval {

// ============================================================================
// HMUX (Homomorphic MUX)
//
// HMUX(sel, ct0, ct1) = sel ? ct1 : ct0
// = ct0 + sel * (ct1 - ct0)
// = ct0 + ExternalProduct(ct1 - ct0, RGSW(sel))
//
// Where sel is an RGSW ciphertext encrypting 0 or 1.
//
// In OnionPIR, HMUX is the fundamental operation for the tree-based
// database traversal in dimensions beyond the first.
// For a binary tree of depth d, we need d HMUX operations to select
// one leaf out of 2^d.
// ============================================================================

// Single HMUX: select between ct0 and ct1 based on encrypted selector
// selector: RGSW ciphertext encrypting 0 or 1
// ct0: "false" branch (selected when selector=0)
// ct1: "true" branch (selected when selector=1)
// ct_out: result = ct0 + selector * (ct1 - ct0)
void hmux(const RlweCt& ct0, const RlweCt& ct1, const RgswCt& selector,
           RlweCt& ct_out, const crypto::CryptoContext& ctx,
           cudaStream_t stream = 0);

// In-place HMUX: ct0 = HMUX(selector, ct0, ct1)
void hmux_inplace(RlweCt& ct0, const RlweCt& ct1, const RgswCt& selector,
                   const crypto::CryptoContext& ctx,
                   cudaStream_t stream = 0);

// Tree HMUX: traverse a binary tree of depth `depth` using `selectors`
// items: 2^depth RLWE ciphertexts (or plaintext-encoded DB rows)
// selectors: `depth` RGSW ciphertexts (query bits)
// result: the selected item
//
// Algorithm (bottom-up):
// For level l from depth-1 down to 0:
//   For each pair (items[2i], items[2i+1]):
//     items[i] = HMUX(selectors[l], items[2i], items[2i+1])
// Final result is items[0]
void tree_hmux(std::vector<RlweCt>& items, const std::vector<RgswCt>& selectors,
                RlweCt& result, const crypto::CryptoContext& ctx,
                cudaStream_t stream = 0);

// Batch tree HMUX: process multiple independent trees simultaneously
// Useful for batching multiple PIR queries
void tree_hmux_batch(std::vector<std::vector<RlweCt>>& item_sets,
                      const std::vector<std::vector<RgswCt>>& selector_sets,
                      std::vector<RlweCt>& results,
                      const crypto::CryptoContext& ctx,
                      StreamPool& streams);

} // namespace eval
} // namespace onionpir
