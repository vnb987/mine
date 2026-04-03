#pragma once
#include "../common/types.cuh"
#include "../crypto/context.cuh"
#include "decompose.cuh"

namespace onionpir {
namespace arith {

// ============================================================================
// Key-Switching
//
// Transforms an RLWE ciphertext encrypted under key s' into one under key s.
// Uses a key-switching key (KSK) which is a set of RLWE encryptions of
// s' * B^j under s.
//
// Process:
// 1. Decompose the 'a' component of input into base-B digits
// 2. Inner product with KSK rows
// 3. Add to 'b' component
// ============================================================================

// Key-switch: transform ct from one key to another
void key_switch(const RlweCt& ct_in, RlweCt& ct_out,
                 const KSwitchKey& ksk, const crypto::CryptoContext& ctx,
                 cudaStream_t stream = 0);

// Key-switch in-place
void key_switch_inplace(RlweCt& ct, const KSwitchKey& ksk,
                         const crypto::CryptoContext& ctx,
                         cudaStream_t stream = 0);

// Batch key-switching for multiple ciphertexts with same KSK
void key_switch_batch(const RlweCt* ct_in, RlweCt* ct_out, uint32_t count,
                       const KSwitchKey& ksk, const crypto::CryptoContext& ctx,
                       cudaStream_t stream = 0);

} // namespace arith
} // namespace onionpir
