#pragma once
#include "context.cuh"

namespace onionpir {
namespace crypto {

// ============================================================================
// Key Generation
// ============================================================================

struct SecretKey {
    RnsPoly s;
    void generate(CryptoContext& ctx, cudaStream_t stream = 0);
};

struct PublicKey {
    RlweCt pk;
    void generate(const SecretKey& sk, CryptoContext& ctx, cudaStream_t stream = 0);
};

void generate_rgsw(RgswCt& out, uint64_t value, const SecretKey& sk,
                    CryptoContext& ctx, cudaStream_t stream = 0);

void generate_auto_key(AutoKey& out, uint32_t galois_elt,
                        const SecretKey& sk, CryptoContext& ctx,
                        cudaStream_t stream = 0);

void generate_eval_keys(EvalKeySet& out, const SecretKey& sk,
                         CryptoContext& ctx, cudaStream_t stream = 0);

void generate_kswitch_key(KSwitchKey& out, const SecretKey& sk_from,
                           const SecretKey& sk_to, CryptoContext& ctx,
                           cudaStream_t stream = 0);

} // namespace crypto
} // namespace onionpir
