#pragma once
#include "../common/types.cuh"
#include "../crypto/context.cuh"
#include "../arith/external_product.cuh"

namespace onionpir {
namespace eval {

__global__ void apply_automorphism_ntt_kernel(const uint64_t* input, uint64_t* output,
                                               const uint32_t* permutation,
                                               uint64_t q, uint32_t n);

struct AutoPermutation {
    uint32_t* d_perm = nullptr;
    uint32_t galois_elt = 0;
    void init(uint32_t gelt);
    void destroy();
    ~AutoPermutation() { destroy(); }
};

void apply_automorphism(const RlweCt& ct_in, RlweCt& ct_out,
                         uint32_t galois_elt, const AutoPermutation& perm,
                         const uint64_t* moduli, uint32_t num_moduli,
                         cudaStream_t stream = 0);

void automorphism_keyed(const RlweCt& ct_in, RlweCt& ct_out,
                         const AutoKey& akey, const AutoPermutation& perm,
                         const crypto::CryptoContext& ctx,
                         cudaStream_t stream = 0);

void ciphertext_expand(const RlweCt& ct_in, std::vector<RlweCt>& ct_out,
                        uint32_t num_outputs,
                        const EvalKeySet& eval_keys,
                        const crypto::CryptoContext& ctx,
                        cudaStream_t stream = 0);

} // namespace eval
} // namespace onionpir
