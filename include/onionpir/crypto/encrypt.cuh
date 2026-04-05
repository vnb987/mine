#pragma once
#include "context.cuh"
#include "keygen.cuh"

namespace onionpir {
namespace crypto {

// ============================================================================
// Encryption & Decryption
// RLWE symmetric encryption using secret key.
// Plaintext is encoded as m * delta where delta = floor(q/t).
// ============================================================================

// Encrypt a plaintext polynomial under secret key
void encrypt_rlwe(RlweCt& ct, const RnsPoly& pt, const SecretKey& sk,
                   CryptoContext& ctx, cudaStream_t stream = 0);

// Encrypt a scalar value into RLWE ciphertext
void encrypt_scalar(RlweCt& ct, uint64_t value, const SecretKey& sk,
                     CryptoContext& ctx, cudaStream_t stream = 0);

// Encrypt a scalar into RGSW ciphertext
void encrypt_rgsw(RgswCt& ct, uint64_t value, const SecretKey& sk,
                   CryptoContext& ctx, cudaStream_t stream = 0);

// Decrypt RLWE ciphertext to recover plaintext
void decrypt_rlwe(RnsPoly& pt, const RlweCt& ct, const SecretKey& sk,
                   CryptoContext& ctx, cudaStream_t stream = 0);

// Decrypt to host vector
void decrypt_to_host(std::vector<uint64_t>& pt_host, const RlweCt& ct,
                      const SecretKey& sk, CryptoContext& ctx,
                      cudaStream_t stream = 0);

// Noise budget estimation (for debugging)
double noise_budget(const RlweCt& ct, const SecretKey& sk,
                     CryptoContext& ctx);

} // namespace crypto
} // namespace onionpir
