#include "onionpir/crypto/encrypt.cuh"
#include "onionpir/kernel/ntt.cuh"
#include "onionpir/kernel/elementwise.cuh"
#include "onionpir/kernel/crt.cuh"

namespace onionpir {
namespace crypto {

// Kernel: encode plaintext by multiplying with delta = floor(q/t)
__global__ void encode_plaintext_kernel(const uint64_t* pt, uint64_t* output,
                                         uint64_t delta, uint64_t q, uint32_t len) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        __uint128_t val = (__uint128_t)pt[idx] * delta;
        output[idx] = (uint64_t)(val % q);
    }
}

// ============================================================================
// RLWE Encryption
// ct = (a, a*s + e + delta*m) where a is random, e is error
// ============================================================================

void encrypt_rlwe(RlweCt& ct, const RnsPoly& pt, const SecretKey& sk,
                   CryptoContext& ctx, cudaStream_t stream) {
    ct.allocate(ctx.num_moduli);

    // Sample random a
    sample_uniform_rns(ct.a, ctx.moduli, ctx.rng, stream);

    // Sample error e
    RnsPoly e;
    e.allocate(ctx.num_moduli);
    sample_gaussian_rns(e, ctx.moduli, 3.2, ctx.rng, stream);
    kernel::ntt_forward_rns(e, ctx.ntt_ctx, stream);

    // Encode plaintext: delta * m for each modulus
    RnsPoly encoded;
    encoded.allocate(ctx.num_moduli);

    uint32_t blocks = (N + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
    for (uint32_t i = 0; i < ctx.num_moduli; i++) {
        uint64_t delta = ctx.moduli[i] / ctx.plain_mod;
        encode_plaintext_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            pt.component(i), encoded.component(i), delta, ctx.moduli[i], N);
    }
    kernel::ntt_forward_rns(encoded, ctx.ntt_ctx, stream);

    // b = a*s + e + encoded_m
    kernel::ewise_mul_rns(ct.a, sk.s, ct.b, ctx.moduli, stream);
    kernel::ewise_add_rns(ct.b, e, ct.b, ctx.moduli, stream);
    kernel::ewise_add_rns(ct.b, encoded, ct.b, ctx.moduli, stream);
}

void encrypt_scalar(RlweCt& ct, uint64_t value, const SecretKey& sk,
                     CryptoContext& ctx, cudaStream_t stream) {
    ct.allocate(ctx.num_moduli);

    sample_uniform_rns(ct.a, ctx.moduli, ctx.rng, stream);

    RnsPoly e;
    e.allocate(ctx.num_moduli);
    sample_gaussian_rns(e, ctx.moduli, 3.2, ctx.rng, stream);
    kernel::ntt_forward_rns(e, ctx.ntt_ctx, stream);

    // b = a*s + e + delta*value
    kernel::ewise_mul_rns(ct.a, sk.s, ct.b, ctx.moduli, stream);
    kernel::ewise_add_rns(ct.b, e, ct.b, ctx.moduli, stream);

    // Add delta*value to all NTT slots
    uint32_t blocks = (N + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
    for (uint32_t i = 0; i < ctx.num_moduli; i++) {
        uint64_t delta = ctx.moduli[i] / ctx.plain_mod;
        uint64_t scaled = (uint64_t)(((__uint128_t)value * delta) % ctx.moduli[i]);
        kernel::ewise_add_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            ct.b.component(i), ct.b.component(i), ct.b.component(i),
            ctx.moduli[i], 0); // no-op placeholder, need proper scalar add

        // Actually add the scalar to all positions (in NTT domain, scalar = constant)
        add_scalar_ntt_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            ct.b.component(i), scaled, ctx.moduli[i], N);
    }
}

void encrypt_rgsw(RgswCt& ct, uint64_t value, const SecretKey& sk,
                   CryptoContext& ctx, cudaStream_t stream) {
    generate_rgsw(ct, value, sk, ctx, stream);
}

// ============================================================================
// Decryption
// m = round((b - a*s) * t / q)
// ============================================================================

// Kernel: compute b - a*s for one modulus component
__global__ void decrypt_phase_kernel(const uint64_t* a, const uint64_t* b,
                                      const uint64_t* s, uint64_t* output,
                                      uint64_t q, uint32_t len) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        // phase = b - a*s mod q
        __uint128_t prod = (__uint128_t)a[idx] * s[idx];
        uint64_t as_mod = (uint64_t)(prod % q);
        output[idx] = b[idx] >= as_mod ? b[idx] - as_mod : b[idx] + q - as_mod;
    }
}

void decrypt_rlwe(RnsPoly& pt, const RlweCt& ct, const SecretKey& sk,
                   CryptoContext& ctx, cudaStream_t stream) {
    pt.allocate(ctx.num_moduli);

    uint32_t blocks = (N + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

    // Compute phase = b - a*s for each modulus
    for (uint32_t i = 0; i < ctx.num_moduli; i++) {
        decrypt_phase_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
            ct.a.component(i), ct.b.component(i), sk.s.component(i),
            pt.component(i), ctx.moduli[i], N);
    }

    // Convert from NTT to coefficient form
    kernel::ntt_inverse_rns(pt, ctx.ntt_ctx, stream);

    // Round: m = round(phase * t / q)
    // Use first modulus for simplicity (single-CRT case)
    kernel::ewise_round_scale_kernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
        pt.component(0), pt.component(0), ctx.moduli[0], ctx.plain_mod, N);
}

void decrypt_to_host(std::vector<uint64_t>& pt_host, const RlweCt& ct,
                      const SecretKey& sk, CryptoContext& ctx,
                      cudaStream_t stream) {
    RnsPoly pt;
    decrypt_rlwe(pt, ct, sk, ctx, stream);

    pt_host.resize(N);
    CUDA_CHECK(cudaMemcpyAsync(pt_host.data(), pt.component(0),
                                N * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

double noise_budget(const RlweCt& ct, const SecretKey& sk,
                     CryptoContext& ctx) {
    // Compute phase = b - a*s in coefficient form
    // Then measure noise: |phase - round(phase * t/q) * q/t|
    std::vector<uint64_t> phase(N);

    RnsPoly pt;
    pt.allocate(ctx.num_moduli);

    uint32_t blocks = (N + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
    for (uint32_t i = 0; i < ctx.num_moduli; i++) {
        decrypt_phase_kernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
            ct.a.component(i), ct.b.component(i), sk.s.component(i),
            pt.component(i), ctx.moduli[i], N);
    }
    kernel::ntt_inverse_rns(pt, ctx.ntt_ctx);

    CUDA_CHECK(cudaMemcpy(phase.data(), pt.component(0),
                           N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    uint64_t q = ctx.moduli[0];
    uint64_t t = ctx.plain_mod;
    double max_noise = 0;
    for (uint32_t i = 0; i < N; i++) {
        double val = (double)phase[i];
        double decoded = round(val * (double)t / (double)q);
        double reencoded = decoded * (double)q / (double)t;
        double noise = fabs(val - reencoded);
        if (noise > (double)q / 2) noise = (double)q - noise;
        if (noise > max_noise) max_noise = noise;
    }

    // Noise budget in bits
    return log2((double)q / (2.0 * max_noise));
}

} // namespace crypto
} // namespace onionpir
