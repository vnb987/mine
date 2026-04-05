#include "onionpir/crypto/context.cuh"

namespace onionpir {
namespace crypto {

static uint64_t h_mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    __uint128_t result = 1, b = base % mod;
    while (exp > 0) {
        if (exp & 1) result = (result * b) % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return (uint64_t)result;
}

static uint64_t h_mod_inv(uint64_t a, uint64_t mod) {
    return h_mod_pow(a, mod - 2, mod);
}

void CryptoContext::init(const uint64_t* qs, uint32_t nmod, uint64_t seed) {
    // Use defaults if not provided
    if (qs == nullptr || nmod == 0) {
        num_moduli = DEFAULT_NUM_MODULI;
        for (uint32_t i = 0; i < num_moduli; i++) {
            moduli[i] = DEFAULT_MODULI[i];
        }
    } else {
        num_moduli = nmod;
        for (uint32_t i = 0; i < nmod; i++) {
            moduli[i] = qs[i];
        }
    }

    // Upload moduli to GPU
    CUDA_CHECK(cudaMalloc(&d_moduli, num_moduli * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_moduli, moduli, num_moduli * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));

    // Initialize NTT tables
    ntt_ctx.init(moduli, num_moduli);

    // Initialize CRT context
    crt_ctx.init(moduli, num_moduli);

    // Initialize RNG
    rng.init(seed);

    // Compute gadget powers: B^j mod q_i for each level j and modulus i
    std::vector<uint64_t> h_gadget(decomp_levels * num_moduli);
    for (uint32_t i = 0; i < num_moduli; i++) {
        uint64_t power = 1;
        for (uint32_t j = 0; j < decomp_levels; j++) {
            h_gadget[j * num_moduli + i] = power;
            power = (uint64_t)(((__uint128_t)power * decomp_base) % moduli[i]);
        }
    }
    CUDA_CHECK(cudaMalloc(&d_gadget_powers,
                           decomp_levels * num_moduli * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_gadget_powers, h_gadget.data(),
                           decomp_levels * num_moduli * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));

    // Compute plain_mod^{-1} mod q_i
    std::vector<uint64_t> h_plain_inv(num_moduli);
    for (uint32_t i = 0; i < num_moduli; i++) {
        h_plain_inv[i] = h_mod_inv(plain_mod, moduli[i]);
    }
    CUDA_CHECK(cudaMalloc(&d_plain_mod_inv, num_moduli * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_plain_mod_inv, h_plain_inv.data(),
                           num_moduli * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Compute delta = floor(q_i / t) for each modulus
    std::vector<uint64_t> h_delta(num_moduli);
    for (uint32_t i = 0; i < num_moduli; i++) {
        h_delta[i] = moduli[i] / plain_mod;
    }
    CUDA_CHECK(cudaMalloc(&d_delta, num_moduli * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_delta, h_delta.data(),
                           num_moduli * sizeof(uint64_t), cudaMemcpyHostToDevice));
}

void CryptoContext::destroy() {
    ntt_ctx.destroy();
    crt_ctx.destroy();
    rng.destroy();
    if (d_moduli) { cudaFree(d_moduli); d_moduli = nullptr; }
    if (d_gadget_powers) { cudaFree(d_gadget_powers); d_gadget_powers = nullptr; }
    if (d_plain_mod_inv) { cudaFree(d_plain_mod_inv); d_plain_mod_inv = nullptr; }
    if (d_delta) { cudaFree(d_delta); d_delta = nullptr; }
    num_moduli = 0;
}

} // namespace crypto
} // namespace onionpir
