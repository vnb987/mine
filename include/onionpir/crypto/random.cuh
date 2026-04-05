#pragma once
#include "../common/types.cuh"
#include <curand.h>
#include <curand_kernel.h>

namespace onionpir {
namespace crypto {

struct RandomContext {
    curandGenerator_t gen;
    curandState_t* d_states = nullptr;
    uint32_t num_states = 0;
    bool initialized = false;

    void init(uint64_t seed = 42);
    void destroy();
    ~RandomContext() { destroy(); }
};

__global__ void sample_uniform_kernel(uint64_t* output, curandState_t* states,
                                       uint64_t q, uint32_t len);
__global__ void sample_ternary_kernel(uint64_t* output, curandState_t* states,
                                       uint64_t q, uint32_t len);
__global__ void sample_gaussian_kernel(uint64_t* output, curandState_t* states,
                                        uint64_t q, double sigma, uint32_t len);
__global__ void sample_cbd_kernel(uint64_t* output, curandState_t* states,
                                   uint64_t q, uint32_t eta, uint32_t len);

void sample_uniform(DevicePoly& poly, uint64_t q, RandomContext& rng, cudaStream_t stream = 0);
void sample_uniform_rns(RnsPoly& poly, const uint64_t* moduli, RandomContext& rng, cudaStream_t stream = 0);
void sample_ternary(DevicePoly& poly, uint64_t q, RandomContext& rng, cudaStream_t stream = 0);
void sample_ternary_rns(RnsPoly& poly, const uint64_t* moduli, RandomContext& rng, cudaStream_t stream = 0);
void sample_gaussian(DevicePoly& poly, uint64_t q, double sigma, RandomContext& rng, cudaStream_t stream = 0);
void sample_gaussian_rns(RnsPoly& poly, const uint64_t* moduli, double sigma, RandomContext& rng, cudaStream_t stream = 0);

} // namespace crypto
} // namespace onionpir
