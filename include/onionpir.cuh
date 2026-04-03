#pragma once

// ============================================================================
// OnionPIRv2 CUDA Implementation
//
// A high-performance CUDA implementation of the OnionPIR v2 Private
// Information Retrieval protocol, organized in 5 hierarchical layers:
//
// Layer 0 - Kernel:  GPU primitives (NTT, CRT, element-wise ops)
// Layer 1 - Crypto:  Cryptographic context (keygen, encrypt, decrypt, RNG)
// Layer 2 - Arith:   Polynomial operations (key-switch, external product, decompose)
// Layer 3 - Eval:    HE operations (HMUX, automorphism, ciphertext expansion)
// Layer 4 - Module:  PIR protocol (database, query processing, batch PIR)
//
// Hierarchical Data Structure:
//   Level 0 (Kernel):  DevicePoly      - raw GPU polynomial (N uint64_t)
//   Level 1 (Crypto):  RnsPoly         - RNS representation (multi-modulus)
//   Level 2 (Arith):   RlweCt / RgswCt - HE ciphertexts
//   Level 3 (Eval):    EvalKeySet      - evaluation key collections
//   Level 4 (Module):  PirDb / PirQuery / PirResponse / PirBatch
//
// Each layer only depends on the layer directly below, isolating changes.
// ============================================================================

// Common types and parameters
#include "onionpir/common/params.h"
#include "onionpir/common/types.cuh"

// Layer 0: Kernel
#include "onionpir/kernel/ntt.cuh"
#include "onionpir/kernel/crt.cuh"
#include "onionpir/kernel/elementwise.cuh"

// Layer 1: Crypto
#include "onionpir/crypto/random.cuh"
#include "onionpir/crypto/context.cuh"
#include "onionpir/crypto/keygen.cuh"
#include "onionpir/crypto/encrypt.cuh"

// Layer 2: Arith
#include "onionpir/arith/decompose.cuh"
#include "onionpir/arith/external_product.cuh"
#include "onionpir/arith/keyswitch.cuh"

// Layer 3: Eval
#include "onionpir/eval/hmux.cuh"
#include "onionpir/eval/automorphism.cuh"

// Layer 4: Module
#include "onionpir/module/database.cuh"
#include "onionpir/module/pir.cuh"
#include "onionpir/module/batch.cuh"
