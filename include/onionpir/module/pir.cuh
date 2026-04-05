#pragma once
#include "../common/types.cuh"
#include "../crypto/context.cuh"
#include "../crypto/keygen.cuh"
#include "../eval/hmux.cuh"
#include "../eval/automorphism.cuh"
#include "database.cuh"

namespace onionpir {
namespace module {

// ============================================================================
// PIR Protocol Implementation
//
// OnionPIRv2 uses a 2-dimensional PIR scheme:
//
// Dimension 1 (SimplePIR-style):
//   - Client sends an RLWE ciphertext encoding a unit vector e_i
//   - Server computes DB * e_i as a matrix-vector product
//   - This selects the i-th column of the DB matrix
//   - Result: N_dim2 RLWE ciphertexts, one per second-dimension entry
//
// Dimension 2 (HMUX-based):
//   - Client sends RGSW ciphertexts encoding index bits
//   - Server uses tree HMUX to select one entry from the column
//   - Result: single RLWE ciphertext encrypting the desired entry
//
// The first dimension is compute-intensive (matrix-vector multiply)
// and benefits most from GPU parallelism.
// ============================================================================

// ============================================================================
// PIR Server
// ============================================================================
class PirServer {
public:
    // Initialize server with database and crypto context
    void init(PirDb& db, crypto::CryptoContext& ctx);

    // Process a single PIR query
    PirResponse process_query(const PirQuery& query);

    // Set evaluation keys (received from client)
    void set_eval_keys(EvalKeySet&& keys);

    // Get server-side state for client's hint computation
    // (In SimplePIR, client needs DB digest for correctness)
    void get_hint(std::vector<uint64_t>& hint);

private:
    PirDb* db_ = nullptr;
    crypto::CryptoContext* ctx_ = nullptr;
    EvalKeySet eval_keys_;
    StreamPool streams_;
    bool initialized_ = false;

    // First dimension: matrix-vector product DB * query_vec
    void process_first_dim(const RlweCt& query_ct,
                            std::vector<RlweCt>& results,
                            cudaStream_t stream);

    // Second dimension: HMUX tree selection
    void process_second_dim(std::vector<RlweCt>& column,
                             const std::vector<RgswCt>& selectors,
                             RlweCt& result, cudaStream_t stream);
};

// ============================================================================
// PIR Client
// ============================================================================
class PirClient {
public:
    void init(crypto::CryptoContext& ctx);

    // Generate keys for PIR
    void generate_keys();

    // Create PIR query for a given index
    PirQuery create_query(size_t index);

    // Decode PIR response to recover the entry
    std::vector<uint8_t> decode_response(const PirResponse& response);

    // Get evaluation keys to send to server
    EvalKeySet& get_eval_keys() { return eval_keys_; }

private:
    crypto::CryptoContext* ctx_ = nullptr;
    crypto::SecretKey sk_;
    EvalKeySet eval_keys_;
    uint32_t dim1_ = 0;
    uint32_t dim2_ = 0;
};

} // namespace module
} // namespace onionpir
