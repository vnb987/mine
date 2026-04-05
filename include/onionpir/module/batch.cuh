#pragma once
#include "../common/types.cuh"
#include "../crypto/context.cuh"
#include "pir.cuh"
#include "database.cuh"

namespace onionpir {
namespace module {

// ============================================================================
// Batch PIR Processing
//
// Processes multiple PIR queries simultaneously to amortize:
// 1. DB access overhead: load DB chunks once, process all queries
// 2. NTT computation: batch NTT across queries
// 3. GPU kernel launch overhead: fuse operations across queries
// 4. Memory transfer overhead: pipeline transfers with computation
//
// Strategy:
// - First dimension: batch matrix-vector products
//   Instead of processing each query independently against the DB,
//   batch queries into a matrix and compute DB * [q1|q2|...|qB]
//   This loads each DB chunk once and multiplies with all query vectors.
//
// - Second dimension: parallel HMUX trees
//   Process HMUX trees for different queries on different CUDA streams.
//
// For a batch of B queries, the first dimension cost is:
//   ~same as 1 query (dominated by DB read) instead of B * single_query_cost
// This is the key optimization for high-throughput PIR.
// ============================================================================

// Batch configuration
struct BatchConfig {
    uint32_t max_batch_size = MAX_BATCH_SIZE;
    uint32_t num_streams = 16;              // CUDA streams for parallelism
    bool pipeline_db_access = true;          // overlap DB transfer with compute
    bool fuse_first_dim = true;              // batch first-dimension processing
    size_t gpu_memory_budget = 0;            // 0 = auto-detect
};

// ============================================================================
// Batch PIR Processor
// ============================================================================
class BatchProcessor {
public:
    // Initialize with server, DB, and configuration
    void init(PirServer& server, PirDb& db,
              crypto::CryptoContext& ctx, const BatchConfig& config = {});

    // Submit a query to the batch
    // Returns a query handle for later retrieval
    uint32_t submit_query(PirQuery&& query);

    // Process all submitted queries as a batch
    // Returns when all queries are processed
    void process_batch();

    // Retrieve the response for a given query handle
    PirResponse get_response(uint32_t handle);

    // Get batch statistics
    struct BatchStats {
        double total_time_ms = 0;
        double first_dim_time_ms = 0;
        double second_dim_time_ms = 0;
        double db_transfer_time_ms = 0;
        uint32_t num_queries = 0;
        double throughput_qps = 0;      // queries per second
        double amortized_time_ms = 0;   // per-query time
    };
    BatchStats get_stats() const { return stats_; }

    void destroy();
    ~BatchProcessor() { destroy(); }

private:
    PirServer* server_ = nullptr;
    PirDb* db_ = nullptr;
    crypto::CryptoContext* ctx_ = nullptr;
    BatchConfig config_;
    StreamPool streams_;
    BatchStats stats_;

    std::vector<PirQuery> queries_;
    std::vector<PirResponse> responses_;

    // Batched first-dimension processing
    // Loads each DB chunk once and multiplies with all query vectors
    void process_first_dim_batch(
        std::vector<std::vector<RlweCt>>& all_column_results);

    // Process first dim for a single DB chunk against all queries
    void process_chunk_batch(const DbChunk& chunk, uint32_t chunk_offset,
                              std::vector<std::vector<RlweCt>>& results);

    // Parallel second-dimension processing
    void process_second_dim_batch(
        std::vector<std::vector<RlweCt>>& column_results);
};

// ============================================================================
// Streaming Batch Processor
// For databases too large to fit in GPU memory.
// Streams DB chunks from host memory while processing previous chunks.
// Uses double-buffering to overlap transfer and compute.
// ============================================================================
class StreamingBatchProcessor {
public:
    void init(PirServer& server, PirDb& db,
              crypto::CryptoContext& ctx, const BatchConfig& config = {});

    // Submit queries and process with DB streaming
    void submit_and_process(std::vector<PirQuery>& queries);

    // Get all responses
    std::vector<PirResponse>& get_responses() { return responses_; }

private:
    PirServer* server_ = nullptr;
    PirDb* db_ = nullptr;
    crypto::CryptoContext* ctx_ = nullptr;
    BatchConfig config_;

    // Double-buffered DB chunks for pipelining
    DbChunk chunk_buffers_[2];
    cudaStream_t transfer_stream_ = 0;
    cudaStream_t compute_stream_ = 0;

    std::vector<PirResponse> responses_;
};

} // namespace module
} // namespace onionpir
