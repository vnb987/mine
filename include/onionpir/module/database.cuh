#pragma once
#include "../common/types.cuh"
#include "../crypto/context.cuh"
#include "../kernel/ntt.cuh"

namespace onionpir {
namespace module {

// ============================================================================
// Database Management for Large-Scale PIR
//
// Supports databases up to hundreds of GB by:
// 1. Streaming chunks from host/disk to GPU
// 2. Pre-encoding entries into NTT polynomial form
// 3. Organizing data in a multi-dimensional hypercube layout
// 4. Memory-mapped file I/O for large DBs
//
// Database layout:
// - First dimension (dim1): N entries processed via SimplePIR (matrix-vector product)
// - Second dimension (dim2): entries processed via HMUX tree
// - Total entries = dim1 * dim2
//
// Each entry is encoded as one or more NTT polynomials.
// For entry_bytes=256 and plain_mod=2^20, each entry needs
// ceil(256 * 8 / 20) ≈ 103 polynomial coefficients, fitting in 1 poly of degree 4096.
// ============================================================================

// Database configuration
struct DbConfig {
    size_t num_entries = 0;
    size_t entry_bytes = DB_ENTRY_BYTES;
    uint32_t dim1 = 0;              // first dimension size (≤ N)
    uint32_t dim2 = 0;              // second dimension size (power of 2)
    uint32_t polys_per_entry = 0;   // polynomials per DB entry
    uint32_t num_moduli = 0;
    bool streaming = false;          // true if DB doesn't fit in GPU memory

    // Compute dimensions and layout from num_entries
    void compute_layout(size_t n_entries, size_t entry_size, uint32_t nmod);
};

// ============================================================================
// Database Builder
// Encodes raw database bytes into NTT-form polynomials on GPU.
// Supports streaming for large databases.
// ============================================================================
class DatabaseBuilder {
public:
    // Initialize builder with configuration
    void init(const DbConfig& config, crypto::CryptoContext& ctx);

    // Add entries from host memory
    // Can be called multiple times for streaming large DBs
    void add_entries(const uint8_t* data, size_t num_entries, size_t offset = 0);

    // Finalize: convert all added entries to NTT form
    void finalize();

    // Get the built database
    PirDb& get_db() { return db_; }

    // Memory-mapped loading from file
    void load_from_file(const char* path, size_t num_entries, size_t entry_bytes);

    ~DatabaseBuilder();

private:
    PirDb db_;
    DbConfig config_;
    crypto::CryptoContext* ctx_ = nullptr;
    cudaStream_t build_stream_ = 0;

    // Encode a batch of raw entries into polynomial coefficients
    void encode_entries(const uint8_t* raw_data, size_t count,
                         uint64_t* d_output, size_t output_offset);

    // Convert encoded polynomials to NTT form
    void ntt_encode_chunk(DbChunk& chunk);
};

// ---- Device kernels for encoding -------------------------------------------

// Encode raw bytes into polynomial coefficients mod plain_mod
// Each entry of entry_bytes is packed into ceil(entry_bytes*8/log2(plain_mod)) coefficients
__global__ void encode_bytes_kernel(const uint8_t* raw_data, uint64_t* output,
                                     uint32_t entry_bytes, uint32_t bits_per_coeff,
                                     uint32_t coeffs_per_entry, uint32_t num_entries,
                                     uint32_t n);

// Pad entries to fill polynomial degree N (zero-pad remaining coefficients)
__global__ void pad_to_poly_kernel(uint64_t* data, uint32_t valid_coeffs,
                                    uint32_t n);

} // namespace module
} // namespace onionpir
