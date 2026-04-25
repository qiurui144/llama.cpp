#pragma once

#include <cstddef>

namespace sqnbitgemm_spacemit_ime {
namespace ime1 {
size_t gemm_kernel_i8i4(size_t            blk_len,
                        const std::byte * quant_a_ptr,
                        const std::byte * quant_b_data,
                        const float *     quant_b_scale,
                        const std::byte * quant_b_zp,
                        float *           c_ptr,
                        size_t            count_m,
                        size_t            count_n,
                        size_t            count_k,
                        size_t            block_count_k,
                        size_t            ldc,
                        const float *     bias,
                        const size_t      scale_stride);

// Same shape as gemm_kernel_i8i4 but takes INT8-weight blocks (block_q8_0x16
// layout). quant_b_zp is unused for Q8_0 (signed Q8_0 has no zero point).
// Returns the number of rows handled (1 or 4), matching the M1 / M4 kernel
// dispatch in ime1_kernels.cpp.
size_t gemm_kernel_i8i8(size_t            blk_len,
                        const std::byte * quant_a_ptr,
                        const std::byte * quant_b_data,
                        const float *     quant_b_scale,
                        const std::byte * quant_b_zp,
                        float *           c_ptr,
                        size_t            count_m,
                        size_t            count_n,
                        size_t            count_k,
                        size_t            block_count_k,
                        size_t            ldc,
                        const float *     bias,
                        const size_t      scale_stride);

void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, std::byte * quant_a_ptr);

void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, std::byte * quant_a_ptr);

}  // namespace ime1
}  // namespace sqnbitgemm_spacemit_ime
