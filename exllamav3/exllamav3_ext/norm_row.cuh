#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "util.cuh"

using bfloat16 = __nv_bfloat16;

template <bool clamp>
__device__ inline void read_half4(float4& f4, const half4* addr)
{
    half4 h4;
    READ64(h4, addr);
    f4.x = LOW_TO_FLOAT(h4.x);
    f4.y = HIGH_TO_FLOAT(h4.x);
    f4.z = LOW_TO_FLOAT(h4.y);
    f4.w = HIGH_TO_FLOAT(h4.y);
    if constexpr (clamp)
    {
        f4.x = CLAMP_FP16(f4.x);
        f4.y = CLAMP_FP16(f4.y);
        f4.z = CLAMP_FP16(f4.z);
        f4.w = CLAMP_FP16(f4.w);
    }
}

__device__ inline void read_bfloat164(float4& f4, const bfloat164* addr)
{
    bfloat164 h4;
    READ64(h4, addr);
    f4.x = __bfloat162float(__low2bfloat16(h4.x));
    f4.y = __bfloat162float(__high2bfloat16(h4.x));
    f4.z = __bfloat162float(__low2bfloat16(h4.y));
    f4.w = __bfloat162float(__high2bfloat16(h4.y));
}

__device__ inline void read_float4(float4& f4, const float4* addr)
{
    READ128(f4, addr);
}

__device__ inline void write_half4(const float4& f4, half4* addr)
{
    half4 h4
    (
        __halves2half2(__float2half_rn(f4.x), __float2half_rn(f4.y)),
        __halves2half2(__float2half_rn(f4.z), __float2half_rn(f4.w))
    );
    WRITE64(addr, h4);
}

__device__ inline void write_bfloat164(const float4& f4, bfloat164* addr)
{
    bfloat164 h4
    (
        __halves2bfloat162(__float2bfloat16_rz(f4.x), __float2bfloat16_rz(f4.y)),
        __halves2bfloat162(__float2bfloat16_rz(f4.z), __float2bfloat16_rz(f4.w))
    );
    WRITE64(addr, h4);
}


__device__ inline void write_float4(const float4& f4, float4* addr)
{
    WRITE128(addr, f4);
}

__device__ inline float sum_sq4(float lsum, const float4& f4)
{
    lsum = fma(f4.x, f4.x, lsum);
    lsum = fma(f4.y, f4.y, lsum);
    lsum = fma(f4.z, f4.z, lsum);
    lsum = fma(f4.w, f4.w, lsum);
    return lsum;
}

__device__ inline void apply4(float4& x4, const float4& w4, const float rmf)
{
    x4.x = x4.x * w4.x * rmf;
    x4.y = x4.y * w4.y * rmf;
    x4.z = x4.z * w4.z * rmf;
    x4.w = x4.w * w4.w * rmf;
}

__device__ inline void apply4_nw(float4& x4, const float rmf)
{
    x4.x = x4.x * rmf;
    x4.y = x4.y * rmf;
    x4.z = x4.z * rmf;
    x4.w = x4.w * rmf;
}

// Block-size-agnostic reduction (any multiple of 32 threads)
__device__ inline float reduce_dyn(float sum, int warp_id, int lane_id)
{
    __shared__ float sums[32];
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    int num_warps = blockDim.x / 32;
    if (num_warps == 1) return sum;
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();
    sum = lane_id < num_warps ? sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    return sum;
}

// res_mode 0: y = norm(x) * w
// res_mode 1: y += norm(x) * w                            (post-norm residual accumulate)
// res_mode 2: r += x; y = norm(r) * w                     (fused pre-norm residual add)
#define RES_NONE 0
#define RES_POST 1
#define RES_IN 2

// One row of rms_norm_kernel, shared with the kernels that fuse a reduction in front of the norm so
// both produce the same bits. Must run with the whole block, sized as rms_norm_impl sizes it
template <int res_mode, typename input_t, typename output_t, typename weight_t, typename residual_t>
__device__ inline void rms_norm_row
(
    const input_t* __restrict__ x,
    const weight_t* __restrict__ w,
    output_t* __restrict__ y,
    residual_t* __restrict__ r,
    const float epsilon,
    const int row,
    const int dim,
    const float constant_bias,
    const float constant_scale,
    const int w_groups          // weight spans w_groups rows, cycled by row index (grouped norm)
)
{
    constexpr bool input_fp32 = std::is_same_v<input_t, float>;
    constexpr bool output_fp32 = std::is_same_v<output_t, float>;
    constexpr bool input_fp16 = std::is_same_v<input_t, half>;
    constexpr bool output_fp16 = std::is_same_v<output_t, half>;
    static_assert(input_fp32 || input_fp16, "rms_norm_kernel: input must be float or half type");
    static_assert(output_fp32 || output_fp16, "rms_norm_kernel: output must be float or half type");
    constexpr bool weight_bf16 = std::is_same_v<weight_t, bfloat16>;
    constexpr bool residual_fp16 = std::is_same_v<residual_t, half>;

    int t = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    const size_t row_off = (size_t) row * dim;   // 64-bit: rows * dim can exceed 2^31 elements

    if (w && w_groups > 1)
        w += (size_t) (row % w_groups) * dim;

    int columns = dim / 4;
    bool single = columns <= blockDim.x;

    auto read_in = [&] (float4& f4, const input_t* addr)
    {
        if constexpr (input_fp16) read_half4<true>(f4, (const half4*) addr);
        if constexpr (input_fp32) read_float4(f4, (const float4*) addr);
    };

    auto add_resid_in = [&] (float4& x4, int column)
    {
        // r += x, rounded to the residual dtype so the result matches an unfused add
        float4 r4;
        if constexpr (residual_fp16) read_half4<false>(r4, ((const half4*) (r + row_off)) + column);
        else                         read_float4(r4, ((const float4*) (r + row_off)) + column);
        x4.x += r4.x;
        x4.y += r4.y;
        x4.z += r4.z;
        x4.w += r4.w;
        if constexpr (residual_fp16)
        {
            half4 h4
            (
                __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y)),
                __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w))
            );
            WRITE64(((half4*) (r + row_off)) + column, h4);
            x4.x = LOW_TO_FLOAT(h4.x);
            x4.y = HIGH_TO_FLOAT(h4.x);
            x4.z = LOW_TO_FLOAT(h4.y);
            x4.w = HIGH_TO_FLOAT(h4.y);
        }
        else
            write_float4(x4, ((float4*) (r + row_off)) + column);
    };

    auto apply_out = [&] (float4& x4, int column, float rmf)
    {
        if (w)
        {
            float4 w4;
            if constexpr (weight_bf16) read_bfloat164   (w4, ((const bfloat164*) w) + column);
            else                       read_half4<false>(w4, ((const half4*)     w) + column);
            if (constant_bias != 0.0f)
            {
                w4.x += constant_bias;
                w4.y += constant_bias;
                w4.z += constant_bias;
                w4.w += constant_bias;
            }
            apply4(x4, w4, rmf);
        }
        else
        {
            apply4_nw(x4, rmf);
        }

        if constexpr (res_mode == RES_POST)
        {
            float4 r4;
            if constexpr (output_fp16) read_half4<false>(r4, ((half4*) (y + row_off)) + column);
            if constexpr (output_fp32) read_float4(r4, ((float4*) (y + row_off)) + column);
            x4.x += r4.x;
            x4.y += r4.y;
            x4.z += r4.z;
            x4.w += r4.w;
        }

        if constexpr (output_fp16) write_half4(x4, ((half4*) (y + row_off)) + column);
        if constexpr (output_fp32) write_float4(x4, ((float4*) (y + row_off)) + column);
    };

    if (single)
    {
        // One float4 per thread: keep the value in a register between the two phases
        float4 x4 = {};
        float sum = 0.0f;
        if (t < columns)
        {
            read_in(x4, x + row_off + 4 * t);
            if constexpr (res_mode == RES_IN) add_resid_in(x4, t);
            sum = sum_sq4(sum, x4);
        }
        sum = reduce_dyn(sum, warp_id, lane_id);
        float rmf = rsqrtf(sum / (float) dim + epsilon) * constant_scale;
        if (t < columns)
            apply_out(x4, t, rmf);
    }
    else
    {
        float sum = 0.0f;
        for (int column = t; column < columns; column += blockDim.x)
        {
            float4 x4;
            read_in(x4, x + row_off + 4 * column);
            if constexpr (res_mode == RES_IN) add_resid_in(x4, column);
            sum = sum_sq4(sum, x4);
        }
        sum = reduce_dyn(sum, warp_id, lane_id);
        float rmf = rsqrtf(sum / (float) dim + epsilon) * constant_scale;

        for (int column = t; column < columns; column += blockDim.x)
        {
            float4 x4;
            // For RES_IN the summed values were written back to r in the first pass
            if constexpr (res_mode == RES_IN)
            {
                if constexpr (residual_fp16) read_half4<false>(x4, ((const half4*) (r + row_off)) + column);
                else                         read_float4(x4, ((const float4*) (r + row_off)) + column);
            }
            else
                read_in(x4, x + row_off + 4 * column);
            apply_out(x4, column, rmf);
        }
    }
}

