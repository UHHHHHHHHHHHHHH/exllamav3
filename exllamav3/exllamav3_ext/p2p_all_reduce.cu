#include "p2p_all_reduce.cuh"
#include "norm_row.cuh"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Two-rank one-shot all-reduce over a P2P mapping of the peer's receive buffer. Each call launches the same
// fixed grid, so every block's device-side epoch advances once per call on both ranks and acts as a call
// counter; the receive slot alternates with it, so a rank running one call ahead never overwrites data the
// peer is still reading.

constexpr int P2P_THREADS = 512;

#define P2P_CUDA_CHECK(x) do { cudaError_t e_ = (x); TORCH_CHECK(e_ == cudaSuccess, "CUDA: ", cudaGetErrorString(e_)); } while (0)

__device__ __forceinline__ int ld_acquire_sys(const int* p)
{
    int v;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_sys(int* p, int v)
{
    asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ half from_f<half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// T is the tensor dtype, W the wire dtype (bf16 for fp32 tensors, as the NCCL path sends them)
template <typename T, typename W>
__global__ void p2p_all_reduce_kernel
(
    T* __restrict__ data,
    int64_t n,
    W* __restrict__ local_recv,
    W* __restrict__ peer_recv,
    int64_t slot_elems,
    int* local_flags,
    int* peer_flags,
    int* epochs
)
{
    __shared__ int e_sh;
    int b = blockIdx.x;
    if (threadIdx.x == 0) e_sh = epochs[b] + 1;
    __syncthreads();
    int e = e_sh;
    int slot = e & 1;

    int64_t per_block = (n + P2P_BLOCKS - 1) / P2P_BLOCKS;
    int64_t start = (int64_t) b * per_block;
    int64_t end = min(n, start + per_block);

    W* dst = peer_recv + slot * slot_elems;
    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        dst[i] = from_f<W>(to_f<T>(data[i]));
    __syncthreads();

    if (threadIdx.x == 0)
    {
        __threadfence_system();
        st_release_sys(peer_flags + slot * P2P_BLOCKS + b, e);
        while (ld_acquire_sys(local_flags + slot * P2P_BLOCKS + b) < e) {}
        __threadfence();
    }
    __syncthreads();

    const W* src = local_recv + slot * slot_elems;
    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x)
    {
        float own = to_f<W>(from_f<W>(to_f<T>(data[i])));
        float sum = own + to_f<W>(src[i]);
        data[i] = from_f<T>(to_f<W>(from_f<W>(sum)));
    }

    if (threadIdx.x == 0) epochs[b] = e;
}

// Row-wise variant for the reduce in front of a residual add: each block owns whole rows (b, b + P2P_BLOCKS,
// ...), exchanges and sums them as p2p_all_reduce_kernel does and writes the sum back to data, then adds it
// to the residual. With res_norm it also normalizes, through rms_norm_kernel's own row body and block size,
// so the result is bit for bit the unfused sequence
template <typename T, typename W, typename R, typename weight_t, bool res_norm>
__global__ __launch_bounds__(1024)
void p2p_reduce_residual_kernel
(
    T* __restrict__ data,
    int rows,
    int dim,
    W* __restrict__ local_recv,
    W* __restrict__ peer_recv,
    int64_t slot_elems,
    int* local_flags,
    int* peer_flags,
    int* epochs,
    R* __restrict__ residual,
    const weight_t* __restrict__ w,
    half* __restrict__ y,
    float epsilon,
    float constant_bias,
    float constant_scale
)
{
    __shared__ int e_sh;
    int b = blockIdx.x;
    if (threadIdx.x == 0) e_sh = epochs[b] + 1;
    __syncthreads();
    int e = e_sh;
    int slot = e & 1;

    W* dst = peer_recv + slot * slot_elems;
    for (int row = b; row < rows; row += P2P_BLOCKS)
        for (int i = threadIdx.x; i < dim; i += blockDim.x)
        {
            int64_t k = (int64_t) row * dim + i;
            dst[k] = from_f<W>(to_f<T>(data[k]));
        }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        __threadfence_system();
        st_release_sys(peer_flags + slot * P2P_BLOCKS + b, e);
        while (ld_acquire_sys(local_flags + slot * P2P_BLOCKS + b) < e) {}
        __threadfence();
    }
    __syncthreads();

    const W* src = local_recv + slot * slot_elems;
    for (int row = b; row < rows; row += P2P_BLOCKS)
        for (int i = threadIdx.x; i < dim; i += blockDim.x)
        {
            int64_t k = (int64_t) row * dim + i;
            float own = to_f<W>(from_f<W>(to_f<T>(data[k])));
            data[k] = from_f<T>(to_f<W>(from_f<W>(own + to_f<W>(src[k]))));
        }
    __syncthreads();

    for (int row = b; row < rows; row += P2P_BLOCKS)
    {
        if constexpr (res_norm)
            rms_norm_row<RES_IN>(data, w, y, residual, epsilon, row, dim, constant_bias, constant_scale, 1);
        else
            for (int i = threadIdx.x; i < dim; i += blockDim.x)
            {
                int64_t k = (int64_t) row * dim + i;
                residual[k] = from_f<R>(to_f<R>(residual[k]) + to_f<T>(data[k]));
            }
    }

    if (threadIdx.x == 0) epochs[b] = e;
}

std::tuple<int64_t, py::bytes> p2p_alloc(int64_t nbytes, int64_t device)
{
    c10::cuda::CUDAGuard g(device);
    void* ptr;
    P2P_CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    P2P_CUDA_CHECK(cudaMemset(ptr, 0, nbytes));
    cudaIpcMemHandle_t h;
    P2P_CUDA_CHECK(cudaIpcGetMemHandle(&h, ptr));
    return {(int64_t) ptr, py::bytes((const char*) &h, sizeof(h))};
}

int64_t p2p_open(py::bytes handle, int64_t device)
{
    c10::cuda::CUDAGuard g(device);
    std::string s = handle;
    TORCH_CHECK(s.size() == sizeof(cudaIpcMemHandle_t), "bad IPC handle size");
    cudaIpcMemHandle_t h;
    memcpy(&h, s.data(), sizeof(h));
    void* ptr;
    P2P_CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess));
    return (int64_t) ptr;
}

void p2p_free(int64_t ptr, int64_t device) { c10::cuda::CUDAGuard g(device); P2P_CUDA_CHECK(cudaFree((void*) ptr)); }
void p2p_close(int64_t ptr, int64_t device) { c10::cuda::CUDAGuard g(device); P2P_CUDA_CHECK(cudaIpcCloseMemHandle((void*) ptr)); }

void p2p_all_reduce(at::Tensor data, int64_t local_base, int64_t peer_base, int64_t slot_bytes, at::Tensor epochs)
{
    const at::cuda::OptionalCUDAGuard g(data.device());
    TORCH_CHECK(data.is_contiguous(), "p2p_all_reduce: data must be contiguous");
    TORCH_CHECK(epochs.numel() >= P2P_BLOCKS && epochs.dtype() == at::kInt, "p2p_all_reduce: bad epochs");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t n = data.numel();
    int* lf = (int*) local_base;
    int* pf = (int*) peer_base;

    #define LAUNCH(T, W) \
    { \
        int64_t slot_elems = slot_bytes / (int64_t) sizeof(W); \
        TORCH_CHECK(n <= slot_elems, "p2p_all_reduce: payload exceeds the receive slot"); \
        W* lr = (W*) ((char*) local_base + P2P_FLAGS_BYTES); \
        W* pr = (W*) ((char*) peer_base + P2P_FLAGS_BYTES); \
        p2p_all_reduce_kernel<T, W><<<P2P_BLOCKS, P2P_THREADS, 0, stream>>>((T*) data.data_ptr(), n, lr, pr, slot_elems, lf, pf, epochs.data_ptr<int>()); \
    }
    if (data.dtype() == at::kFloat) LAUNCH(float, __nv_bfloat16)
    else if (data.dtype() == at::kHalf) LAUNCH(half, half)
    else if (data.dtype() == at::kBFloat16) LAUNCH(__nv_bfloat16, __nv_bfloat16)
    else TORCH_CHECK(false, "p2p_all_reduce: unsupported dtype");
    #undef LAUNCH
    P2P_CUDA_CHECK(cudaPeekAtLastError());
}

void p2p_reduce_residual
(
    at::Tensor data,
    at::Tensor residual,
    c10::optional<at::Tensor> weight,
    c10::optional<at::Tensor> y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    int64_t local_base,
    int64_t peer_base,
    int64_t slot_bytes,
    at::Tensor epochs
)
{
    const at::cuda::OptionalCUDAGuard g(data.device());
    TORCH_CHECK(data.is_contiguous() && residual.is_contiguous(), "p2p_reduce_residual: tensors must be contiguous");
    TORCH_CHECK(data.numel() == residual.numel(), "p2p_reduce_residual: residual shape mismatch");
    TORCH_CHECK(epochs.numel() >= P2P_BLOCKS && epochs.dtype() == at::kInt, "p2p_reduce_residual: bad epochs");
    int dim = data.size(-1);
    int rows = data.numel() / dim;
    TORCH_CHECK(dim % 4 == 0, "p2p_reduce_residual: dim must be divisible by 4");
    bool res_norm = y.has_value();
    if (res_norm)
    {
        TORCH_CHECK(y.value().dtype() == at::kHalf && y.value().numel() == data.numel() && y.value().is_contiguous(), "p2p_reduce_residual: bad output");
        TORCH_CHECK(!weight.has_value() || weight.value().numel() == dim, "p2p_reduce_residual: bad weight");
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    // rms_norm_impl's block size, so the norm's reduction order is the same
    int threads = std::min(1024, (dim / 4 + 31) / 32 * 32);
    int* lf = (int*) local_base;
    int* pf = (int*) peer_base;
    half* y_ptr = res_norm ? (half*) y.value().data_ptr() : nullptr;
    const void* w_ptr = weight.has_value() ? weight.value().data_ptr() : nullptr;
    bool w_bf16 = weight.has_value() && weight.value().dtype() == at::kBFloat16;

    #define LAUNCH(T, W, R, WT, NORM) \
    { \
        int64_t slot_elems = slot_bytes / (int64_t) sizeof(W); \
        TORCH_CHECK(data.numel() <= slot_elems, "p2p_reduce_residual: payload exceeds the receive slot"); \
        W* lr = (W*) ((char*) local_base + P2P_FLAGS_BYTES); \
        W* pr = (W*) ((char*) peer_base + P2P_FLAGS_BYTES); \
        p2p_reduce_residual_kernel<T, W, R, WT, NORM><<<P2P_BLOCKS, threads, 0, stream>>>( \
            (T*) data.data_ptr(), rows, dim, lr, pr, slot_elems, lf, pf, epochs.data_ptr<int>(), \
            (R*) residual.data_ptr(), (const WT*) w_ptr, y_ptr, epsilon, constant_bias, constant_scale); \
    }
    #define BY_WEIGHT(T, W, R) \
        if (!res_norm) LAUNCH(T, W, R, half, false) \
        else if (w_bf16) LAUNCH(T, W, R, __nv_bfloat16, true) \
        else LAUNCH(T, W, R, half, true)
    #define BY_RESIDUAL(T, W) \
        if (residual.dtype() == at::kFloat) { BY_WEIGHT(T, W, float) } \
        else if (residual.dtype() == at::kHalf) { BY_WEIGHT(T, W, half) } \
        else TORCH_CHECK(false, "p2p_reduce_residual: unsupported residual dtype");

    if (data.dtype() == at::kFloat) { BY_RESIDUAL(float, __nv_bfloat16) }
    else if (data.dtype() == at::kHalf) { BY_RESIDUAL(half, half) }
    else TORCH_CHECK(false, "p2p_reduce_residual: unsupported dtype");
    #undef BY_RESIDUAL
    #undef BY_WEIGHT
    #undef LAUNCH
    P2P_CUDA_CHECK(cudaPeekAtLastError());
}
