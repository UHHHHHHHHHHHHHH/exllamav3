#include "p2p_all_reduce.cuh"
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
