#include "common.cuh"

template<typename _Tp>
struct MinOp
{
    static __device__ __forceinline__ _Tp init()
    {
        return std::numeric_limits<_Tp>::max();
    }

    static __device__ __forceinline__ _Tp reduce(const _Tp a, const _Tp b)
    {
        return a < b ? a : b;
    }
};

template<typename _Tp>
struct MaxOp
{
    static __device__ __forceinline__ _Tp init()
    {
        return std::numeric_limits<_Tp>::min();
    }

    static __device__ __forceinline__ _Tp reduce(const _Tp a, const _Tp b)
    {
        return a > b ? a : b;
    }
};

template<typename T, typename KT, class Op, int CH>
__global__ void morphology_kernel(const T *__restrict__ src, T *__restrict__ dst, const KT *__restrict__ SE,
                                  const int ksh, const int ksw, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int anchor_y = ksh / 2;
    const int anchor_x = ksw / 2;

    const int base = (y * W + x) * CH;

#pragma unroll
    for (int c = 0; c < CH; c++)
    {
        T acc = Op::init(); // 累积值

        // 遍历结构元素
        for (int ky = 0; ky < ksh; ky++)
        {
            const int sy = y + ky - anchor_y;
            if (sy < 0 || sy >= H)
                continue; // 超出边界

            for (int kx = 0; kx < ksw; kx++)
            {
                if (SE[ky * ksw + kx] == 0)
                    continue; // kernel 此位置无效

                const int sx = x + kx - anchor_x;
                if (sx < 0 || sx >= W)
                    continue;

                const int src_idx = (sy * W + sx) * CH + c;
                const T   val     = src[src_idx];

                acc = Op::reduce(acc, val);
            }
        }

        dst[base + c] = acc;
    }
}

template<typename T, typename KT, class Op, int CH>
__global__ void morphology_no_cond_kernel(const T *__restrict__ src, T *__restrict__ dst, const KT *__restrict__ SE,
                                          const int ksh, const int ksw, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int anchor_y = ksh / 2;
    const int anchor_x = ksw / 2;

    const int base = (y * W + x) * CH;

#pragma unroll
    for (int c = 0; c < CH; c++)
    {
        T acc = Op::init(); // 累积值

        // 遍历结构元素
        for (int ky = 0; ky < ksh; ky++)
        {
            const int sy = border_replicate(y + ky - anchor_y, H);

            for (int kx = 0; kx < ksw; kx++)
            {
                if (SE[ky * ksw + kx] == 0)
                    continue; // kernel 此位置无效

                const int sx = border_replicate(x + kx - anchor_x, W);

                const int src_idx = (sy * W + sx) * CH + c;
                const T   val     = src[src_idx];

                acc = Op::reduce(acc, val);
            }
        }

        dst[base + c] = acc;
    }
}

template<typename T, typename KT, class Op, int CH>
__global__ void morphology_shared_kernel(const T *__restrict__ src, T *__restrict__ dst, const KT *__restrict__ SE,
                                         const int ksh, const int ksw, const int H, const int W, const int N)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x  = blockIdx.x * blockDim.x + tx;
    const int y  = blockIdx.y * blockDim.y + ty;

    const int anchor_y = ksh / 2;
    const int anchor_x = ksw / 2;

    // 动态共享内存，tile + halo
    extern __shared__ unsigned char smem[];

    T *tile = reinterpret_cast<T *>(smem);

    const int tile_h = blockDim.y + ksh - 1;
    const int tile_w = blockDim.x + ksw - 1;

    // global 起始位置（包含 halo）
    const int gx0 = blockIdx.x * blockDim.x - anchor_x;
    const int gy0 = blockIdx.y * blockDim.y - anchor_y;

    // 载入共享内存（含 halo）
    for (int c = 0; c < CH; ++c)
    {
        for (int j = ty; j < tile_h; j += blockDim.y)
        {
            int gy = gy0 + j;
            for (int i = tx; i < tile_w; i += blockDim.x)
            {
                int gx = gx0 + i;

                T val = 0;
                if (gx >= 0 && gx < W && gy >= 0 && gy < H)
                    val = src[(gy * W + gx) * CH + c];

                const int idx = (j * tile_w + i) * CH + c;
                tile[idx]     = val;
            }
        }
    }

    __syncthreads();

    // 输出范围外直接跳过
    if (x >= W || y >= H)
        return;

    const int shared_x = tx + anchor_x;
    const int shared_y = ty + anchor_y;

    for (int c = 0; c < CH; ++c)
    {
        T acc = Op::init();

        // 遍历结构元素
        for (int ky = 0; ky < ksh; ++ky)
        {
            const int sy = y + ky - anchor_y;
            if (sy < 0 || sy >= H)
                continue; // 超出边界直接跳过

            for (int kx = 0; kx < ksw; ++kx)
            {
                if (SE[ky * ksw + kx] == 0)
                    continue;

                const int sx = x + kx - anchor_x;
                if (sx < 0 || sx >= W)
                    continue; // 超出边界直接跳过

                const int shared_j   = shared_y + ky - anchor_y;
                const int shared_i   = shared_x + kx - anchor_x;
                const int shared_idx = (shared_j * tile_w + shared_i) * CH + c;

                acc = Op::reduce(acc, tile[shared_idx]);
            }
        }

        dst[(y * W + x) * CH + c] = acc;
    }
}

template<typename T, typename KT, class Op, int CH>
__global__ void morphology_v_kernel(const T *__restrict__ src, T *__restrict__ tmp, const KT *__restrict__ SE,
                                    const int ksh, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int x        = tid % W;
    const int y        = tid / W;
    const int anchor_y = ksh / 2;
    const int base     = (y * W + x) * CH;

#pragma unroll
    for (int c = 0; c < CH; c++)
    {
        T acc = Op::init();
        for (int ky = 0; ky < ksh; ky++)
        {
            if (SE[ky] == 0)
                continue;

            const int sy = y + ky - anchor_y;
            if (sy < 0 || sy >= H)
                continue;

            const int src_idx = (sy * W + x) * CH + c;
            acc               = Op::reduce(acc, src[src_idx]);
        }
        tmp[base + c] = acc;
    }
}

template<typename T, typename KT, class Op, int CH>
__global__ void morphology_h_kernel(const T *__restrict__ tmp, T *__restrict__ dst, const KT *__restrict__ SE,
                                    const int ksw, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int x        = tid % W;
    const int y        = tid / W;
    const int anchor_x = ksw / 2;
    const int base     = (y * W + x) * CH;

#pragma unroll
    for (int c = 0; c < CH; c++)
    {
        T acc = Op::init();
        for (int kx = 0; kx < ksw; kx++)
        {
            if (SE[kx] == 0)
                continue;

            const int sx = x + kx - anchor_x;
            if (sx < 0 || sx >= W)
                continue;

            const int tmp_idx = (y * W + sx) * CH + c;
            acc               = Op::reduce(acc, tmp[tmp_idx]);
        }
        dst[base + c] = acc;
    }
}

#define TORCH_BINDING_MORPHOLOGY(tag, th_type, element_type, kernel_type, Op, n_pack)                               \
    void tag##_##element_type##_##kernel_type(torch::Tensor src, torch::Tensor dst, torch::Tensor kernel,           \
                                              const int ksh, const int ksw)                                         \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(kernel)                                                                           \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
        dim3      block(THREADS);                                                                                   \
        dim3      grid(divUp(N, THREADS));                                                                          \
        if (CH == 1)                                                                                                \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, Op, 1><<<grid, block>>>(                                   \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
        else if (CH == 3)                                                                                           \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, Op, 3><<<grid, block>>>(                                   \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_TEMPLATE(tag, kname, th_type, element_type, kernel_type, Op, n_pack)               \
    void tag##_##kname##_##element_type##_##kernel_type(torch::Tensor src, torch::Tensor dst, torch::Tensor kernel, \
                                                        const int ksh, const int ksw)                               \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(kernel)                                                                           \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
        dim3      block(THREADS);                                                                                   \
        dim3      grid(divUp(N, THREADS));                                                                          \
        if (CH == 1)                                                                                                \
        {                                                                                                           \
            morphology_##kname##_kernel<element_type, kernel_type, Op, 1><<<grid, block>>>(                         \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
        else if (CH == 3)                                                                                           \
        {                                                                                                           \
            morphology_##kname##_kernel<element_type, kernel_type, Op, 3><<<grid, block>>>(                         \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_SHARED(tag, kname, th_type, element_type, kernel_type, Op, n_pack)                 \
    void tag##_##kname##_##element_type##_##kernel_type(torch::Tensor src, torch::Tensor dst, torch::Tensor kernel, \
                                                        const int ksh, const int ksw)                               \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(kernel)                                                                           \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
        dim3      block(16, 16);                                                                                    \
        dim3      grid(divUp(W, block.x), divUp(H, block.y));                                                       \
        size_t    smem_size = sizeof(element_type) * CH * (block.y + ksh - 1) * (block.x + ksw - 1);                \
        if (CH == 1)                                                                                                \
        {                                                                                                           \
            morphology_##kname##_kernel<element_type, kernel_type, Op, 1><<<grid, block, smem_size, 0>>>(           \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
        else if (CH == 3)                                                                                           \
        {                                                                                                           \
            morphology_##kname##_kernel<element_type, kernel_type, Op, 3><<<grid, block, smem_size, 0>>>(           \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_SEPARABLE(tag, th_type, element_type, kernel_type, Op, n_pack)                        \
    void tag##_separable_##element_type##_##kernel_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,       \
                                                        torch::Tensor kernel_x, torch::Tensor kernel_y, const int ksh, \
                                                        const int ksw)                                                 \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(kernel_x)                                                                            \
        CHECK_TORCH_TENSOR_DEVICE(kernel_y)                                                                            \
        const int H  = src.size(0);                                                                                    \
        const int W  = src.size(1);                                                                                    \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                               \
        const int N  = H * W;                                                                                          \
        dim3      block(THREADS);                                                                                      \
        dim3      grid(divUp(N, THREADS));                                                                             \
        if (CH == 1)                                                                                                   \
        {                                                                                                              \
            morphology_v_kernel<element_type, kernel_type, Op, 1><<<grid, block>>>(                                    \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(tmp.data_ptr()),    \
                reinterpret_cast<kernel_type *>(kernel_y.data_ptr()), ksh, H, W, N);                                   \
            morphology_h_kernel<element_type, kernel_type, Op, 1><<<grid, block>>>(                                    \
                reinterpret_cast<element_type *>(tmp.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),    \
                reinterpret_cast<kernel_type *>(kernel_x.data_ptr()), ksw, H, W, N);                                   \
        }                                                                                                              \
        else if (CH == 3)                                                                                              \
        {                                                                                                              \
            morphology_v_kernel<element_type, kernel_type, Op, 3><<<grid, block>>>(                                    \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(tmp.data_ptr()),    \
                reinterpret_cast<kernel_type *>(kernel_y.data_ptr()), ksh, H, W, N);                                   \
            morphology_h_kernel<element_type, kernel_type, Op, 3><<<grid, block>>>(                                    \
                reinterpret_cast<element_type *>(tmp.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),    \
                reinterpret_cast<kernel_type *>(kernel_x.data_ptr()), ksw, H, W, N);                                   \
        }                                                                                                              \
    }

TORCH_BINDING_MORPHOLOGY(erode, torch::kUInt8, uint8_t, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY(dilate, torch::kUInt8, uint8_t, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_TEMPLATE(erode, no_cond, torch::kUInt8, uint8_t, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_TEMPLATE(dilate, no_cond, torch::kUInt8, uint8_t, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_SHARED(erode, shared, torch::kUInt8, uint8_t, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_SHARED(dilate, shared, torch::kUInt8, uint8_t, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_SEPARABLE(erode, torch::kUInt8, uint8_t, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_SEPARABLE(dilate, torch::kUInt8, uint8_t, uint8_t, MaxOp<uint8_t>, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(erode_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_no_cond_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_no_cond_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_shared_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_shared_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_separable_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable_uint8_t_uint8_t)
}