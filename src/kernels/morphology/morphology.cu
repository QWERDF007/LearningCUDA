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

template<typename T, class Op, int CH>
__global__ void morphology_h_kernel(const T *__restrict__ tmp, T *__restrict__ dst, const int ksw, const int H,
                                    const int W, const int N)
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
            const int sx = x + kx - anchor_x;
            if (sx < 0 || sx >= W)
                continue;

            const int tmp_idx = (y * W + sx) * CH + c;
            acc               = Op::reduce(acc, tmp[tmp_idx]);
        }
        dst[base + c] = acc;
    }
}

template<typename T, class Op, int CH>
__global__ void morphology_v_kernel(const T *__restrict__ src, T *__restrict__ tmp, const int ksh, const int H,
                                    const int W, const int N)
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
            const int sy = y + ky - anchor_y;
            if (sy < 0 || sy >= H)
                continue;

            const int src_idx = (sy * W + x) * CH + c;
            acc               = Op::reduce(acc, src[src_idx]);
        }
        tmp[base + c] = acc;
    }
}

template<typename T, class Op>
__global__ void morphology_h_shared_kernel(const T *__restrict__ tmp, T *__restrict__ dst, const int ksw, const int H,
                                           const int W, const int N)
{
    extern __shared__ T sdata[]; // 动态共享内存

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int anchor_x = ksw / 2;

    // 每个 block 处理一行的一段
    const int x_start = bx * blockDim.x;
    const int x_end   = min(W, x_start + blockDim.x);
    const int y       = by;

    if (y >= H)
        return;

    const int valid_len = x_end - x_start;

    // 每个线程负责加载一个元素到共享内存
    // 但要额外加载左右边界区域
    const int halo_left  = anchor_x;
    const int halo_right = anchor_x;

    // 全部共享内存区域长度 = 数据 + 左右边界
    const int total_len = valid_len + halo_left + halo_right;

    // 全局读取时的起始位置
    const int global_base = y * W;

    // 计算每个线程对应共享内存加载位置（带边界）
    for (int i = tx; i < total_len; i += blockDim.x)
    {
        int gx   = x_start + i - halo_left; // 对应的全局 x 坐标
        gx       = max(0, min(W - 1, gx));  // 边界 clamp
        sdata[i] = tmp[global_base + gx];
    }

    __syncthreads();

    // 每个线程计算输出
    if (tx < valid_len)
    {
        T acc = Op::init();

        const int sx = tx + halo_left;

        // 在共享内存内滑动
        for (int kx = 0; kx < ksw; kx++)
        {
            const int lx = sx + kx - anchor_x;
            acc          = Op::reduce(acc, sdata[lx]);
        }

        dst[global_base + x_start + tx] = acc;
    }
}

template<typename T, class Op>
__global__ void morphology_v_shared_kernel(const T *__restrict__ tmp, T *__restrict__ dst, const int ksh, const int H,
                                           const int W, const int N)
{
    extern __shared__ T sdata[]; // 动态共享内存

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int anchor_y = ksh / 2;

    // 每个 block 处理一列块的一段（沿垂直方向）
    const int y_start = by * blockDim.x;
    const int y_end   = min(H, y_start + blockDim.x);
    const int x       = bx;

    if (x >= W)
        return;

    const int valid_len = y_end - y_start;

    // halo 区域（上 / 下）
    const int halo_top    = anchor_y;
    const int halo_bottom = anchor_y;

    const int total_len = valid_len + halo_top + halo_bottom;

    // 加载时以列为主（固定列 x）
    for (int i = tx; i < total_len; i += blockDim.x)
    {
        int gy = y_start + i - halo_top; // 对应全局 y
        gy     = max(0, min(H - 1, gy)); // 边界 clamp

        sdata[i] = tmp[gy * W + x];
    }

    __syncthreads();

    // 每个线程计算一个像素（输出）
    if (tx < valid_len)
    {
        T acc = Op::init();

        const int sy = tx + halo_top;

        for (int ky = 0; ky < ksh; ky++)
        {
            const int ly = sy + ky - anchor_y;
            acc          = Op::reduce(acc, sdata[ly]);
        }

        dst[(y_start + tx) * W + x] = acc;
    }
}

// 注意：shared_mem 需要分配 (block_size*4 + ksw - 1) * sizeof(T)
template<typename T, class Op>
__global__ void morphology_h_shared_vec4_kernel(const T *__restrict__ tmp, T *__restrict__ dst, const int ksw,
                                                const int H, const int W, const int N)
{
    extern __shared__ T sdata[]; // 大小按调用者传入

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int vec_size = 4; // 一次处理4个连续像素
    const int anchor_x = ksw / 2;

    // 每个 block 处理的像素起点（全局坐标）
    const int block_pixel_start = bx * blockDim.x * vec_size; // e.g. bx * 256 * 4
    const int block_len         = blockDim.x * vec_size;      // 有效像素（不含 halo）
    const int y                 = by;
    if (y >= H)
        return;

    const int global_base = y * W;

    // valid_len: 实际本 block 内在图像范围的有效像素数量 (可能是 block_len 的一部分)
    const int valid_len = max(0, min(W - block_pixel_start, block_len));

    // total shared region = valid_len + left_halo + right_halo
    const int halo_left  = anchor_x;
    const int halo_right = anchor_x;
    const int total_len  = valid_len + halo_left + halo_right; // 以像素为单位

    // 每个线程负责若干个 sdata 元素的加载（按 1-by-1 加载）
    // 我们在加载时，对于越界处写入 Op::init()（中立元素），而不是 clamp 到边界像素
    for (int idx = tx; idx < total_len; idx += blockDim.x)
    {
        // 对应全局 x 坐标
        int gx = block_pixel_start + idx - halo_left;
        if (gx < 0 || gx >= W)
        {
            sdata[idx] = Op::init(); // 越界 -> 写中立元素（相当于在原标量中 skip）
        }
        else
        {
            sdata[idx] = tmp[global_base + gx];
        }
    }

    __syncthreads();

    // 每个线程在共享内存中计算 vec_size 个连续像素
    const int thread_pixel_offset = tx * vec_size; // 在 block 内的偏移（0..block_len-1）

    //     // 在共享内存中计算输出
    //     T acc[vec_size];
    // #pragma unroll
    //     for (int i = 0; i < vec_size; i++)
    //     {
    //         acc[i] = Op::init();
    //     }

    // 初始化 accumulators
    T acc0 = Op::init();
    T acc1 = Op::init();
    T acc2 = Op::init();
    T acc3 = Op::init();

    // 对每个 kernel 元素遍历并累积（从 sdata 中读取）
    // sdata 基址对应 block_pixel_start - halo_left
    const int sbase = halo_left + thread_pixel_offset; // 线程第0个像素在 sdata 中的索引

    for (int kx = 0; kx < ksw; ++kx)
    {
        const int offset = kx - anchor_x;

        // #pragma unroll
        //         for (int i = 0; i < vec_size; i++)
        //         {
        //             acc[i] = Op::reduce(acc[i], sdata[sbase + i + offset]);
        //         }

        // 读取 sdata 的四个位置（如果这些位置本身是越界，在加载时已经被写成 Op::init()）
        const T v0 = sdata[sbase + 0 + offset];
        const T v1 = sdata[sbase + 1 + offset];
        const T v2 = sdata[sbase + 2 + offset];
        const T v3 = sdata[sbase + 3 + offset];

        acc0 = Op::reduce(acc0, v0);
        acc1 = Op::reduce(acc1, v1);
        acc2 = Op::reduce(acc2, v2);
        acc3 = Op::reduce(acc3, v3);
    }

    //     // 写出连续4个结果
    // #pragma unroll
    //     for (int i = 0; i < vec_size; i++)
    //     {
    //         const int gx = block_pixel_start + thread_pixel_offset + i;
    //         if (gx < W)
    //             dst[global_base + gx] = acc[i];
    //     }

    // 写回 global memory（注意检查越界：block_pixel_start + thread_pixel_offset + i < W）
    const int out_x0 = block_pixel_start + thread_pixel_offset + 0;
    const int out_x1 = block_pixel_start + thread_pixel_offset + 1;
    const int out_x2 = block_pixel_start + thread_pixel_offset + 2;
    const int out_x3 = block_pixel_start + thread_pixel_offset + 3;

    if (out_x0 < W)
        dst[global_base + out_x0] = acc0;
    if (out_x1 < W)
        dst[global_base + out_x1] = acc1;
    if (out_x2 < W)
        dst[global_base + out_x2] = acc2;
    if (out_x3 < W)
        dst[global_base + out_x3] = acc3;
}

// template<typename T, typename KT, class Op>
// __global__ void morphology_h_shared_vec4_kernel(const T *__restrict__ tmp, T *__restrict__ dst,
//                                                 const KT *__restrict__ SE, const int ksw, const int H, const int W,
//                                                 const int N)
// {
//     extern __shared__ T sdata[];

//     const int tx = threadIdx.x;
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;

//     const int anchor_x = ksw / 2;
//     const int vec_size = 4; // 一次处理4个元素

//     const int y = by;

//     // 边界检查必须在加载共享内存之后，否则会导致部分线程提前退出，共享内存未完全加载
//     if (y >= H)
//         return;

//     const int global_base = y * W;
//     const int halo_left   = anchor_x;
//     const int halo_right  = anchor_x;

//     // 本 block 覆盖的有效像素范围（包括左右 halo）
//     const int block_len = blockDim.x * vec_size;
//     const int total_len = block_len + halo_left + halo_right;

//     // 加载共享内存（每个线程加载若干元素）- 所有线程都必须参与
//     const int block_x_start = bx * blockDim.x * vec_size;
//     for (int i = tx; i < total_len; i += blockDim.x)
//     {
//         int gx   = block_x_start + i - halo_left;
//         gx       = max(0, min(W - 1, gx)); // clamp 边界
//         sdata[i] = tmp[global_base + gx];
//     }
//     __syncthreads();

//     // 计算当前线程的起始位置
//     const int x_start = (bx * blockDim.x + tx) * vec_size;
//     if (x_start >= W)
//         return;

//     // 在共享内存中计算输出
//     T acc[vec_size];
// #pragma unroll
//     for (int i = 0; i < vec_size; i++)
//     {
//         acc[i] = Op::init();
//     }

//     const int sx_base = tx * vec_size + halo_left;

// #pragma unroll
//     for (int kx = 0; kx < ksw; kx++)
//     {
//         if (SE[kx] == 0)
//             continue;
//         const int offset = kx - anchor_x;

// #pragma unroll
//         for (int i = 0; i < vec_size; i++)
//         {
//             const int lx = sx_base + i + offset;
//             acc[i]       = Op::reduce(acc[i], sdata[lx]);
//         }
//     }

//     // 写出连续4个结果
// #pragma unroll
//     for (int i = 0; i < vec_size; i++)
//     {
//         const int gx = x_start + i;
//         if (gx < W)
//             dst[global_base + gx] = acc[i];
//     }
// }

template<typename T, class Op>
__global__ void morphology_h_shared_vec4_u8_kernel(const T *__restrict__ tmp, T *__restrict__ dst, const int ksw,
                                                   const int H, const int W, const int N)
{
    extern __shared__ T sdata[]; // 动态共享内存 (字节)

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int vec_size          = 4; // 每个线程处理4个像素
    const int anchor_x          = ksw / 2;
    const int block_pixel_start = bx * blockDim.x * vec_size;
    const int y                 = by;

    if (y >= H)
        return;

    const int global_base = y * W;
    const int block_len   = blockDim.x * vec_size;
    const int valid_len   = max(0, min(W - block_pixel_start, block_len));

    const int halo_left  = anchor_x;
    const int halo_right = anchor_x;
    const int total_len  = valid_len + halo_left + halo_right;

    // ---- 1️⃣ 加载共享内存（uchar4 向量化） ----
    // 每线程加载若干 uchar4
    uchar4 neutral4 = make_uchar4(Op::init(), Op::init(), Op::init(), Op::init());

    const int total_vec = (total_len + vec_size - 1) / vec_size;

    for (int vi = tx; vi < total_vec; vi += blockDim.x)
    {
        int gx = block_pixel_start + vi * vec_size - halo_left;

        uchar4 v;
        if (gx + 3 < 0 || gx >= W)
        {
            // 全部越界
            v = neutral4;
        }
        else
        {
            T tmpv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                int xx = gx + j;
                if (xx < 0 || xx >= W)
                    tmpv[j] = Op::init(); // 越界 → 中立值
                else
                    tmpv[j] = tmp[global_base + xx];
            }
            v = make_uchar4(tmpv[0], tmpv[1], tmpv[2], tmpv[3]);
        }

        reinterpret_cast<uchar4 *>(sdata)[vi] = v;
    }

    __syncthreads();

    // ---- 2️⃣ 计算4个输出像素 ----
    const int thread_pixel_offset = tx * vec_size;
    if (thread_pixel_offset >= valid_len)
        return;

    //     T acc[4];
    // #pragma unroll
    //     for (int i = 0; i < 4; i++)
    //     {
    //         acc[i] = Op::init();
    //     }

    // 初始化 accumulators
    T acc0 = Op::init();
    T acc1 = Op::init();
    T acc2 = Op::init();
    T acc3 = Op::init();

    const int sbase = halo_left + thread_pixel_offset;

    for (int kx = 0; kx < ksw; ++kx)
    {
        const int offset = kx - anchor_x;

        // #pragma unroll
        //         for (int i = 0; i < 4; i++)
        //         {
        //             T v    = sdata[sbase + i + offset];
        //             acc[i] = Op::reduce(acc[i], v);
        //         }

        const T v0 = sdata[sbase + 0 + offset];
        const T v1 = sdata[sbase + 1 + offset];
        const T v2 = sdata[sbase + 2 + offset];
        const T v3 = sdata[sbase + 3 + offset];

        acc0 = Op::reduce(acc0, v0);
        acc1 = Op::reduce(acc1, v1);
        acc2 = Op::reduce(acc2, v2);
        acc3 = Op::reduce(acc3, v3);
    }

    // ---- 3️⃣ 写回结果（uchar4 向量化） ----
    // uchar4    outv  = make_uchar4(acc[0], acc[1], acc[2], acc[3]);
    uchar4    outv  = make_uchar4(acc0, acc1, acc2, acc3);
    const int out_x = block_pixel_start + thread_pixel_offset;

    reinterpret_cast<uchar4 *>(dst + global_base)[out_x / vec_size] = outv;

    //     if (out_x < W)
    //     {
    //         // 安全写（可能不满4像素）
    //         T  *out_ptr = dst + global_base + out_x;
    //         int remain  = min(4, W - out_x);
    //         T  *vals    = reinterpret_cast<T *>(&outv);
    // #pragma unroll
    //         for (int i = 0; i < remain; i++)
    //         {
    //             out_ptr[i] = vals[i];
    //         }
    //     }
}

// vertical pass: write as uchar4 (no per-pixel safety writes)
// tmp: input image (u8), dst: output image (u8)
// grid.x spans columns (in groups of 4), grid.y spans rows
template<typename T, class Op>
__global__ void morphology_v_shared_vec4_u8_kernel(const T *__restrict__ tmp, T *__restrict__ dst, const int ksh,
                                                   const int H, const int W, const int N)
{
    extern __shared__ T sdata[]; // size = (block_len * ksh) bytes

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int vec_size = 4; // 每线程处理 4 列
    const int anchor_y = ksh / 2;

    // block 起始列（全局坐标）
    const int block_col_start = bx * blockDim.x * vec_size;
    const int block_len       = blockDim.x * vec_size; // 列数（不含纵向 halo）
    const int y               = by;
    if (y >= H)
        return; // 超出行范围

    const int global_row_base = y * W;

    // valid 列数（当前 block 在图像内真正覆盖的列数）
    const int valid_cols = max(0, min(W - block_col_start, block_len));

    // 共享内存布局： ksh 行 × block_len 列
    // sdata[ ky * block_len + col ] == tmp[ (y + ky - anchor_y) * W + (block_col_start + col) ] (若 in-range)
    // 若对应行越界，则填入 Op::init()
    T neutral = Op::init();

    // 每线程负责若干列（按 vec_size 步进），并为每列加载 ksh 个像素
    // 我们以“每次处理 vec_size 列”为单位来加载，这样后面索引与 uchar4 写入对齐
    for (int col = tx * vec_size; col < block_len; col += blockDim.x * vec_size)
    {
        // 对这组 vec_size 列中的每一列加载 ksh 个像素（按行）
        for (int j = 0; j < vec_size; ++j)
        {
            int col_idx = col + j;                   // 0..block_len-1
            int gx      = block_col_start + col_idx; // global x
            if (gx < 0 || gx >= W)
            {
                // 整列都在图像外（通常不会发生，因为 block_col_start >= 0），但以防万一
                for (int ky = 0; ky < ksh; ++ky)
                {
                    sdata[ky * block_len + col_idx] = neutral;
                }
            }
            else
            {
                // 列在图像宽度范围内，但对应的不同行可能越界 (top/bottom)
                for (int ky = 0; ky < ksh; ++ky)
                {
                    int gy = y + ky - anchor_y; // 对应的全局行
                    if (gy < 0 || gy >= H)
                    {
                        sdata[ky * block_len + col_idx] = neutral;
                    }
                    else
                    {
                        sdata[ky * block_len + col_idx] = tmp[gy * W + gx];
                    }
                }
            }
        }
    }

    __syncthreads();

    // 每个线程计算它负责的 4 列在当前行 y 上的输出
    const int thread_col_offset = tx * vec_size;
    if (thread_col_offset >= valid_cols)
        return; // 本线程不负责有效列（安全退出）

    T acc0 = Op::init();
    T acc1 = Op::init();
    T acc2 = Op::init();
    T acc3 = Op::init();

    const int col0 = thread_col_offset + 0;
    const int col1 = thread_col_offset + 1;
    const int col2 = thread_col_offset + 2;
    const int col3 = thread_col_offset + 3;

    for (int ky = 0; ky < ksh; ++ky)
    {
        const int base = ky * block_len; // sdata 基址行

        // 读取对应列的值（若列越界已在加载阶段处理为 neutral）
        T v0 = sdata[base + col0];
        T v1 = sdata[base + col1];
        T v2 = sdata[base + col2];
        T v3 = sdata[base + col3];

        acc0 = Op::reduce(acc0, v0);
        acc1 = Op::reduce(acc1, v1);
        acc2 = Op::reduce(acc2, v2);
        acc3 = Op::reduce(acc3, v3);
    }

    // 组合为 uchar4 并直接写回（无单像素安全写）
    uchar4 outv = make_uchar4(acc0, acc1, acc2, acc3);

    // 计算 vec 写入索引（以列为主，vec 每个对应一组 4 列）
    // dst 按行存储，reinterpret_cast<uchar4*>(dst + global_row_base) 将行的列按 uchar4 布局写
    const int vec_out_idx = (block_col_start + thread_col_offset) / vec_size;

    reinterpret_cast<uchar4 *>(dst + global_row_base)[vec_out_idx] = outv; // 直接写入（要求 launch 保证不写越界）
}

__global__ void mat_transpose_u8x16_coalesced_write_2d_kernel(uint8_t *A, uint8_t *B, const int H, const int W)
{
    const int x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int ax = x;
    const int ay = y * 16;
    if (ax >= W || ay + 15 >= H)
        return;

    // 从 A 非连续读取 16 个元素（分散读取）
    uint8_t pack[16];

#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        pack[i] = A[(ay + i) * W + ax];
    }

    // 连续写入 B（合并写入），使用 uint4 一次写入 16 字节
    reinterpret_cast<uint4 *>(B)[x * H / 16 + y] = *reinterpret_cast<uint4 *>(pack);
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

#define TORCH_BINDING_MORPHOLOGY_SEPARABLE(tag, th_type, element_type, Op, n_pack)                                   \
    void tag##_separable_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp, torch::Tensor tmpT, \
                                        const int ksh, const int ksw)                                                \
    {                                                                                                                \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                              \
        const int H  = src.size(0);                                                                                  \
        const int W  = src.size(1);                                                                                  \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                             \
        const int N  = H * W;                                                                                        \
                                                                                                                     \
        dim3 block1(THREADS);                                                                                        \
        dim3 grid1(divUp(N, THREADS));                                                                               \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                     \
        if (CH == 1)                                                                                                 \
        {                                                                                                            \
            morphology_h_kernel<element_type, Op, 1>                                                                 \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(src.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, Op, 1>                                                                 \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);                 \
        }                                                                                                            \
        else if (CH == 3)                                                                                            \
        {                                                                                                            \
            morphology_h_kernel<element_type, Op, 3>                                                                 \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(src.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, Op, 3>                                                                 \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);                 \
        }                                                                                                            \
    }

#define TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED(tag, kname, th_type, element_type, Op, n_pack)                    \
    void tag##_separable_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,          \
                                                  torch::Tensor tmpT, const int ksh, const int ksw)                 \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                             \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
                                                                                                                    \
        dim3   block(THREADS);                                                                                      \
        dim3   gridH(divUp(W, block.x *n_pack), H);                                                                 \
        dim3   gridV(W, divUp(H, block.x *n_pack));                                                                 \
        size_t smem_h_size = (block.x * n_pack + ksw - 1) * sizeof(element_type);                                   \
        size_t smem_v_size = (block.x * n_pack + ksh - 1) * sizeof(element_type);                                   \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                    \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, Op>                                                         \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            morphology_v_##kname##_kernel<element_type, Op>                                                         \
                <<<gridV, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                \
                                                   reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N); \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_SEPARABLE2_SHARED(tag, kname, th_type, element_type, Op, n_pack)                   \
    void tag##_separable2_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,         \
                                                   torch::Tensor tmpT, const int ksh, const int ksw)                \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                             \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
                                                                                                                    \
        dim3   block(THREADS);                                                                                      \
        dim3   gridH(divUp(W, block.x *n_pack), H);                                                                 \
        dim3   gridV(divUp(N, THREADS));                                                                            \
        size_t smem_h_size = (block.x * n_pack + ksw - 1) * sizeof(element_type);                                   \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                    \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, Op>                                                         \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            morphology_v_kernel<element_type, Op, 1>                                                                \
                <<<gridV, block>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                   reinterpret_cast<element_type *>(dst.data_ptr()), ksh, H, W, N);                 \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_VEC(tag, kname, th_type, element_type, Op, n_pack)               \
    void tag##_separable_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,         \
                                                  torch::Tensor tmpT, const int ksh, const int ksw)                \
    {                                                                                                              \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                  \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                             \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                             \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                             \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                            \
        const int H  = src.size(0);                                                                                \
        const int W  = src.size(1);                                                                                \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                           \
        const int N  = H * W;                                                                                      \
                                                                                                                   \
        dim3   block(THREADS);                                                                                     \
        dim3   grid(divUp(W, block.x *n_pack), H);                                                                 \
        size_t smem_h_size = (block.x * n_pack + ksw - 1) * sizeof(element_type);                                  \
        size_t smem_v_size = (block.x * n_pack) * ksh * sizeof(element_type);                                      \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                   \
        {                                                                                                          \
            morphology_h_##kname##_kernel<element_type, Op>                                                        \
                <<<grid, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                \
                                                  reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            morphology_v_##kname##_kernel<element_type, Op>                                                        \
                <<<grid, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                \
                                                  reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N); \
        }                                                                                                          \
    }

#define TORCH_BINDING_MORPHOLOGY_SEPARABLE_T(tag, th_type, element_type, Op, n_pack)                                   \
    void tag##_separable_T_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp, torch::Tensor tmpT, \
                                          const int ksh, const int ksw)                                                \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                                 \
        const int H  = src.size(0);                                                                                    \
        const int W  = src.size(1);                                                                                    \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                               \
        const int N  = H * W;                                                                                          \
                                                                                                                       \
        dim3 block1(THREADS);                                                                                          \
        dim3 grid1(divUp(N, THREADS));                                                                                 \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                       \
        if (CH == 1)                                                                                                   \
        {                                                                                                              \
            morphology_h_kernel<element_type, Op, 1>                                                                   \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(src.data_ptr()),                                  \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                   \
        }                                                                                                              \
        else if (CH == 3)                                                                                              \
        {                                                                                                              \
            morphology_h_kernel<element_type, Op, 3>                                                                   \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(src.data_ptr()),                                  \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                   \
        }                                                                                                              \
                                                                                                                       \
        /* Transpose tmp (H x W) -> tmpT (W x H) */                                                                    \
        {                                                                                                              \
            /* configure transpose kernel: each y-thread handles 16 rows */                                            \
            dim3 tblock(32, 1);                                                                                        \
            dim3 tgrid(divUp(W, tblock.x), divUp(H, 16));                                                              \
            mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid, tblock>>>(                                          \
                reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(tmpT.data_ptr()), H, W);      \
        }                                                                                                              \
                                                                                                                       \
        /* Second pass: apply horizontal morphology on transposed image (effectively vertical on original) */          \
        /* when transposed, H' = W, W' = H, use kernel_y as horizontal SE with size ksh */                             \
        dim3 block2(THREADS);                                                                                          \
        dim3 grid2(divUp(N, THREADS));                                                                                 \
        if (CH == 1)                                                                                                   \
        {                                                                                                              \
            morphology_h_kernel<element_type, Op, 1>                                                                   \
                <<<grid2, block2>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                                 \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksh, W, H, N);                   \
        }                                                                                                              \
        else if (CH == 3)                                                                                              \
        {                                                                                                              \
            morphology_h_kernel<element_type, Op, 3>                                                                   \
                <<<grid2, block2>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                                 \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksh, W, H, N);                   \
        }                                                                                                              \
                                                                                                                       \
        /* Transpose back tmp (now holds transposed final result W x H) -> dst (H x W) */                              \
        {                                                                                                              \
            dim3 tblock2(32, 1);                                                                                       \
            dim3 tgrid2(divUp(H, tblock2.x), divUp(W, 16));                                                            \
            mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid2, tblock2>>>(                                        \
                reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H);       \
        }                                                                                                              \
    }

#define TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_T(tag, kname, th_type, element_type, Op, n_pack)                 \
    void tag##_separable_T_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,       \
                                                    torch::Tensor tmpT, const int ksh, const int ksw)              \
    {                                                                                                              \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                             \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                             \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                             \
        const int H  = src.size(0);                                                                                \
        const int W  = src.size(1);                                                                                \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                           \
        const int N  = H * W;                                                                                      \
                                                                                                                   \
        dim3   block1(THREADS);                                                                                    \
        dim3   grid1(divUp(W, block1.x *n_pack), H);                                                               \
        size_t smem_size = (block1.x * n_pack + ksw - 1) * sizeof(element_type);                                   \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                   \
        {                                                                                                          \
            morphology_h_##kname##_kernel<element_type, Op>                                                        \
                <<<grid1, block1, smem_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                \
                                                  reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
        }                                                                                                          \
                                                                                                                   \
        /* Transpose tmp (H x W) -> tmpT (W x H) */                                                                \
        {                                                                                                          \
            /* configure transpose kernel: each y-thread handles 16 rows */                                        \
            dim3 tblock(32, 1);                                                                                    \
            dim3 tgrid(divUp(W, tblock.x), divUp(H, 16));                                                          \
            mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid, tblock>>>(                                      \
                reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(tmpT.data_ptr()), H, W);  \
        }                                                                                                          \
                                                                                                                   \
        /* Second pass: apply horizontal morphology on transposed image (effectively vertical on original) */      \
        /* when transposed, H' = W, W' = H, use kernel_y as horizontal SE with size ksh */                         \
        dim3 block2(THREADS);                                                                                      \
        dim3 grid2(divUp(H, block2.x *n_pack), W);                                                                 \
        smem_size = (block2.x * n_pack + ksh - 1) * sizeof(element_type);                                          \
        {                                                                                                          \
            morphology_h_##kname##_kernel<element_type, Op>                                                        \
                <<<grid2, block2, smem_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),               \
                                                  reinterpret_cast<element_type *>(tmp.data_ptr()), ksh, W, H, N); \
        }                                                                                                          \
                                                                                                                   \
        /* Transpose back tmp (now holds transposed final result W x H) -> dst (H x W) */                          \
        {                                                                                                          \
            dim3 tblock2(32, 1);                                                                                   \
            dim3 tgrid2(divUp(H, tblock2.x), divUp(W, 16));                                                        \
            mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid2, tblock2>>>(                                    \
                reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H);   \
        }                                                                                                          \
    }

/***************************** OPEN & CLOSE ************************************/

#define TORCH_BINDING_MORPHOLOGY_OPEN(tag, th_type, element_type, kernel_type, n_pack)                              \
    void tag##_##element_type##_##kernel_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,              \
                                              torch::Tensor kernel, const int ksh, const int ksw)                   \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(kernel)                                                                           \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                              \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
        dim3      block(THREADS);                                                                                   \
        dim3      grid(divUp(N, THREADS));                                                                          \
        if (CH == 1)                                                                                                \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, MinOp<element_type>, 1><<<grid, block>>>(                  \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(tmp.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
        else if (CH == 3)                                                                                           \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, MinOp<element_type>, 3><<<grid, block>>>(                  \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(tmp.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
        if (CH == 1)                                                                                                \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, MaxOp<element_type>, 1><<<grid, block>>>(                  \
                reinterpret_cast<element_type *>(tmp.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
        else if (CH == 3)                                                                                           \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, MaxOp<element_type>, 3><<<grid, block>>>(                  \
                reinterpret_cast<element_type *>(tmp.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_CLOSE(tag, th_type, element_type, kernel_type, n_pack)                             \
    void tag##_##element_type##_##kernel_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,              \
                                              torch::Tensor kernel, const int ksh, const int ksw)                   \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(kernel)                                                                           \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                              \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
        dim3      block(THREADS);                                                                                   \
        dim3      grid(divUp(N, THREADS));                                                                          \
        if (CH == 1)                                                                                                \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, MaxOp<element_type>, 1><<<grid, block>>>(                  \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(tmp.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
        else if (CH == 3)                                                                                           \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, MaxOp<element_type>, 3><<<grid, block>>>(                  \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(tmp.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
        if (CH == 1)                                                                                                \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, MinOp<element_type>, 1><<<grid, block>>>(                  \
                reinterpret_cast<element_type *>(tmp.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
        else if (CH == 3)                                                                                           \
        {                                                                                                           \
            morphology_kernel<element_type, kernel_type, MinOp<element_type>, 3><<<grid, block>>>(                  \
                reinterpret_cast<element_type *>(tmp.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE(tag, th_type, element_type, n_pack)                                  \
    void tag##_separable_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp, torch::Tensor tmpT, \
                                        const int ksh, const int ksw)                                                \
    {                                                                                                                \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                              \
        const int H  = src.size(0);                                                                                  \
        const int W  = src.size(1);                                                                                  \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                             \
        const int N  = H * W;                                                                                        \
                                                                                                                     \
        dim3 block1(THREADS);                                                                                        \
        dim3 grid1(divUp(N, THREADS));                                                                               \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                     \
        if (CH == 1)                                                                                                 \
        {                                                                                                            \
            morphology_h_kernel<element_type, MinOp<element_type>, 1>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(src.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, MinOp<element_type>, 1>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmpT.data_ptr()), ksw, H, W, N);                \
        }                                                                                                            \
        else if (CH == 3)                                                                                            \
        {                                                                                                            \
            morphology_h_kernel<element_type, MinOp<element_type>, 3>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(src.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, MinOp<element_type>, 3>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmpT.data_ptr()), ksw, H, W, N);                \
        }                                                                                                            \
        if (CH == 1)                                                                                                 \
        {                                                                                                            \
            morphology_h_kernel<element_type, MaxOp<element_type>, 1>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                               \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, MaxOp<element_type>, 1>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);                 \
        }                                                                                                            \
        else if (CH == 3)                                                                                            \
        {                                                                                                            \
            morphology_h_kernel<element_type, MaxOp<element_type>, 3>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                               \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, MaxOp<element_type>, 3>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);                 \
        }                                                                                                            \
    }

#define TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE(tag, th_type, element_type, n_pack)                                 \
    void tag##_separable_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp, torch::Tensor tmpT, \
                                        const int ksh, const int ksw)                                                \
    {                                                                                                                \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                              \
        const int H  = src.size(0);                                                                                  \
        const int W  = src.size(1);                                                                                  \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                             \
        const int N  = H * W;                                                                                        \
                                                                                                                     \
        dim3 block1(THREADS);                                                                                        \
        dim3 grid1(divUp(N, THREADS));                                                                               \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                     \
        if (CH == 1)                                                                                                 \
        {                                                                                                            \
            morphology_h_kernel<element_type, MaxOp<element_type>, 1>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(src.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, MaxOp<element_type>, 1>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmpT.data_ptr()), ksw, H, W, N);                \
        }                                                                                                            \
        else if (CH == 3)                                                                                            \
        {                                                                                                            \
            morphology_h_kernel<element_type, MaxOp<element_type>, 3>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(src.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, MaxOp<element_type>, 3>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(tmpT.data_ptr()), ksw, H, W, N);                \
        }                                                                                                            \
        if (CH == 1)                                                                                                 \
        {                                                                                                            \
            morphology_h_kernel<element_type, MinOp<element_type>, 1>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                               \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, MinOp<element_type>, 1>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);                 \
        }                                                                                                            \
        else if (CH == 3)                                                                                            \
        {                                                                                                            \
            morphology_h_kernel<element_type, MinOp<element_type>, 3>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                               \
                                    reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);                 \
            morphology_v_kernel<element_type, MinOp<element_type>, 3>                                                \
                <<<grid1, block1>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                    reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);                 \
        }                                                                                                            \
    }

#define TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE_SHARED(tag, kname, th_type, element_type, n_pack)                    \
    void tag##_separable_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,           \
                                                  torch::Tensor tmpT, const int ksh, const int ksw)                  \
    {                                                                                                                \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                              \
        const int H  = src.size(0);                                                                                  \
        const int W  = src.size(1);                                                                                  \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                             \
        const int N  = H * W;                                                                                        \
                                                                                                                     \
        dim3   block(THREADS);                                                                                       \
        dim3   gridH(divUp(W, block.x *n_pack), H);                                                                  \
        dim3   gridV(W, divUp(H, block.x *n_pack));                                                                  \
        size_t smem_h_size = (block.x * n_pack + ksw - 1) * sizeof(element_type);                                    \
        size_t smem_v_size = (block.x * n_pack + ksh - 1) * sizeof(element_type);                                    \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                     \
        {                                                                                                            \
            morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                         \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                 \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);  \
            morphology_v_##kname##_kernel<element_type, MinOp<element_type>>                                         \
                <<<gridV, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                 \
                                                   reinterpret_cast<element_type *>(tmpT.data_ptr()), ksw, H, W, N); \
        }                                                                                                            \
        {                                                                                                            \
            morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                         \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);  \
            morphology_v_##kname##_kernel<element_type, MaxOp<element_type>>                                         \
                <<<gridV, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                 \
                                                   reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);  \
        }                                                                                                            \
    }

#define TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE_SHARED(tag, kname, th_type, element_type, n_pack)                   \
    void tag##_separable_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,           \
                                                  torch::Tensor tmpT, const int ksh, const int ksw)                  \
    {                                                                                                                \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                              \
        const int H  = src.size(0);                                                                                  \
        const int W  = src.size(1);                                                                                  \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                             \
        const int N  = H * W;                                                                                        \
                                                                                                                     \
        dim3   block(THREADS);                                                                                       \
        dim3   gridH(divUp(W, block.x *n_pack), H);                                                                  \
        dim3   gridV(W, divUp(H, block.x *n_pack));                                                                  \
        size_t smem_h_size = (block.x * n_pack + ksw - 1) * sizeof(element_type);                                    \
        size_t smem_v_size = (block.x * n_pack + ksh - 1) * sizeof(element_type);                                    \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                     \
        {                                                                                                            \
            morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                         \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                 \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);  \
            morphology_v_##kname##_kernel<element_type, MaxOp<element_type>>                                         \
                <<<gridV, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                 \
                                                   reinterpret_cast<element_type *>(tmpT.data_ptr()), ksw, H, W, N); \
        }                                                                                                            \
        {                                                                                                            \
            morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                         \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);  \
            morphology_v_##kname##_kernel<element_type, MinOp<element_type>>                                         \
                <<<gridV, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                 \
                                                   reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);  \
        }                                                                                                            \
    }

#define TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE2_SHARED(tag, kname, th_type, element_type, n_pack)                  \
    void tag##_separable2_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,         \
                                                   torch::Tensor tmpT, const int ksh, const int ksw)                \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                             \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
                                                                                                                    \
        dim3   block(THREADS);                                                                                      \
        dim3   gridH(divUp(W, block.x *n_pack), H);                                                                 \
        dim3   gridV(divUp(N, THREADS));                                                                            \
        size_t smem_h_size = (block.x * n_pack + ksw - 1) * sizeof(element_type);                                   \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                    \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                        \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            morphology_v_kernel<element_type, MinOp<element_type>, 1>                                               \
                <<<gridV, block>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                   reinterpret_cast<element_type *>(tmpT.data_ptr()), ksh, H, W, N);                \
        }                                                                                                           \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                        \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),               \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            morphology_v_kernel<element_type, MaxOp<element_type>, 1>                                               \
                <<<gridV, block>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                   reinterpret_cast<element_type *>(dst.data_ptr()), ksh, H, W, N);                 \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE2_SHARED(tag, kname, th_type, element_type, n_pack)                 \
    void tag##_separable2_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,         \
                                                   torch::Tensor tmpT, const int ksh, const int ksw)                \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                             \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
                                                                                                                    \
        dim3   block(THREADS);                                                                                      \
        dim3   gridH(divUp(W, block.x *n_pack), H);                                                                 \
        dim3   gridV(divUp(N, THREADS));                                                                            \
        size_t smem_h_size = (block.x * n_pack + ksw - 1) * sizeof(element_type);                                   \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                    \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                        \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            morphology_v_kernel<element_type, MaxOp<element_type>, 1>                                               \
                <<<gridV, block>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                   reinterpret_cast<element_type *>(tmpT.data_ptr()), ksh, H, W, N);                \
        }                                                                                                           \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                        \
                <<<gridH, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),               \
                                                   reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            morphology_v_kernel<element_type, MinOp<element_type>, 1>                                               \
                <<<gridV, block>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                                \
                                   reinterpret_cast<element_type *>(dst.data_ptr()), ksh, H, W, N);                 \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE_SHARED_VEC(tag, kname, th_type, element_type, n_pack)               \
    void tag##_separable_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,          \
                                                  torch::Tensor tmpT, const int ksh, const int ksw)                 \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                             \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
                                                                                                                    \
        dim3   block(THREADS);                                                                                      \
        dim3   grid(divUp(W, block.x *n_pack), H);                                                                  \
        size_t smem_h_size = (block.x * n_pack + ksw - 1) * sizeof(element_type);                                   \
        size_t smem_v_size = (block.x * n_pack) * ksh * sizeof(element_type);                                       \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                    \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                        \
                <<<grid, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                 \
                                                  reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);  \
            morphology_v_##kname##_kernel<element_type, MinOp<element_type>>                                        \
                <<<grid, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                 \
                                                  reinterpret_cast<element_type *>(tmpT.data_ptr()), ksw, H, W, N); \
        }                                                                                                           \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                        \
                <<<grid, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                \
                                                  reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);  \
            morphology_v_##kname##_kernel<element_type, MaxOp<element_type>>                                        \
                <<<grid, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                 \
                                                  reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);  \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE_SHARED_VEC(tag, kname, th_type, element_type, n_pack)              \
    void tag##_separable_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,          \
                                                  torch::Tensor tmpT, const int ksh, const int ksw)                 \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(tmpT, (th_type))                                                                   \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(tmpT)                                                                             \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
                                                                                                                    \
        dim3   block(THREADS);                                                                                      \
        dim3   grid(divUp(W, block.x *n_pack), H);                                                                  \
        size_t smem_h_size = (block.x * n_pack + ksw - 1) * sizeof(element_type);                                   \
        size_t smem_v_size = (block.x * n_pack) * ksh * sizeof(element_type);                                       \
        /* First pass: horizontal morphology on original image -> tmp (H x W) */                                    \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                        \
                <<<grid, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                 \
                                                  reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);  \
            morphology_v_##kname##_kernel<element_type, MaxOp<element_type>>                                        \
                <<<grid, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                 \
                                                  reinterpret_cast<element_type *>(tmpT.data_ptr()), ksw, H, W, N); \
        }                                                                                                           \
        {                                                                                                           \
            morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                        \
                <<<grid, block, smem_h_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),                \
                                                  reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N);  \
            morphology_v_##kname##_kernel<element_type, MinOp<element_type>>                                        \
                <<<grid, block, smem_v_size, 0>>>(reinterpret_cast<element_type *>(tmp.data_ptr()),                 \
                                                  reinterpret_cast<element_type *>(dst.data_ptr()), ksw, H, W, N);  \
        }                                                                                                           \
    }

#define TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE_SHARED_T(tag, kname, th_type, element_type, n_pack)                    \
    void tag##_separable_T_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,           \
                                                    torch::Tensor tmpT, const int ksh, const int ksw)                  \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                                 \
        const int H  = src.size(0);                                                                                    \
        const int W  = src.size(1);                                                                                    \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                               \
        const int N  = H * W;                                                                                          \
        {                                                                                                              \
            /* First pass: horizontal morphology on original image -> tmp (H x W) */                                   \
            {                                                                                                          \
                dim3   block1(THREADS);                                                                                \
                dim3   grid1(divUp(W, block1.x *n_pack), H);                                                           \
                size_t smem_size = (block1.x * n_pack + ksw - 1) * sizeof(element_type);                               \
                morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                       \
                    <<<grid1, block1, smem_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                \
                                                      reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            }                                                                                                          \
                                                                                                                       \
            /* Transpose tmp (H x W) -> tmpT (W x H) */                                                                \
            {                                                                                                          \
                /* configure transpose kernel: each y-thread handles 16 rows */                                        \
                dim3 tblock(32, 1);                                                                                    \
                dim3 tgrid(divUp(W, tblock.x), divUp(H, 16));                                                          \
                mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid, tblock>>>(                                      \
                    reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(tmpT.data_ptr()), H, W);  \
            }                                                                                                          \
                                                                                                                       \
            /* Second pass: apply horizontal morphology on transposed image (effectively vertical on original) */      \
            /* when transposed, H' = W, W' = H, use kernel_y as horizontal SE with size ksh */                         \
            {                                                                                                          \
                dim3   block2(THREADS);                                                                                \
                dim3   grid2(divUp(H, block2.x *n_pack), W);                                                           \
                size_t smem_size = (block2.x * n_pack + ksh - 1) * sizeof(element_type);                               \
                morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                       \
                    <<<grid2, block2, smem_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),               \
                                                      reinterpret_cast<element_type *>(tmp.data_ptr()), ksh, W, H, N); \
            }                                                                                                          \
                                                                                                                       \
            /* Transpose back tmp (now holds transposed final result W x H) -> dst (H x W) */                          \
            {                                                                                                          \
                dim3 tblock2(32, 1);                                                                                   \
                dim3 tgrid2(divUp(H, tblock2.x), divUp(W, 16));                                                        \
                mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid2, tblock2>>>(                                    \
                    reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(tmpT.data_ptr()), W, H);  \
            }                                                                                                          \
        }                                                                                                              \
        {                                                                                                              \
            /* First pass: horizontal morphology on original image -> tmp (H x W) */                                   \
            {                                                                                                          \
                dim3   block1(THREADS);                                                                                \
                dim3   grid1(divUp(W, block1.x *n_pack), H);                                                           \
                size_t smem_size = (block1.x * n_pack + ksw - 1) * sizeof(element_type);                               \
                morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                       \
                    <<<grid1, block1, smem_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),               \
                                                      reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            }                                                                                                          \
                                                                                                                       \
            /* Transpose tmp (H x W) -> tmpT (W x H) */                                                                \
            {                                                                                                          \
                /* configure transpose kernel: each y-thread handles 16 rows */                                        \
                dim3 tblock(32, 1);                                                                                    \
                dim3 tgrid(divUp(W, tblock.x), divUp(H, 16));                                                          \
                mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid, tblock>>>(                                      \
                    reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(tmpT.data_ptr()), H, W);  \
            }                                                                                                          \
                                                                                                                       \
            /* Second pass: apply horizontal morphology on transposed image (effectively vertical on original) */      \
            /* when transposed, H' = W, W' = H, use kernel_y as horizontal SE with size ksh */                         \
            {                                                                                                          \
                dim3   block2(THREADS);                                                                                \
                dim3   grid2(divUp(H, block2.x *n_pack), W);                                                           \
                size_t smem_size = (block2.x * n_pack + ksh - 1) * sizeof(element_type);                               \
                morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                       \
                    <<<grid2, block2, smem_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),               \
                                                      reinterpret_cast<element_type *>(tmp.data_ptr()), ksh, W, H, N); \
            }                                                                                                          \
                                                                                                                       \
            /* Transpose back tmp (now holds transposed final result W x H) -> dst (H x W) */                          \
            {                                                                                                          \
                dim3 tblock2(32, 1);                                                                                   \
                dim3 tgrid2(divUp(H, tblock2.x), divUp(W, 16));                                                        \
                mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid2, tblock2>>>(                                    \
                    reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H);   \
            }                                                                                                          \
        }                                                                                                              \
    }

#define TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE_SHARED_T(tag, kname, th_type, element_type, n_pack)                   \
    void tag##_separable_T_##kname##_##element_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp,           \
                                                    torch::Tensor tmpT, const int ksh, const int ksw)                  \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(tmp, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                                 \
        const int H  = src.size(0);                                                                                    \
        const int W  = src.size(1);                                                                                    \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                               \
        const int N  = H * W;                                                                                          \
        {                                                                                                              \
            /* First pass: horizontal morphology on original image -> tmp (H x W) */                                   \
            {                                                                                                          \
                dim3   block1(THREADS);                                                                                \
                dim3   grid1(divUp(W, block1.x *n_pack), H);                                                           \
                size_t smem_size = (block1.x * n_pack + ksw - 1) * sizeof(element_type);                               \
                morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                       \
                    <<<grid1, block1, smem_size, 0>>>(reinterpret_cast<element_type *>(src.data_ptr()),                \
                                                      reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            }                                                                                                          \
                                                                                                                       \
            /* Transpose tmp (H x W) -> tmpT (W x H) */                                                                \
            {                                                                                                          \
                /* configure transpose kernel: each y-thread handles 16 rows */                                        \
                dim3 tblock(32, 1);                                                                                    \
                dim3 tgrid(divUp(W, tblock.x), divUp(H, 16));                                                          \
                mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid, tblock>>>(                                      \
                    reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(tmpT.data_ptr()), H, W);  \
            }                                                                                                          \
                                                                                                                       \
            /* Second pass: apply horizontal morphology on transposed image (effectively vertical on original) */      \
            /* when transposed, H' = W, W' = H, use kernel_y as horizontal SE with size ksh */                         \
            {                                                                                                          \
                dim3   block2(THREADS);                                                                                \
                dim3   grid2(divUp(H, block2.x *n_pack), W);                                                           \
                size_t smem_size = (block2.x * n_pack + ksh - 1) * sizeof(element_type);                               \
                morphology_h_##kname##_kernel<element_type, MaxOp<element_type>>                                       \
                    <<<grid2, block2, smem_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),               \
                                                      reinterpret_cast<element_type *>(tmp.data_ptr()), ksh, W, H, N); \
            }                                                                                                          \
                                                                                                                       \
            /* Transpose back tmp (now holds transposed final result W x H) -> dst (H x W) */                          \
            {                                                                                                          \
                dim3 tblock2(32, 1);                                                                                   \
                dim3 tgrid2(divUp(H, tblock2.x), divUp(W, 16));                                                        \
                mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid2, tblock2>>>(                                    \
                    reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(tmpT.data_ptr()), W, H);  \
            }                                                                                                          \
        }                                                                                                              \
        {                                                                                                              \
            /* First pass: horizontal morphology on original image -> tmp (H x W) */                                   \
            {                                                                                                          \
                dim3   block1(THREADS);                                                                                \
                dim3   grid1(divUp(W, block1.x *n_pack), H);                                                           \
                size_t smem_size = (block1.x * n_pack + ksw - 1) * sizeof(element_type);                               \
                morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                       \
                    <<<grid1, block1, smem_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),               \
                                                      reinterpret_cast<element_type *>(tmp.data_ptr()), ksw, H, W, N); \
            }                                                                                                          \
                                                                                                                       \
            /* Transpose tmp (H x W) -> tmpT (W x H) */                                                                \
            {                                                                                                          \
                /* configure transpose kernel: each y-thread handles 16 rows */                                        \
                dim3 tblock(32, 1);                                                                                    \
                dim3 tgrid(divUp(W, tblock.x), divUp(H, 16));                                                          \
                mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid, tblock>>>(                                      \
                    reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(tmpT.data_ptr()), H, W);  \
            }                                                                                                          \
                                                                                                                       \
            /* Second pass: apply horizontal morphology on transposed image (effectively vertical on original) */      \
            /* when transposed, H' = W, W' = H, use kernel_y as horizontal SE with size ksh */                         \
            {                                                                                                          \
                dim3   block2(THREADS);                                                                                \
                dim3   grid2(divUp(H, block2.x *n_pack), W);                                                           \
                size_t smem_size = (block2.x * n_pack + ksh - 1) * sizeof(element_type);                               \
                morphology_h_##kname##_kernel<element_type, MinOp<element_type>>                                       \
                    <<<grid2, block2, smem_size, 0>>>(reinterpret_cast<element_type *>(tmpT.data_ptr()),               \
                                                      reinterpret_cast<element_type *>(tmp.data_ptr()), ksh, W, H, N); \
            }                                                                                                          \
                                                                                                                       \
            /* Transpose back tmp (now holds transposed final result W x H) -> dst (H x W) */                          \
            {                                                                                                          \
                dim3 tblock2(32, 1);                                                                                   \
                dim3 tgrid2(divUp(H, tblock2.x), divUp(W, 16));                                                        \
                mat_transpose_u8x16_coalesced_write_2d_kernel<<<tgrid2, tblock2>>>(                                    \
                    reinterpret_cast<uint8_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H);   \
            }                                                                                                          \
        }                                                                                                              \
    }

TORCH_BINDING_MORPHOLOGY(erode, torch::kUInt8, uint8_t, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY(dilate, torch::kUInt8, uint8_t, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_TEMPLATE(erode, no_cond, torch::kUInt8, uint8_t, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_TEMPLATE(dilate, no_cond, torch::kUInt8, uint8_t, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_SHARED(erode, shared, torch::kUInt8, uint8_t, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_SHARED(dilate, shared, torch::kUInt8, uint8_t, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_SEPARABLE(erode, torch::kUInt8, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_SEPARABLE(dilate, torch::kUInt8, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED(erode, shared, torch::kUInt8, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED(dilate, shared, torch::kUInt8, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_SEPARABLE2_SHARED(erode, shared, torch::kUInt8, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_SEPARABLE2_SHARED(dilate, shared, torch::kUInt8, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_VEC(erode, shared_vec4_u8, torch::kUInt8, uint8_t, MinOp<uint8_t>, 4)
TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_VEC(dilate, shared_vec4_u8, torch::kUInt8, uint8_t, MaxOp<uint8_t>, 4)

TORCH_BINDING_MORPHOLOGY_SEPARABLE2_SHARED(erode, shared_vec4_u8, torch::kUInt8, uint8_t, MinOp<uint8_t>, 4)
TORCH_BINDING_MORPHOLOGY_SEPARABLE2_SHARED(dilate, shared_vec4_u8, torch::kUInt8, uint8_t, MaxOp<uint8_t>, 4)

TORCH_BINDING_MORPHOLOGY_SEPARABLE_T(erode, torch::kUInt8, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_SEPARABLE_T(dilate, torch::kUInt8, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_T(erode, shared, torch::kUInt8, uint8_t, MinOp<uint8_t>, 1)
TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_T(dilate, shared, torch::kUInt8, uint8_t, MaxOp<uint8_t>, 1)

TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_T(erode, shared_vec4, torch::kUInt8, uint8_t, MinOp<uint8_t>, 4)
TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_T(dilate, shared_vec4, torch::kUInt8, uint8_t, MaxOp<uint8_t>, 4)

TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_T(erode, shared_vec4_u8, torch::kUInt8, uint8_t, MinOp<uint8_t>, 4)
TORCH_BINDING_MORPHOLOGY_SEPARABLE_SHARED_T(dilate, shared_vec4_u8, torch::kUInt8, uint8_t, MaxOp<uint8_t>, 4)

/***************************** OPEN & CLOSE ************************************/

TORCH_BINDING_MORPHOLOGY_OPEN(open, torch::kUInt8, uint8_t, uint8_t, 1)
TORCH_BINDING_MORPHOLOGY_CLOSE(close, torch::kUInt8, uint8_t, uint8_t, 1)

TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE(open, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE(close, torch::kUInt8, uint8_t, 1)

TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE_SHARED(open, shared, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE_SHARED(close, shared, torch::kUInt8, uint8_t, 1)

TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE2_SHARED(open, shared, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE2_SHARED(close, shared, torch::kUInt8, uint8_t, 1)

TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE_SHARED_VEC(open, shared_vec4_u8, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE_SHARED_VEC(close, shared_vec4_u8, torch::kUInt8, uint8_t, 4)

TORCH_BINDING_MORPHOLOGY_OPEN_SEPARABLE_SHARED_T(open, shared_vec4_u8, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_MORPHOLOGY_CLOSE_SEPARABLE_SHARED_T(close, shared_vec4_u8, torch::kUInt8, uint8_t, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(erode_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_no_cond_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_no_cond_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_shared_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_shared_uint8_t_uint8_t)

    TORCH_BINDING_COMMON_EXTENSION(erode_separable_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_separable_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_separable2_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable2_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_separable_shared_vec4_u8_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable_shared_vec4_u8_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_separable2_shared_vec4_u8_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable2_shared_vec4_u8_uint8_t)

    TORCH_BINDING_COMMON_EXTENSION(erode_separable_T_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable_T_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_separable_T_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable_T_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_separable_T_shared_vec4_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable_T_shared_vec4_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(erode_separable_T_shared_vec4_u8_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(dilate_separable_T_shared_vec4_u8_uint8_t)

    TORCH_BINDING_COMMON_EXTENSION(open_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(close_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(open_separable_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(close_separable_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(open_separable_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(close_separable_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(open_separable2_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(close_separable2_shared_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(open_separable_shared_vec4_u8_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(close_separable_shared_vec4_u8_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(open_separable_T_shared_vec4_u8_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(close_separable_T_shared_vec4_u8_uint8_t)
}