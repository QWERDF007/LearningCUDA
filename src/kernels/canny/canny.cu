#include "common.cuh"

template<typename T, typename CT, typename KT, int CH>
__global__ void filter_h_kernel(const T *__restrict__ src, CT *__restrict__ dst, const KT *__restrict__ kernel,
                                const int ksize, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int y = tid / W;
    const int x = tid % W;

    const int anchor = ksize / 2;

    const int base = tid * CH;

    // 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        float sum = 0;
        // 卷积计算
        for (int k = 0; k < ksize; ++k)
        {
            int xx = border_reflect_101(x + k - anchor, W);
            sum += src[(y * W + xx) * CH + c] * kernel[k];
        }
        dst[base + c] = saturate_cast<CT>(sum); // cv2.Sobel(img)
    }
}

template<typename T, typename CT, typename KT, int CH>
__global__ void filter_v_kernel(const T *__restrict__ src, CT *__restrict__ dst, const KT *__restrict__ kernel,
                                const int ksize, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int y = tid / W;
    const int x = tid % W;

    const int anchor = ksize / 2;

    const int base = tid * CH;

    // 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        float sum = 0;
        // 卷积计算
        for (int k = 0; k < ksize; ++k)
        {
            int yy = border_reflect_101(y + k - anchor, H);
            sum += src[(yy * W + x) * CH + c] * kernel[k];
        }
        dst[base + c] = saturate_cast<CT>(sum); // cv2.Sobel(img)
    }
}

template<typename T, typename CT, typename KT, int CH>
__global__ void filter_2D_kernel(const T *__restrict__ src, CT *__restrict__ dst, const KT *__restrict__ kernel,
                                 const int ksh, const int ksw, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int y = tid / W;
    const int x = tid % W;

    const int half_w = ksw / 2;
    const int half_h = ksh / 2;

    const int base = tid * CH;

    // 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        float sum = 0;
        // 卷积计算
        for (int ky = -half_h; ky <= half_h; ++ky)
        {
            int yy = border_reflect_101(y + ky, H);
            for (int kx = -half_w; kx <= half_w; ++kx)
            {
                int xx   = border_reflect_101(x + kx, W);
                KT  kval = kernel[(ky + half_h) * ksw + (kx + half_w)];
                sum += src[(yy * W + xx) * CH + c] * kval;
            }
        }
        dst[base + c] = saturate_cast<CT>(sum); // cv2.Sobel(img)
    }
}

////////////////////////////////////////////////////////////////////////////////////
// 计算梯度幅值 (L1 / L2)
////////////////////////////////////////////////////////////////////////////////////
__global__ void magnitudeKernel(const int16_t *dx, const int16_t *dy, int32_t *mag, const int width, const int height,
                                const bool L2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    int gx = dx[idx];
    int gy = dy[idx];

    if (L2)
        mag[idx] = gx * gx + gy * gy;
    else
        mag[idx] = abs(gx) + abs(gy);
}

static const int TG22 = 13573;

////////////////////////////////////////////////////////////////////////////////////
// 非极大值抑制 + 双阈值
// 输出 map:
//   0 = 非边
//   1 = 弱边
//   2 = 强边
////////////////////////////////////////////////////////////////////////////////////
__global__ void nmsThresholdKernel(const int16_t *dx, const int16_t *dy, const int32_t *mag, int32_t *map,
                                   const int width, const int height, const int low, const int high, bool L2)
{
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx <= 0 || tx >= width - 1 || ty <= 0 || ty >= height - 1)
        return;

    const int idx = ty * width + tx;

    const int m = mag[idx]; // 梯度幅值
    if (m <= low)
    {
        map[idx] = 0; // 低于低阈值，标记为非边缘
        return;
    }

    short xs = dx[idx];
    short ys = dy[idx];

    int x = (int)abs(xs);
    int y = (int)abs(ys) << 15;

    int tg22x = x * TG22; // TG22=13573，tan(22.5°)的近似值

    // 幅值环形缓冲区：_mag_p(前一行), _mag_a(当前行), _mag_n(后一行)

    // 根据梯度方向判断局部最大值
    bool isMax = false;
    if (y < tg22x) // 水平方向（0°/180°）
    {
        isMax = (m > mag[idx - 1] && m >= mag[idx + 1]);
    }
    else
    {
        int tg67x = tg22x + (x << 16); // tan(67.5°)的近似值

        if (y > tg67x) // 垂直方向（90°/270°）
        {
            isMax = (m > mag[idx - width] && m >= mag[idx + width]);
        }
        else // 对角线方向（45°/135°或-45°/-135°）
        {
            int s = (xs ^ ys) < 0 ? -1 : 1;
            isMax = (m > mag[idx - width - s] && m > mag[idx + width + s]);
        }
    }

    if (isMax)
    {
        map[idx] = m > high ? 2 : 1;
    }

    // float gx = dx[idx];
    // float gy = dy[idx];

    // float ang = atan2f(gy, gx) * 180.0f / 3.14159265f;
    // if (ang < 0)
    //     ang += 180.0f;

    // float m1, m2;

    // if (ang < 22.5f || ang >= 157.5f)
    // {
    //     m1 = mag[idx - 1];
    //     m2 = mag[idx + 1];
    // }
    // else if (ang < 67.5f)
    // {
    //     m1 = mag[idx - width + 1];
    //     m2 = mag[idx + width - 1];
    // }
    // else if (ang < 112.5f)
    // {
    //     m1 = mag[idx - width];
    //     m2 = mag[idx + width];
    // }
    // else
    // {
    //     m1 = mag[idx - width - 1];
    //     m2 = mag[idx + width + 1];
    // }

    // if (m < m1 || m < m2)
    // {
    //     map[idx] = 0;
    //     return;
    // }

    // map[idx] = (m > high ? 2 : 1);
}

////////////////////////////////////////////////////////////////////////////////////
// Hysteresis: weak → strong
////////////////////////////////////////////////////////////////////////////////////
__global__ void hysteresisKernel(int32_t *map, int32_t *const d_changed, int width, const int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
        return;

    int idx = y * width + x;

    if (map[idx] != 1) // 非弱边
        return;

    // 8邻域
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0)
                continue;
            if (map[idx + dy * width + dx] == 2)
            {
                map[idx] = 2;
                atomicExch(d_changed, 1);
                return;
            }
        }
}

__global__ void buildInitialQueue(const int32_t *map, int *queue, int *queue_tail, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    if (map[idx] == 2)
    {
        int pos    = atomicAdd(queue_tail, 1);
        queue[pos] = idx;
    }
}

__global__ void hysteresisDevice(int32_t *map, int *queue, int *queue_head, int *queue_tail, int width, int height)
{
    while (true)
    {
        int pos = atomicAdd(queue_head, 1);
        if (pos >= *queue_tail)
            break;

        int idx = queue[pos];

        int x = idx % width;
        int y = idx / width;

        // 8 neighbors
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0)
                    continue;

                int nx = x + dx;
                int ny = y + dy;
                if (nx <= 0 || nx >= width - 1 || ny <= 0 || ny >= height - 1)
                    continue;

                int nidx = ny * width + nx;

                // weak → strong
                if (map[nidx] == 1)
                {
                    // promote
                    if (atomicCAS((int *)&map[nidx], 1, 2) == 1)
                    {
                        // push into queue
                        int qpos    = atomicAdd(queue_tail, 1);
                        queue[qpos] = nidx;
                    }
                }
            }
    }
}

__device__ int d_changed;

__global__ void hysteresisKernelDevice(int32_t *mapA, int32_t *mapB, int width, int height)
{
    int N = width * height;

    while (true)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            d_changed = 0;
        __syncthreads();

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N)
        {
            int32_t v = mapA[idx];

            if (v == 1)
            {
                int x = idx % width;
                int y = idx / width;

                if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
                {
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dy == 0)
                                continue;
                            if (mapA[idx + dy * width + dx] == 2)
                            {
                                v = 2;
                                atomicExch(&d_changed, 1);
                                goto done;
                            }
                        }
                }
            }

done:
            mapB[idx] = v;
        }

        __syncthreads();

        if (d_changed == 0)
            break;

        // swap
        int32_t *tmp = mapA;
        mapA         = mapB;
        mapB         = tmp;

        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////////
// 最终输出 255 / 0
////////////////////////////////////////////////////////////////////////////////////
__global__ void finalKernel(const int32_t *map, uint8_t *dst, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx  = y * width + x;
    dst[idx] = (map[idx] == 2 ? 255 : 0);
}

void cudaCanny(torch::Tensor src, torch::Tensor dst, torch::Tensor kernelx, torch::Tensor kernely, double low_thresh,
               double high_thresh, int apertureSize, bool L2gradient)
{
    const int H = src.size(0);
    const int W = src.size(1);
    const int N = H * W;

    auto mag_options     = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    auto map_options     = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    auto changed_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    auto dx_options      = torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCUDA, 0);
    auto dy_options      = torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCUDA, 0);

    torch::Tensor d_mag  = torch::zeros({N}, mag_options);
    torch::Tensor d_map  = torch::zeros({N}, map_options);
    torch::Tensor d_mapB = torch::zeros({N}, map_options);

    torch::Tensor d_changed    = torch::zeros({1}, changed_options);
    torch::Tensor d_queue      = torch::zeros({N}, changed_options);
    torch::Tensor d_queue_head = torch::zeros({1}, changed_options);
    torch::Tensor d_queue_tail = torch::zeros({1}, changed_options);

    torch::Tensor dx = torch::zeros({N}, dx_options);
    torch::Tensor dy = torch::zeros({N}, dy_options);

    if (L2gradient)
    {
        low_thresh  = std::min(32767.0, low_thresh);
        high_thresh = std::min(32767.0, high_thresh);

        if (low_thresh > 0)
            low_thresh *= low_thresh;
        if (high_thresh > 0)
            high_thresh *= high_thresh;
    }

    int low  = std::floor(low_thresh);
    int high = std::floor(high_thresh);

    {
        dim3 block(THREADS);
        dim3 grid(divUp(N, THREADS));

        filter_2D_kernel<uint8_t, int16_t, int8_t, 1>
            <<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()), reinterpret_cast<int16_t *>(dx.data_ptr()),
                              reinterpret_cast<int8_t *>(kernelx.data_ptr()), apertureSize, apertureSize, H, W, N);
        filter_2D_kernel<uint8_t, int16_t, int8_t, 1>
            <<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()), reinterpret_cast<int16_t *>(dy.data_ptr()),
                              reinterpret_cast<int8_t *>(kernely.data_ptr()), apertureSize, apertureSize, H, W, N);

        // filter_h_kernel<uint8_t, int16_t, int8_t, 1>
        //     <<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()), reinterpret_cast<int16_t *>(tmp.data_ptr()),
        //                       reinterpret_cast<int8_t *>(kernelx.data_ptr()), ksw, H, W, N);
        // filter_v_kernel<int16_t, int16_t, int8_t, 1>
        //     <<<grid, block>>>(reinterpret_cast<int16_t *>(tmp.data_ptr()), reinterpret_cast<int16_t *>(dst.data_ptr()),
        //                       reinterpret_cast<int8_t *>(kernely.data_ptr()), ksh, H, W, N);
    }

    dim3 block(16, 16);
    dim3 grid(divUp(W, block.x), divUp(H, block.y));

    // 1. 幅值
    magnitudeKernel<<<grid, block>>>(reinterpret_cast<int16_t *>(dx.data_ptr()),
                                     reinterpret_cast<int16_t *>(dy.data_ptr()),
                                     reinterpret_cast<int32_t *>(d_mag.data_ptr()), W, H, L2gradient);

    // 2. NMS + threshold
    nmsThresholdKernel<<<grid, block>>>(reinterpret_cast<int16_t *>(dx.data_ptr()),
                                        reinterpret_cast<int16_t *>(dy.data_ptr()),
                                        reinterpret_cast<int32_t *>(d_mag.data_ptr()),
                                        reinterpret_cast<int32_t *>(d_map.data_ptr()), W, H, low, high, L2gradient);

    // 3. Hysteresis
    int changed = 1;
    while (changed)
    {
        cudaMemset(reinterpret_cast<int32_t *>(d_changed.data_ptr()), 0, sizeof(int32_t));

        hysteresisKernel<<<grid, block>>>(reinterpret_cast<int32_t *>(d_map.data_ptr()),
                                          reinterpret_cast<int32_t *>(d_changed.data_ptr()), W, H);

        cudaMemcpy(&changed, reinterpret_cast<int32_t *>(d_changed.data_ptr()), sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
    }

    // {
    //     // 1. 初始化队列
    //     buildInitialQueue<<<grid, block>>>(reinterpret_cast<int32_t *>(d_map.data_ptr()),
    //                                        reinterpret_cast<int32_t *>(d_queue.data_ptr()),
    //                                        reinterpret_cast<int32_t *>(d_queue_tail.data_ptr()), W, H);

    //     // 2. 完整 hysteresis（一次 kernel）
    //     hysteresisDevice<<<128, 256>>>(reinterpret_cast<int32_t *>(d_map.data_ptr()),
    //                                    reinterpret_cast<int32_t *>(d_queue.data_ptr()),
    //                                    reinterpret_cast<int32_t *>(d_queue_head.data_ptr()),
    //                                    reinterpret_cast<int32_t *>(d_queue_tail.data_ptr()), W, H);
    // }

    // {
    //     dim3 block(THREADS);
    //     dim3 grid(divUp(N, THREADS));
    //     hysteresisKernelDevice<<<grid, block>>>(reinterpret_cast<int32_t *>(d_map.data_ptr()),
    //                                             reinterpret_cast<int32_t *>(d_mapB.data_ptr()), W, H);
    // }

    // 4. Final: map → dst
    finalKernel<<<grid, block>>>(reinterpret_cast<int32_t *>(d_map.data_ptr()),
                                 reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(cudaCanny)
}
