#include "common.cuh"

__global__ void genMotionKernelCUDA(float *dst, int width, int height, double angle, int dist)
{
    double sinVal = sin(angle * M_PI / 180.0);
    double cosVal = cos(angle * M_PI / 180.0);

    int half = dist / 2;
    int cx   = width / 2;
    int cy   = height / 2;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > half)
        return;

    int offx = __float2int_rn(tid * cosVal);
    int offy = __float2int_rn(tid * sinVal);

    int pos1 = (cy + offy) * width + (cx + offx);
    int pos2 = (cy - offy) * width + (cx - offx);
    if (pos1 < width * height)
        dst[pos1] = 1.f;
    if (pos2 < width * height)
        dst[pos2] = 1.f;
}

__global__ void conv2d_kernel(const uint8_t *src, uint8_t *dst, const float *kernel, const int kH, const int kW,
                              const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int y = tid / W;
    const int x = tid % W;

    int kCenterX = kW / 2;
    int kCenterY = kH / 2;

    float sum = 0.0f;
    for (int m = 0; m < kH; m++)
    {
        int mm = kH - 1 - m; // kernel flipped
        for (int n = 0; n < kW; n++)
        {
            int nn = kW - 1 - n;

            int yy = y + (m - kCenterY);
            int xx = x + (n - kCenterX);

            if (xx >= 0 && xx < W && yy >= 0 && yy < H)
                sum += src[yy * W + xx] * kernel[mm * kW + nn];
        }
    }
    dst[y * W + x] = saturate_cast<uint8_t>(sum);
}

void motion_blur(torch::Tensor src, torch::Tensor dst, torch::Tensor kernel, const int KH, const int KW)
{
    CHECK_TORCH_TENSOR_DTYPE(src, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DTYPE(dst, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)
    CHECK_TORCH_TENSOR_DEVICE(kernel)
    const int H  = src.size(0);
    const int W  = src.size(1);
    const int CH = src.dim() == 2 ? 1 : src.size(2);
    const int N  = H * W;

    dim3 block(THREADS);
    dim3 grid(divUp(N, THREADS));

    conv2d_kernel<<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                   reinterpret_cast<uint8_t *>(src.data_ptr()),
                                   reinterpret_cast<float *>(src.data_ptr()), KH, KW, H, W, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(motion_blur)
}