#include "common.cuh"

// BORDER_REFLECT: fedcba|abcdefgh|hgfedcb
__device__ int reflect(int coord, int size)
{
    if (coord < 0)
        return -coord - 1;
    if (coord >= size)
        return 2 * size - coord - 1;
    return coord;
}


// BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba
__device__ int reflect_101(int coord, int size)
{
    if (coord < 0)
        return -coord;
    if (coord >= size)
        return 2 * size - coord - 2;
    return coord;
}

__global__ void blur_u8_kernel(uint8_t *in, uint8_t *out, const int ks_w, const int ks_h, const int img_w,
                               const int img_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_w || y >= img_h)
        return;

    const int idx = y * img_w + x;

    int half_w = ks_w / 2;
    int half_h = ks_h / 2;
    int sum    = 0;
    float count  = ks_w * ks_h;  // 所有像素都会被处理

    for (int ky = -half_h; ky <= half_h; ++ky)
    {
        int yy = reflect_101(y + ky, img_h);  // 使用BORDER_REFLECT_101
        for (int kx = -half_w; kx <= half_w; ++kx)
        {
            int xx = reflect_101(x + kx, img_w);  // 使用BORDER_REFLECT_101
            sum += in[yy * img_w + xx];
        }
    }

    out[idx] = (uint8_t)roundf(sum / count);
}

void blur_u8(torch::Tensor in, const int ksz, torch::Tensor out)
{
    CHECK_TORCH_TENSOR_DTYPE(in, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(in)
    CHECK_TORCH_TENSOR_DEVICE(out)

    const int H = in.size(0);
    const int W = in.size(1);
    const int N = H * W;
    dim3      block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3      grid(divUp(W, block.x), divUp(H, block.y));

    blur_u8_kernel<<<grid, block>>>(reinterpret_cast<uint8_t *>(in.data_ptr()),
                                    reinterpret_cast<uint8_t *>(out.data_ptr()), ksz, ksz, W, H);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(blur_u8)
}