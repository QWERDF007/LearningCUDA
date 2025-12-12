#include "common.cuh"

__device__ __forceinline__ void init_labels_8(uint32_t *g_labels, const uint8_t *g_image, const uint32_t ix,
                                              const uint32_t iy, const int W, const int H)
{
    // 1. 读取当前像素值
    const uint8_t pyx = g_image[iy * W + ix];

    // 2. 预计算邻域连通性（8连通逻辑）
    // 只有当坐标在图像范围内且像素值相等时，才视为连通

    // 上方像素 (x, y-1)
    const bool nym1x = (iy > 0) ? (pyx == g_image[(iy - 1) * W + ix]) : false;
    // 左方像素 (x-1, y)
    const bool nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*W + ix - 1]) : false;
    // 左上像素 (x-1, y-1)
    const bool nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy - 1) * W + ix - 1]) : false;
    // 右上像素 (x+1, y-1)
    const bool nym1xp1 = ((iy > 0) && (ix < W - 1)) ? (pyx == g_image[(iy - 1) * W + ix + 1]) : false;

    // 3. 确定初始标签
    // 默认情况下，标签指向自己（线性索引 = y * W + x）
    // 这里的策略是：如果有连通的邻居，尝试指向邻居的索引。
    // 代码中的覆盖顺序暗示了优先级：左上 > 上 > 右上 > 左
    // 目的是尽可能传播更小的索引（通常上方和左方的索引更小），建立初步的树结构。

    uint32_t label;

    // 如果左方连通，继承左方标签，否则用自身
    label = (nyxm1) ? iy * W + ix - 1 : iy * W + ix;
    // 如果右上连通，优先级更高，继承右上标签
    label = (nym1xp1) ? (iy - 1) * W + ix + 1 : label;
    // 如果上方连通，优先级更高，继承上方标签
    label = (nym1x) ? (iy - 1) * W + ix : label;
    // 如果左上连通，优先级最高，继承左上标签
    label = (nym1xm1) ? (iy - 1) * W + ix - 1 : label;

    g_labels[iy * W + ix] = label;
}

__device__ __forceinline__ void init_labels_4(uint32_t *g_labels, const uint8_t *g_image, const uint32_t ix,
                                              const uint32_t iy, const int W, const int H)
{
    // 读取当前像素值
    const uint8_t pyx = g_image[iy * W + ix];

    // 当前是否和上方像素 (x, y-1) 连通
    const bool nym1x = (iy > 0) ? (pyx == g_image[(iy - 1) * W + ix]) : false;
    // 当前是否和左方像素 (x-1, y) 连通
    const bool nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*W + ix - 1]) : false;

    uint32_t label;

    // 如果左方连通，继承左方，否则用自身
    label = (nyxm1) ? iy * W + ix - 1 : iy * W + ix;

    // 如果上方连通，继承上方（优先级更高，覆盖左方）
    // 这里谁优先级高不影响正确性，只会影响初始树的形状
    label = (nym1x) ? (iy - 1) * W + ix : label;

    g_labels[iy * W + ix] = label;
}

/**
 * @brief CUDA核函数：初始化每个像素的连通域标签 （8连通）
 * 
 * 核心逻辑：每个线程处理一个像素。查看该像素的8邻域（实际上只需要查看“过去”的邻居：左、左上、上、右上）。
 * 如果有连通的邻居，就继承邻居的标签（实际上是邻居的线性索引）。
 * 
 * @param g_labels 输出标签数组，初始状态下每个像素存储其父节点的索引
 * @param g_image  输入二值图像，0表示背景，非0表示前景
 * @param W  图像宽度（列数）
 * @param H  图像高度（行数）
 */
template<int connectivity>
__global__ void init_labels(uint32_t *g_labels, const uint8_t *g_image, const int W, const int H)
{
    // 计算全局唯一的线程索引，对应像素坐标 (ix, iy)
    const uint32_t ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t iy = (blockIdx.y * blockDim.y) + threadIdx.y;

    // 边界检查
    if (ix >= W || iy >= H)
        return;

    if constexpr (connectivity == 8)
    {
        init_labels_8(g_labels, g_image, ix, iy, W, H);
    }
    else if constexpr (connectivity == 4)
    {
        init_labels_4(g_labels, g_image, ix, iy, W, H);
    }
}

/**
 * @brief 查找连通域标签链的根节点。
 * 
 * 这是一个典型的并查集（Union-Find）中的 Find 操作。
 * 它沿着 g_labels 数组中存储的父节点索引一直向上查找，直到找到根节点（Label == Parent）。
 * 注意：这里没有做路径压缩（Path Compression），路径压缩在 resolve_labels 核函数中完成。
 * 
 * @param labels 连通域标签数组，labels[i]表示i号像素当前的父标签
 * @param label 需要查找根节点的起始标签
 */
inline __device__ uint32_t find_root(uint32_t *labels, uint32_t label)
{
    // 取出当前label对应的父标签
    uint32_t next = labels[label];

    // 循环直到找到根节点 (next == label)
    // 根节点存储自身的 idx, 而其他节点存储的是其父节点的 idx
    while (label != next)
    {
        label = next;
        next  = labels[label];
    }

    return label;
}

/**
 * @brief 解析标签（路径压缩/扁平化）
 * 
 * 将 g_labels 中的每个像素标签直接更新为其所属树的根节点。
 * 经过 init_labels 后，可能形成长链（A->B->C），此函数将其打平（A->C, B->C）。
 * 
 * @param g_labels 输出标签数组
 */
__global__ void resolve_labels(uint32_t *g_labels, const int W, const int H)
{
    // Calculate index
    const int      x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int      y   = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t idx = y * W + x;

    if (idx >= (H * W))
        return;
    // 找到根并直接赋值回数组，实现完全路径压缩
    // 实现（A->B->C）到（A->C, B->C）
    g_labels[idx] = find_root(g_labels, g_labels[idx]);
}

/**
 * @brief Label Reduction (标签规约/合并)。并不是简单的赋值，而是查找两个标签的根节点，
 *        并原子地将其中较大的根指向较小的根。这实现了并查集中的 Union 操作。
 * 
 * @param g_labels 全局标签数组
 * @param label1 第一个像素的标签
 * @param label2 第二个像素的标签
 * @return 返回合并后的共同根标签
 */
inline __device__ uint32_t reduction(uint32_t *g_labels, uint32_t label1, uint32_t label2)
{
    // 1. 预读取父节点
    // 如果两个标签本身就不相等，尝试读取它们的父节点；否则设为0（实际上如果不等，下面循环会处理）
    uint32_t next1 = (label1 != label2) ? g_labels[label1] : 0;
    uint32_t next2 = (label1 != label2) ? g_labels[label2] : 0;

    // 2. 查找 label1 的根节点
    // 循环条件：label1 还没追上 label2，且 label1 还不是根节点（label1 != next1）
    while ((label1 != label2) && (label1 != next1))
    {
        // 移动到父节点
        label1 = next1;
        // 获取新的父节点的父节点
        next1 = g_labels[label1];
    }

    // 3. 查找 label2 的根节点
    // 逻辑同上，找到 label2 所属树的根
    while ((label1 != label2) && (label2 != next2))
    {
        label2 = next2;
        next2  = g_labels[label2];
    }

    // 此时，label1 和 label2 都变成了各自树的根节点（或者是同一个根）

    uint32_t label3;

    // 4. 合并循环 (Union Loop)
    // 只要两个根节点不同，就必须把它们合并
    while (label1 != label2)
    {
        // 保证 label1 是较大的那个，label2 是较小的那个
        // 我们的策略是将“大ID”挂载到“小ID”下面（Min-label priority）
        if (label1 < label2)
        {
            // 使用异或交换算法 (XOR Swap) 交换 label1 和 label2
            // 效果等同于: temp = a; a = b; b = temp;
            label1 = label1 ^ label2;
            label2 = label1 ^ label2;
            label1 = label1 ^ label2;
        }

        // 关键步骤：原子最小值操作 (Atomic Min)
        // 尝试将 g_labels[label1] 的值修改为 label2。
        // atomicMin 返回修改前 g_labels[label1] 的旧值，存入 label3。
        //
        // 这里的逻辑是：试图让 label1（大）的父节点指向 label2（小）。
        label3 = atomicMin(&g_labels[label1], label2);

        // 5. 检查合并结果与路径压缩
        // 情况 A: 如果 label3 == label1，说明 g_labels[label1] 原来就是指向自己的（它是根），
        //         原子操作成功将其修改为 label2。合并成功！
        //         此时我们将 label1 更新为 label2，循环条件 (label1 != label2) 不再满足，退出循环。
        //
        // 情况 B: 如果 label3 != label1，说明在我们尝试修改之前，已经有其他线程
        //         把 g_labels[label1] 修改成了 label3（且 label3 < label1）。
        //         这意味着 label1 已经不再是根了，它的父节点变成了 label3。
        //         原子操作失败（或者说我们不仅要跟 label2 合并，还要跟 label3 合并）。
        //         此时，我们需要继续循环，尝试合并新的父节点 label3 和 label2。
        label1 = (label1 == label3) ? label2 : label3;
    }

    // 返回合并后的统一根标签
    return label1;
}

__device__ __forceinline__ void label_reduction_8(uint32_t *g_labels, const uint8_t *g_image, const uint32_t ix,
                                                  const uint32_t iy, const int W, const int H)
{
    const uint8_t pyx = g_image[iy * W + ix];

    // 检查与正上方 (x, y-1) 的连通性
    const bool nym1x = (iy > 0) ? (pyx == g_image[(iy - 1) * W + ix]) : false;

    // 优化策略：如果正上方已经连通，init_labels 通常能很好地处理传播。
    // 冲突主要发生在正上方不连通，但对角线方向连通的情况。
    if (nym1x)
        return;
    // 获取其他邻域的连通性
    // 左上 (x-1, y-1)
    const bool nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy - 1) * W + ix - 1]) : false;
    // 左 (x-1, y)
    const bool nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*W + ix - 1]) : false;
    // 右上 (x+1, y-1)
    const bool nym1xp1 = ((iy > 0) && (ix < W - 1)) ? (pyx == g_image[(iy - 1) * W + ix + 1]) : false;

    if (!nym1xp1)
        return;
    // 如果与右上角连通，这通常是潜在冲突的来源
    // 情况 1: (左上连通 且 左连通) 或者 (左上连通 且 左不连通)
    // 实际上就是：只要左上 (Top-Left) 连通，就需要检查。
    // 这里的逻辑主要在处理两棵树汇聚到当前像素的情况。
    if ((nym1xm1 && nyxm1) || (nym1xm1 && !nyxm1))
    {
        // 获取当前像素的标签和右上角像素的标签
        // 此时 g_labels 已经在 resolve_labels 中被更新为根节点了
        uint32_t label1 = g_labels[(iy)*W + ix];
        uint32_t label2 = g_labels[(iy - 1) * W + ix + 1];

        // 合并这两个标签所在的集合
        reduction(g_labels, label1, label2);
    }

    // 情况 2: 左上不连通，但左边连通
    if (!nym1xm1 && nyxm1)
    {
        // Get labels
        // 当前 (x, y)
        uint32_t label1 = g_labels[(iy)*W + ix];
        // 左 (x-1, y)
        uint32_t label2 = g_labels[(iy)*W + ix - 1];

        // 合并
        reduction(g_labels, label1, label2);
    }
}

__device__ __forceinline__ void label_reduction_4(uint32_t *g_labels, const uint8_t *g_image, const uint32_t ix,
                                                  const uint32_t iy, const int W, const int H)
{
    const uint8_t pyx = g_image[iy * W + ix];

    // 1. 检查上方连通性
    const bool nym1x = (iy > 0) ? (pyx == g_image[(iy - 1) * W + ix]) : false;
    // 2. 检查左方连通性
    const bool nyxm1 = (ix > 0) ? (pyx == g_image[(iy)*W + ix - 1]) : false;

    // 4连通仅在同时连接 上方 和 左方 时发生冲突
    // 如果只连通上方，init_labels 已经设为上方的 label 了。
    // 如果只连通左方，init_labels 已经设为左方的 label 了。
    // 只有当两者都连通时，init_labels 选了其中一个（比如上方），
    // 但我们需要把“被抛弃”的那个（左方）也合并进来。
    if (nym1x && nyxm1)
    {
        // 获取上方像素的标签
        uint32_t label1 = g_labels[(iy - 1) * W + ix];
        // 获取左方像素的标签
        uint32_t label2 = g_labels[(iy)*W + ix - 1];

        // 如果两个标签不同（意味着它们属于不同的根），则合并
        if (label1 != label2)
        {
            reduction(g_labels, label1, label2);
        }
    }
}

/**
 * @brief 标签规约 / 合并（处理8连通中的冲突）
 * 
 * 在 init_labels 中，由于是局部判断，可能会出现同一个连通域被标记为两个不同标签的情况（比如 U 形物体）。
 * 此函数检查特定的“临界配置（Critical Configurations）”，发现连通但标签不同的部分，并调用 reduction 进行合并。
 * 
 * @param g_labels 输出标签数组
 * @param g_image 输入二值图像
 */
template<int connectivity>
__global__ void label_reduction(uint32_t *g_labels, const uint8_t *g_image, const int W, const int H)
{
    const uint32_t iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
    const uint32_t ix = ((blockIdx.x * blockDim.x) + threadIdx.x);

    // 边界检查
    if (ix >= W || iy >= H)
        return;
    // Compare Image Values

    if constexpr (connectivity == 8)
    {
        label_reduction_8(g_labels, g_image, ix, iy, W, H);
    }
    else if constexpr (connectivity == 4)
    {
        label_reduction_4(g_labels, g_image, ix, iy, W, H);
    }
}

/**
 * @brief 最终处理：区分背景与前景
 * 
 * 将背景像素的标签设为 0。
 * 将前景像素的标签 +1。这样做的目的是腾出 0 给背景，并且确保所有前景标签 > 0。
 * 注意：这一步假设之前的 resolve_labels 已经确保 g_labels 存储的是根节点。
 */
__global__ void resolve_background(uint32_t *g_labels, const uint8_t *g_image, const int W, const int H)
{
    // Calculate index
    const uint32_t iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
    const uint32_t ix = ((blockIdx.x * blockDim.x) + threadIdx.x);
    const uint32_t id = iy * W + ix;

    if ((ix < W) && (iy < H))
    {
        g_labels[id] = (g_image[id] > 0) ? g_labels[id] + 1 : 0;
    }
}

__global__ void ccl2binary_kernel(uint32_t *g_labels, uint8_t *g_output, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    g_output[tid] = (g_labels[tid] == 0) ? 0 : 255;
}

void cclToBinary(uint32_t *g_labels, uint8_t *g_output, const int H, const int W, const int CH, cudaStream_t stream)
{
    const int N = H * W;
    dim3      block(256);
    dim3      grid(divUp(N, block.x));
    ccl2binary_kernel<<<grid, block, 0, stream>>>(g_labels, g_output, N);
}

__global__ void init_stats_kernel(uint32_t *g_area, int *g_min_x, int *g_min_y, int *g_max_x, int *g_max_y,
                                  int max_labels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < max_labels)
    {
        g_area[idx] = 0;
        // 初始化 min 为最大整数，max 为最小整数
        g_min_x[idx] = INT_MAX;
        g_min_y[idx] = INT_MAX;
        g_max_x[idx] = INT_MIN;
        g_max_y[idx] = INT_MIN;
    }
}

__global__ void calc_stats_kernel(const uint32_t *g_labels, uint32_t *g_area, int *g_min_x, int *g_min_y, int *g_max_x,
                                  int *g_max_y, const int W, const int H, const int max_labels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H)
    {
        uint32_t idx   = y * W + x;
        uint32_t label = g_labels[idx];

        // 忽略背景 (0) 并检查边界
        if (label > 0 && label < max_labels)
        {
            // 1. 统计面积
            atomicAdd(&g_area[label], 1);

            // 2. 更新边界框 (利用原子 min/max)
            // 注意：atomicMin/Max 支持 int 类型
            atomicMin(&g_min_x[label], x);
            atomicMin(&g_min_y[label], y);

            atomicMax(&g_max_x[label], x);
            atomicMax(&g_max_y[label], y);
        }
    }
}

/**
 * @brief Butterfly Reduction（蝴蝶归约）算法（时间复杂度 O(logN)）
 */
__global__ void calc_stats_kernel_optimized(const uint32_t *g_labels, uint32_t *g_area, int *g_min_x, int *g_min_y,
                                            int *g_max_x, int *g_max_y, const int W, const int H, const int max_labels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 1. Label 预处理
    // 越界或背景像素赋予无效 Label (0xFFFFFFFF)，保证 Warp 内所有线程都“活跃”
    // 这样所有线程都能参与下面的 __match_any_sync 和 shuffle
    uint32_t label    = 0xFFFFFFFF;
    bool     is_valid = (x < W && y < H);

    if (is_valid)
    {
        uint32_t val = g_labels[y * W + x];
        if (val > 0 && val < max_labels)
        {
            label = val;
        }
        else
        {
            is_valid = false; // 背景像素视为无效
        }
    }
    else
    {
        is_valid = false; // 越界像素视为无效
    }

    // 2. Warp 级匹配
    // 生成掩码，标记 Warp 中所有具有相同 label 的线程
    uint32_t mask    = __match_any_sync(0xffffffff, label);
    int      lane_id = threadIdx.x % 32;

    // 3. 局部变量初始化
    // 初始化为当前像素的坐标。无效线程的数据虽然也会被 shuffle，
    // 但因为它们只和同样 label=0xFFFFFFFF 的线程交互，不会污染有效 label 的结果。
    int my_min_x = x;
    int my_max_x = x;
    int my_min_y = y;
    int my_max_y = y;

// 4. Butterfly Reduction (蝴蝶归约)
// 这一步必须所有线程都执行！不能放在 if (leader) 里面。
// 通过 XOR 交换，在 5 步内（32=2^5）让每个线程都获得同组内的极值。
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        // 从“对友”线程获取数据
        int peer_min_x = __shfl_xor_sync(mask, my_min_x, offset);
        int peer_max_x = __shfl_xor_sync(mask, my_max_x, offset);
        int peer_min_y = __shfl_xor_sync(mask, my_min_y, offset);
        int peer_max_y = __shfl_xor_sync(mask, my_max_y, offset);

        // 检查：只有当“对友”线程确实在 mask 中（即具有相同 label）时，才合并数据。
        // (lane_id ^ offset) 计算的是当前 offset 下的对友 lane_id。
        if (mask & (1 << (lane_id ^ offset)))
        {
            my_min_x = min(my_min_x, peer_min_x);
            my_max_x = max(my_max_x, peer_max_x);
            my_min_y = min(my_min_y, peer_min_y);
            my_max_y = max(my_max_y, peer_max_y);
        }
    }

    // 5. 选举 Leader 并写入全局内存
    // __ffs(mask) 找到最低位的 1，作为 Leader
    int leader = __ffs(mask) - 1;

    // 只有有效的前景像素，且是 Leader 的线程才执行原子写
    if (is_valid && lane_id == leader)
    {
        // A. 统计数量：直接计算 mask 中 1 的个数
        int count = __popc(mask);
        atomicAdd(&g_area[label], count);

        // B. 写入归约后的边界框
        // 此时 my_min_x 等变量已经包含了 mask 中所有线程的极值
        atomicMin(&g_min_x[label], my_min_x);
        atomicMin(&g_min_y[label], my_min_y);
        atomicMax(&g_max_x[label], my_max_x);
        atomicMax(&g_max_y[label], my_max_y);
    }
}

__global__ void compact_soa_kernel(const uint32_t *g_in_area, const int *g_in_min_x, const int *g_in_min_y,
                                   const int *g_in_max_x, const int *g_in_max_y, uint32_t *out_ids, uint32_t *out_areas,
                                   int *g_out_x, int *g_out_y, int *g_out_w, int *g_out_h, uint32_t *g_total_count,
                                   int max_labels, int max_output_size)
{
    int label = blockIdx.x * blockDim.x + threadIdx.x;

    if (label < max_labels)
    {
        uint32_t area = g_in_area[label];

        if (area > 0)
        {
            // 获取紧凑数组的索引
            uint32_t pos = atomicAdd(g_total_count, 1);

            // 边界检查：确保 pos 不超出输出数组范围
            if (pos >= max_output_size)
                return;

            int min_x = g_in_min_x[label];
            int min_y = g_in_min_y[label];
            int max_x = g_in_max_x[label];
            int max_y = g_in_max_y[label];

            // 1. Label 和 Area 直接拷贝
            out_ids[pos]   = label;
            out_areas[pos] = area;

            // 2. x, y
            g_out_x[pos] = min_x;
            g_out_y[pos] = min_y;

            // 3. w, h
            // w = max_x - min_x + 1
            // h = max_y - min_y + 1
            g_out_w[pos] = max_x - min_x + 1;
            g_out_h[pos] = max_y - min_y + 1;
        }
    }
}

// void __connectedComponent(const uint8_t *g_image, uint32_t *g_labels, const int W, const int H)
// {
//     dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
//     dim3 grid(divUp(W, block.x), divUp(H, block.y));

//     // 1. 初始化标签：建立初步的连接树
//     init_labels<<<grid, block>>>(g_labels, g_image, W, H);

//     // 2. 解析标签：将树扁平化，让像素直接指向当前的根
//     resolve_labels<<<grid, block>>>(g_labels, W, H);

//     // 3. 标签规约：检查临界点，合并本应连通但标签不同的树
//     label_reduction<<<grid, block>>>(g_labels, g_image, W, H);

//     // 4. 再次解析：规约后，根节点可能发生了变化，再次扁平化
//     resolve_labels<<<grid, block>>>(g_labels, W, H);

//     // 5. 背景处理：格式化输出，背景置0
//     resolve_background<<<grid, block>>>(g_labels, g_image, W, H);
// }

void connectedComponent(torch::Tensor src, torch::Tensor dst, const int connectivity)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(dst, torch::kUInt32)
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)
    const int H  = src.size(0);
    const int W  = src.size(1);
    const int CH = src.dim() == 2 ? 1 : src.size(2);

    TORCH_CHECK(CH == 1, "CH must be 1")

    const uint8_t *g_image  = reinterpret_cast<const uint8_t *>(src.data_ptr());
    uint32_t      *g_labels = reinterpret_cast<uint32_t *>(dst.data_ptr());

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(divUp(W, block.x), divUp(H, block.y));

    // 1. 初始化标签：建立初步的连接树
    if (connectivity == 8)
    {
        init_labels<8><<<grid, block>>>(g_labels, g_image, W, H);
    }
    else if (connectivity == 4)
    {
        init_labels<4><<<grid, block>>>(g_labels, g_image, W, H);
    }

    // 2. 解析标签：将树扁平化，让像素直接指向当前的根
    resolve_labels<<<grid, block>>>(g_labels, W, H);

    // 3. 标签规约：检查临界点，合并本应连通但标签不同的树
    if (connectivity == 8)
    {
        label_reduction<8><<<grid, block>>>(g_labels, g_image, W, H);
    }
    else if (connectivity == 4)
    {
        label_reduction<4><<<grid, block>>>(g_labels, g_image, W, H);
    }

    // 4. 再次解析：规约后，根节点可能发生了变化，再次扁平化
    resolve_labels<<<grid, block>>>(g_labels, W, H);

    // 5. 背景处理：格式化输出，背景置0
    resolve_background<<<grid, block>>>(g_labels, g_image, W, H);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(connectedComponent)
}