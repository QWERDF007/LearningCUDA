#pragma once

/**
 * @brief BORDER_REFLECT:fedcba|abcdefgh|hgfedcb
 * 
 * @param coord 输入坐标值，可能超出有效范围 [0, size)
 * @param size 有效坐标范围的大小
 * @return 经过边界反射处理后的有效坐标值
 */
__device__ __forceinline__ int border_reflect(int coord, int size)
{
    if (coord < 0)
        return -coord - 1;
    if (coord >= size)
        return 2 * size - coord - 1;
    return coord;
}

/**
 * @brief BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba
 * 
 * @param coord 输入坐标值，可能超出有效范围[0, size)
 * @param size 有效坐标范围的大小
 * @return 经过边界反射处理后的有效坐标值
 */
__device__ __forceinline__ int border_reflect_101(int coord, int size)
{
    if (coord < 0)
        return -coord;
    if (coord >= size)
        return 2 * size - coord - 2;
    return coord;
}

/**
 * @brief BORDER_REFLECT_101边界处理函数的优化版本
 * 使用条件运算符替代if-else分支，可能在某些GPU架构上有更好的性能
 * 优化成 selp 指令（无显式分支），减少 divergence
 *
 * @note 性能优化高度依赖于具体的硬件架构和软件环境
 * 在 win11 + wsl Ubuntu 22.04 + 4090 vs  Ubuntu 18.04 + 4090D 测试下, 4090 无分支版本更慢
 * 
 * @param coord 输入坐标值，可能超出有效范围[0, size)
 * @param size 有效坐标范围的大小，坐标应在[0, size)范围内
 * @return 经过边界反射处理后的有效坐标值
 */
__device__ __forceinline__ int border_reflect_101_no_branch(int coord, int size)
{
    int res = (coord < 0) ? -coord : coord;
    res     = (res >= size) ? 2 * size - res - 2 : res;
    return res;
}

/**
 * @brief BORDER_REPLICATE : aaaaaa|abcdefgh|hhhhhhh
 * 
 * @param coord 输入坐标值，可能超出有效范围[0, size)
 * @param size 有效坐标范围的大小
 * @return 经过边界复制处理后的有效坐标值
 */
__device__ __forceinline__ int border_replicate(int coord, int size)
{
    if (coord < 0)
        return 0;
    if (coord >= size)
        return size - 1;
    return coord;
}