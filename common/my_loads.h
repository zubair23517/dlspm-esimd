#ifndef MY_LOADS_H
#define MY_LOADS_H

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <typename T, int S>
inline simd<T, S> my_lsc_block_load(T const *p)
{
    return lsc_block_load<T, S>(p);
}

template <>
inline simd<float, 128> my_lsc_block_load<float, 128>(const float *p)
{
    simd<float, 128> val;
    float const *const pnext = p + 64;
    val.select<64, 1>(0) = lsc_block_load<float, 64>(p);
    val.select<64, 1>(64) = lsc_block_load<float, 64>(pnext);

    return val;
}

template <>
inline simd<int, 128> my_lsc_block_load<int, 128>(const int *p)
{
    simd<int, 128> val;
    val.select<64, 1>(0) = lsc_block_load<int, 64>(p);
    val.select<64, 1>(64) = lsc_block_load<int, 64>(p + 64);

    return val;
}

template <>
inline simd<float, 256> my_lsc_block_load<float, 256>(const float *p)
{
    simd<float, 256> val;
    val.select<64, 1>(0) = lsc_block_load<float, 64>(p);
    val.select<64, 1>(64) = lsc_block_load<float, 64>(p + 64);
    val.select<64, 1>(128) = lsc_block_load<float, 64>(p + 128);
    val.select<64, 1>(192) = lsc_block_load<float, 64>(p + 192);

    return val;
}

template <>
inline simd<int, 256> my_lsc_block_load<int, 256>(const int *p)
{
    simd<int, 256> val;
    val.select<64, 1>(0) = lsc_block_load<int, 64>(p);
    val.select<64, 1>(64) = lsc_block_load<int, 64>(p + 64);
    val.select<64, 1>(128) = lsc_block_load<int, 64>(p + 128);
    val.select<64, 1>(192) = lsc_block_load<int, 64>(p + 192);

    return val;
}

template <typename T, int S>
inline simd<T, S> my_lsc_slm_block_load(uint32_t offset)
{
    return lsc_slm_block_load<T, S>(offset);
}

template <>
inline simd<float, 128> my_lsc_slm_block_load<float, 128>(uint32_t offset)
{
    simd<float, 128> val;
    val.select<64, 1>(0) = lsc_slm_block_load<float, 64>(offset);
    val.select<64, 1>(64) = lsc_slm_block_load<float, 64>(offset + 64 * sizeof(float));

    return val;
}

template <>
inline simd<int, 128> my_lsc_slm_block_load<int, 128>(uint32_t offset)
{
    simd<int, 128> val;
    val.select<64, 1>(0) = lsc_slm_block_load<int, 64>(offset);
    val.select<64, 1>(64) = lsc_slm_block_load<int, 64>(offset + 64 * sizeof(int));

    return val;
}

template <>
inline simd<float, 256> my_lsc_slm_block_load<float, 256>(uint32_t offset)
{
    simd<float, 256> val;
    val.select<64, 1>(0) = lsc_slm_block_load<float, 64>(offset);
    val.select<64, 1>(64) = lsc_slm_block_load<float, 64>(offset + 64 * sizeof(float));
    val.select<64, 1>(128) = lsc_slm_block_load<float, 64>(offset + 128 * sizeof(float));
    val.select<64, 1>(192) = lsc_slm_block_load<float, 64>(offset + 192 * sizeof(float));

    return val;
}

template <>
inline simd<int, 256> my_lsc_slm_block_load<int, 256>(uint32_t offset)
{
    simd<int, 256> val;
    val.select<64, 1>(0) = lsc_slm_block_load<int, 64>(offset);
    val.select<64, 1>(64) = lsc_slm_block_load<int, 64>(offset + 64 * sizeof(int));
    val.select<64, 1>(128) = lsc_slm_block_load<int, 64>(offset + 128 * sizeof(int));
    val.select<64, 1>(192) = lsc_slm_block_load<int, 64>(offset + 192 * sizeof(int));

    return val;
}

template <typename T, int S>
inline void my_lsc_slm_block_store(uint32_t offset, simd<T, S> vals)
{
    lsc_slm_block_store<T, S>(offset, vals);
}

template <>
inline void my_lsc_slm_block_store<float, 128>(uint32_t offset, simd<float, 128> vals)
{
    lsc_slm_block_store<float, 64>(offset, vals.select<64, 1>(0));
    lsc_slm_block_store<float, 64>(offset + 64 * sizeof(float), vals.select<64, 1>(64));
}

template <>
inline void my_lsc_slm_block_store<int, 128>(uint32_t offset, simd<int, 128> vals)
{
    lsc_slm_block_store<int, 64>(offset, vals.select<64, 1>(0));
    lsc_slm_block_store<int, 64>(offset + 64 * sizeof(int), vals.select<64, 1>(64));
}

template <>
inline void my_lsc_slm_block_store<float, 256>(uint32_t offset, simd<float, 256> vals)
{
    lsc_slm_block_store<float, 64>(offset, vals.select<64, 1>(0));
    lsc_slm_block_store<float, 64>(offset + 64 * sizeof(float), vals.select<64, 1>(64));
    lsc_slm_block_store<float, 64>(offset + 128 * sizeof(float), vals.select<64, 1>(128));
    lsc_slm_block_store<float, 64>(offset + 192 * sizeof(float), vals.select<64, 1>(192));
}

template <>
inline void my_lsc_slm_block_store<int, 256>(uint32_t offset, simd<int, 256> vals)
{
    lsc_slm_block_store<int, 64>(offset, vals.select<64, 1>(0));
    lsc_slm_block_store<int, 64>(offset + 64 * sizeof(int), vals.select<64, 1>(64));
    lsc_slm_block_store<int, 64>(offset + 128 * sizeof(int), vals.select<64, 1>(128));
    lsc_slm_block_store<int, 64>(offset + 192 * sizeof(int), vals.select<64, 1>(192));
}

template <typename T, int S>
inline void my_lsc_block_store(T *p, simd<T, S> vals)
{
    lsc_block_store(p, vals);
}

template <>
inline void my_lsc_block_store<float, 128>(float *p, simd<float, 128> vals)
{
    float *const pnext = p + 64;
    lsc_block_store<float, 64>(p, vals.select<64, 1>(0));
    lsc_block_store<float, 64>(pnext, vals.select<64, 1>(64));
}

template <typename T, int S>
inline simd<T, S> my_lsc_gather(T const *const p, simd<uint32_t, S> offsets, const int nnz)
{
    simd<T, S> ret;
    if (nnz == 0)
        return ret;
    simd_mask<S> mask = offsets < nnz * sizeof(T);
    ret = lsc_gather<T, 1>(p, offsets, mask);
    return ret;
}

template <>
inline simd<float, 64> my_lsc_gather<float, 64>(float const *const p, simd<uint32_t, 64> offsets, const int nnz)
{
    simd<float, 64> ret = 0;
    if (nnz == 0)
        return ret;
    simd_mask<32> mask = offsets.select<32, 1>(0) < nnz * sizeof(float);
    ret.select<32, 1>(0) = lsc_gather<float, 1>(p, offsets.select<32, 1>(0), mask);

    if (nnz < 32)
        return ret;

    mask = offsets.select<32, 1>(32) < nnz * sizeof(float);
    ret.select<32, 1>(32) = lsc_gather<float, 1>(p + 32, offsets.select<32, 1>(32), mask);

    return ret;
}

template <>
inline simd<int, 64> my_lsc_gather<int, 64>(int const *const p, simd<uint32_t, 64> offsets, const int nnz)
{
    simd<int, 64> ret = 0;
    if (nnz == 0)
        return ret;
    simd_mask<32> mask = offsets.select<32, 1>(0) < nnz * sizeof(int);
    ret.select<32, 1>(0) = lsc_gather<int, 1>(p, offsets.select<32, 1>(0), mask);

    if (nnz < 32)
        return ret;

    mask = offsets.select<32, 1>(32) < nnz * sizeof(int);
    ret.select<32, 1>(32) = lsc_gather<int, 1>(p + 32, offsets.select<32, 1>(32), mask);

    return ret;
}

template <typename T, int S>
inline void my_lsc_scatter(T *p, simd<uint32_t, S> offsets, simd<T, S> vals, int nnz)
{
    if (nnz == 0)
        return;
    simd_mask<S> mask = offsets < nnz * sizeof(T);
    lsc_scatter<T, 1, lsc_data_size::default_size, cache_hint::streaming, cache_hint::write_back>(p, offsets, vals, mask);
}

template <>
inline void my_lsc_scatter<float, 64>(float *p, simd<uint32_t, 64> offsets, simd<float, 64> vals, int nnz)
{
    if (nnz == 0)
        return;
    simd_mask<32> mask = offsets.select<32, 1>(0) < nnz * sizeof(float);
    lsc_scatter<float, 1, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>(
        p, offsets.select<32, 1>(0), vals.select<32, 1>(0), mask);

    if (nnz < 32)
        return;
    mask = offsets.select<32, 1>(32) < nnz * sizeof(float);
    lsc_scatter<float, 1, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>(
        p + 32, offsets.select<32, 1>(32), vals.select<32, 1>(32), mask);
}

template <>
inline void my_lsc_scatter<float, 128>(float *p, simd<uint32_t, 128> offsets, simd<float, 128> vals, int nnz)
{
    if (nnz == 0)
        return;
    if (nnz < 32)
    {
        simd_mask<32> mask = offsets.select<32, 1>(0) < nnz * sizeof(float);
        lsc_scatter<float, 1, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>(
            p, offsets.select<32, 1>(0), vals.select<32, 1>(0), mask);
        return;
    }
    else
        lsc_block_store<float, 32>(p, vals.select<32, 1>(0));

    if (nnz < 64)
    {
        simd_mask<32> mask = offsets.select<32, 1>(32) < nnz * sizeof(float);
        lsc_scatter<float, 1, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>(
            p + 32, offsets.select<32, 1>(32), vals.select<32, 1>(32), mask);
        return;
    }
    else
        lsc_block_store<float, 32>(p + 32, vals.select<32, 1>(32));

    if (nnz < 96)
    {
        simd_mask<32> mask = offsets.select<32, 1>(64) < nnz * sizeof(float);
        lsc_scatter<float, 1, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>(
            p + 64, offsets.select<32, 1>(64), vals.select<32, 1>(64), mask);
        return;
    }
    else
        lsc_block_store<float, 32>(p + 64, vals.select<32, 1>(64));

    if (nnz < 128)
    {
        simd_mask<32> mask = offsets.select<32, 1>(96) < nnz * sizeof(float);
        lsc_scatter<float, 1, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>(
            p + 96, offsets.select<32, 1>(96), vals.select<32, 1>(96), mask);
        return;
    }
    else
        lsc_block_store<float, 32>(p + 96, vals.select<32, 1>(96));
}

template <>
inline void my_lsc_scatter<float, 256>(float *p, simd<uint32_t, 256> offsets, simd<float, 256> vals, int nnz)
{
    for (int iter = 0; iter < 8; iter++)
    {
        if (nnz == iter * 32)
            return;
        simd_mask<32> mask = offsets.select<32, 1>(iter * 32) < nnz * sizeof(float);
        lsc_scatter<float, 1, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>(
            p + iter * 32, offsets.select<32, 1>(iter * 32), vals.select<32, 1>(iter * 32), mask);
    }
}

template <typename T, int S>
inline void my_lsc_l1_l3_prefetch(T const *const p)
{
    lsc_prefetch<T, S, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>(p);
}

template <>
inline void my_lsc_l1_l3_prefetch<float, 128>(float const *const p)
{
    float const *const pnext = p + 64;
    lsc_prefetch<float, 64, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>(p);
    lsc_prefetch<float, 64, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>(pnext);
}

template <>
inline void my_lsc_l1_l3_prefetch<float, 256>(float const *const p)
{
    lsc_prefetch<float, 64, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>(p);
    lsc_prefetch<float, 64, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>(p + 64);
    lsc_prefetch<float, 64, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>(p + 128);
    lsc_prefetch<float, 64, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>(p + 192);
}

template <typename T, int S>
inline void my_lsc_nl1_l3_prefetch(T *p)
{
    lsc_prefetch<T, S, lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached>(p);
}

template <>
inline void my_lsc_nl1_l3_prefetch<float, 128>(float *p)
{
    lsc_prefetch<float, 64, lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached>(p);
    lsc_prefetch<float, 64, lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached>(p + 64);
}

#endif
