#pragma once

#include "../common/my_loads.h"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

#define L1_C cache_hint::cached
#define L3_C cache_hint::cached
#define L1_NC cache_hint::uncached
#define L3_NC cache_hint::uncached
#define DSZ lsc_data_size::default_size

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define VLC 32

constexpr int32_t my_ceil(float num)
{
    return (static_cast<float>(static_cast<int32_t>(num)) == num)
               ? static_cast<int32_t>(num)
               : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

constexpr int ComputeNT(const int MSZ)
{
    constexpr int n_hardware_threads_1t_pvc = 4096;
    double max_occupancy = 0;
    int NT = 1;
    // we find the best NT value in [1,2,4,8,16] to maximize occupancy.
    // If multiple NT values achieve same occupancy, we take the smallest.
    for (int tmp_nt = 1; tmp_nt <= 16; tmp_nt *= 2)
    {
        double tmp_occupancy = static_cast<double>(MSZ * tmp_nt) / n_hardware_threads_1t_pvc;
        tmp_occupancy /= my_ceil(tmp_occupancy);  // normalize occupancy between 0 and 100%;
        if (tmp_occupancy > max_occupancy + 0.01) // increase in occupancy by at least 1%
        {
            max_occupancy = tmp_occupancy;
            NT = tmp_nt;
        }
    }

    return NT;
}

// ESIMD Code
// General version which uses variable NT and VLC. Works well on
// all sizes except the largest cases.

// ESIMD Code
// Note that NSZ is the inner dimension in this case and the MSZx KSZ are the results dimensions
// #ifndef NSZ_kernel
// #define NSZ_kernel 32
// #endif
template <int NT, int CHUNKSIZE_NSZ>
inline void SddmmEsimd(int const *const __restrict__ row_offsets,
                       int const *const __restrict__ column_indices, float const *const __restrict__ lhs_matrix,
                       float const *const __restrict__ rhs_matrix, float *const __restrict__ output_values, const sycl::nd_item<1> &item,
                       const int /*NSZ_kernel*/)
{
    // number of threads needed to storare a row of size K is K/VL
    // assume K is VL and NT >= K/VL
    const int i = item.get_group(0);    // work group id
    const int j = item.get_local_id(0); // thread id

    const simd<int, 2> lrowptr2 = lsc_block_load<int, 2>(row_offsets + i);
    simd<float, VLC> resultA;

    // number of non-zero elements in a row
    const int nnzr = lrowptr2[1] - lrowptr2[0];

    // const int nnz_chunks = (nnzr + VLC - 1) / VLC;
    // const int nnz_chunks_thread = (nnz_chunks + NT - 1) / NT;
    // const int nnzt = nnz_chunks_thread * VLC;
    // number of non-zero elements processed by thread j
    const int nnzt = (nnzr + NT - 1) / NT;

    const int nnzm = MAX(0, MIN(nnzt, nnzr - j * nnzt)); // left over part
    if (nnzm == 0)
        return;

    // loop over nnzm non-zero elements in chunks of size VL elements except the last chunk
    const int nchunks = (nnzm + VLC - 1) / VLC;

    int idxb = lrowptr2[0] + j * nnzt;

    for (int l = 0; l < nchunks - 1; l++)
    {
        const simd<int, VLC> idx_na = lsc_block_load<int, VLC>(column_indices + idxb);
        resultA = 0;

        for (int nsz_chunk_iter = 0; nsz_chunk_iter < NSZ_kernel; nsz_chunk_iter += CHUNKSIZE_NSZ)
        {
            // stream this to ensure we do not pollute the cache for B?
            const simd<float, CHUNKSIZE_NSZ> reg_left = my_lsc_block_load<float, CHUNKSIZE_NSZ>(lhs_matrix + i * NSZ_kernel + nsz_chunk_iter);

#pragma unroll
            for (int l0 = 0; l0 < VLC; l0++)
            {
                const simd<float, CHUNKSIZE_NSZ> reg_right = my_lsc_block_load<float, CHUNKSIZE_NSZ>(
                    rhs_matrix + idx_na[l0] * NSZ_kernel + nsz_chunk_iter);
                resultA.select<1, 1>(l0) += reduce<float, float, CHUNKSIZE_NSZ>(reg_left * reg_right, std::plus<>());
            }
        }
        lsc_block_store<float, VLC, DSZ, cache_hint::streaming, cache_hint::write_back>(output_values + idxb, resultA);
        idxb += VLC;
    }
    // last chunk
    const int chunk_size = nnzm - (nchunks - 1) * VLC;
    const simd<int, VLC> idx_na = lsc_block_load<int, VLC, DSZ>(column_indices + idxb);
    resultA = 0;

    for (int nsz_chunk_iter = 0; nsz_chunk_iter < NSZ_kernel; nsz_chunk_iter += CHUNKSIZE_NSZ)
    {
        const simd<float, CHUNKSIZE_NSZ> reg_left = my_lsc_block_load<float, CHUNKSIZE_NSZ>(lhs_matrix + i * NSZ_kernel + nsz_chunk_iter);
#pragma unroll 4
        for (int l0 = 0; l0 < chunk_size; l0++)
        {
            // index of the non-zero element
            const simd<float, CHUNKSIZE_NSZ> reg_right = my_lsc_block_load<float, CHUNKSIZE_NSZ>(
                rhs_matrix + idx_na[l0] * NSZ_kernel + nsz_chunk_iter);
            resultA.select<1, 1>(l0) += reduce<float, float, CHUNKSIZE_NSZ>(reg_left * reg_right, std::plus<>());
        }
    }

    my_lsc_scatter<float, VLC>(output_values + idxb, simd<uint32_t, VLC>(0, sizeof(float)), resultA, chunk_size);
}

template <int NT>
inline sycl::event LauncherNSZCHUNKS(int const *const __restrict__ row_offsets_dev,
                                     int const *const __restrict__ column_indices_dev, float const *const __restrict__ lhs_matrix_dev,
                                     float const *const __restrict__ rhs_matrix_dev, float *const __restrict__ output_dev,
                                     const int NSZ, const int nh_threads, sycl::queue dev_queue)
{
    // if (NSZ % 1024)
    //     return dev_queue.parallel_for(
    //         sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
    //         { SddmmEsimd<NT, 1024>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
    //                                rhs_matrix_dev, output_dev, item, NSZ); });
    // else if (NSZ % 512)
    //     return dev_queue.parallel_for(
    //         sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
    //         { SddmmEsimd<NT, 512>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
    //                               rhs_matrix_dev, output_dev, item,NSZ); });
    // else
    // if (NSZ % 256 == 0)
    //     return dev_queue.parallel_for(
    //         sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
    //         { SddmmEsimd<NT, 256>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
    //                               rhs_matrix_dev, output_dev, item, NSZ); });
    // else
    if (NSZ % 128 == 0)
        return dev_queue.parallel_for(
            sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
            { SddmmEsimd<NT, 128>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                                  rhs_matrix_dev, output_dev, item, NSZ); });
    else if (NSZ % 64 == 0)
        return dev_queue.parallel_for(
            sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
            { SddmmEsimd<NT, 64>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                                 rhs_matrix_dev, output_dev, item, NSZ); });
    else if (NSZ % 32 == 0)
        return dev_queue.parallel_for(
            sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
            { SddmmEsimd<NT, 32>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                                 rhs_matrix_dev, output_dev, item, NSZ); });
    else if (NSZ % 16 == 0)
        return dev_queue.parallel_for(
            sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
            { SddmmEsimd<NT, 16>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                                 rhs_matrix_dev, output_dev, item, NSZ); });
    else
        throw std::invalid_argument("NSZ needs to be at least a multiple of 16");
}

static inline sycl::event Launcher(int const *const __restrict__ row_offsets,
                                   int const *const __restrict__ column_indices, float const *const __restrict__ lhs_matrix,
                                   float const *const __restrict__ rhs_matrix, float *const __restrict__ output_values,
                                   const int NSZ, const int NT, const int nh_threads, sycl::queue dev_queue)
{
    if (NT == 1)
        return LauncherNSZCHUNKS<1>(row_offsets,
                                    column_indices, lhs_matrix,
                                    rhs_matrix, output_values,
                                    NSZ, nh_threads, dev_queue);
    else if (NT == 2)
        return LauncherNSZCHUNKS<2>(row_offsets,
                                    column_indices, lhs_matrix,
                                    rhs_matrix, output_values,
                                    NSZ, nh_threads, dev_queue);
    else if (NT == 4)
        return LauncherNSZCHUNKS<4>(row_offsets,
                                    column_indices, lhs_matrix,
                                    rhs_matrix, output_values,
                                    NSZ, nh_threads, dev_queue);
    else if (NT == 8)
        return LauncherNSZCHUNKS<8>(row_offsets,
                                    column_indices, lhs_matrix,
                                    rhs_matrix, output_values,
                                    NSZ, nh_threads, dev_queue);
    else if (NT == 16)
        return LauncherNSZCHUNKS<16>(row_offsets,
                                     column_indices, lhs_matrix,
                                     rhs_matrix, output_values,
                                     NSZ, nh_threads, dev_queue);
    else
        throw std::logic_error("NT should not be larger than 16");
}
