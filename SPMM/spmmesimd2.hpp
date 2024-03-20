#ifndef SPMMESIMD_HPP
#define SPMMESIMD_HPP

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "../common/my_loads.h"

#define L1_C cache_hint::cached
#define L3_C cache_hint::cached
#define L1_NC cache_hint::uncached
#define L3_NC cache_hint::uncached
#define DSZ lsc_data_size::default_size

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define VLC 16

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

constexpr int32_t my_ceil(float num)
{
    return (static_cast<float>(static_cast<int32_t>(num)) == num)
               ? static_cast<int32_t>(num)
               : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

constexpr void ComputeNT(const int MSZ, const int NSZ, int &chunksize, int &NT)
{
    constexpr int n_hardware_threads_1t_pvc = 4096;
    double max_occupancy = 0;
    chunksize = 16; // chunksize 16 and NT 1 always work
    NT = 1;
    // we find the best NT value in [1,2,4,8,16] to maximize occupancy.
    // If multiple NT values achieve same occupancy, we take the smallest.
    for (int tmp_chunksize = 128; tmp_chunksize >= 16; tmp_chunksize /= 2)
    {
        if (NSZ % tmp_chunksize != 0)
            continue;

        int maxNT = MIN(NSZ / tmp_chunksize, 8);

        for (int tmp_nt = 1; tmp_nt <= maxNT; tmp_nt *= 2)
        {
            double tmp_occupancy =
                static_cast<double>(MSZ * tmp_nt) / n_hardware_threads_1t_pvc;
            tmp_occupancy /=
                my_ceil(tmp_occupancy); // normalize occupancy between 0 and 100%;
            if (tmp_occupancy >
                max_occupancy + 0.01) // increase in occupancy by at least 1%
            {
                max_occupancy = tmp_occupancy;
                NT = tmp_nt;
                chunksize = tmp_chunksize;
            }
        }
    }
}

// restriction: NSZ needs to be a multiple of CHUNKSIZE_NSZ
template <int CHUNKSIZE_NSZ>
void SpmmEsimd(float const *const __restrict__ a_values,
               int const *const __restrict__ a_row_offsets, int const *const __restrict__ a_column_indices,
               float const *const __restrict__ d, float *const __restrict__ e, sycl::nd_item<1> item,
               const int MSZ, const int KSZ, const int NSZ)
{
    const int bid = item.get_group(0);  // work group id
    const int j = item.get_local_id(0); // thread id
                                        //  check if it is last block to process a row

    const int NT = item.get_local_range(0);

    const int total_chunks = NSZ / CHUNKSIZE_NSZ;
    const int nBlks = (NSZ + (NT * CHUNKSIZE_NSZ) - 1) / (NT * CHUNKSIZE_NSZ);
    const int lbt = total_chunks - (nBlks - 1) * NT;

    const int lbid = bid % nBlks;
    const int ltid = lbid == nBlks - 1 ? lbt : NT;

    if (j >= ltid)
        return;

    const int i = bid / nBlks; // row id
    int lb_offset = lbid * CHUNKSIZE_NSZ * NT + j * CHUNKSIZE_NSZ;

    const simd<int, 2> lrowptr2 = lsc_block_load<int, 2>(a_row_offsets + i);
    const int nnzr = lrowptr2[1] - lrowptr2[0];
    if (nnzr == 0)
        return;
    const int nchunks = nnzr % VLC == 0 ? nnzr / VLC + 1 : (nnzr + VLC - 1) / VLC;

    // sycl::ext::oneapi::experimental::printf("nchunks %d i = %d j = %d\n", nchunks, i, j);

    simd<int, VLC> col_ida;
    simd<float, VLC> a_row;
    simd<float, CHUNKSIZE_NSZ> e_row = 0.0f;

    simd<float, CHUNKSIZE_NSZ> d_row;

    int idxb = lrowptr2[0];
    lsc_prefetch<int, VLC, DSZ, L1_C, L3_C>(a_column_indices + idxb);
    lsc_prefetch<float, VLC, DSZ, L1_C, L3_C>(a_values + idxb);

    for (int l = 0; l < nchunks - 1; l++)
    {
        col_ida = lsc_block_load<int, VLC>(a_column_indices + idxb) * NSZ;
        a_row = lsc_block_load<float, VLC>(a_values + idxb);
        idxb += VLC;
        lsc_prefetch<int, VLC, DSZ, L1_C, L3_C>(a_column_indices + idxb);
        lsc_prefetch<float, VLC, DSZ, L1_C, L3_C>(a_values + idxb);
#pragma unroll
        for (int j0 = 0; j0 < VLC; j0++)
        {
            // load b_row of size  NSZ identified by colid
            d_row = my_lsc_block_load<float, CHUNKSIZE_NSZ>(d + col_ida[j0] + lb_offset);
            const float av = a_row[j0];
            // d_row.copy_from(d + colid);
            e_row += av * d_row;
        }
    }

    // last chunk size
    const int lcsz = nnzr - (nchunks - 1) * VLC;
    if (lcsz > 0)
    {
        // load  VLC column indices and non-zero values in  vector registers

        col_ida = 0;
        col_ida = my_lsc_block_load<int, VLC>(a_column_indices + idxb) * NSZ; // my_lsc_gather<int, VLC>(a_column_indices + idxb, simd<uint32_t, VLC>(0, sizeof(int)), lcsz);
        a_row = 0.0f;
        a_row = my_lsc_block_load<float, VLC>(a_values + idxb); // my_lsc_gather<float, VLC>(a_values + idxb, simd<uint32_t, VLC>(0, sizeof(float)), lcsz);
#pragma unroll 4
        for (int j0 = 0; j0 < lcsz; j0++)
        {
            // load b_row of size  NSZ identified by colid
            d_row = my_lsc_block_load<float, CHUNKSIZE_NSZ>(d + col_ida[j0] + lb_offset);
            const float av = a_row[j0];
            e_row += av * d_row;
        }
    }

    // store in c
    const int eoffset = i * NSZ + lb_offset;
    my_lsc_block_store(e + eoffset, e_row);
}

#endif