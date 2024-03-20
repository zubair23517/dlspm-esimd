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

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

// restriction: NSZ needs to be a multiple of CHUNKSIZE_NSZ
template <int CHUNKSIZE_NSZ>
void SpmmEsimd(float const *const __restrict__ a_values,
               int const *const __restrict__ a_row_offsets, int const *const __restrict__ a_column_indices,
               float const *const __restrict__ d, float *const __restrict__ e, sycl::nd_item<1> item,
               const int MSZ, const int KSZ, const int NSZ)
{
    const int bid = item.get_group(0);  // work group id
    const int j = item.get_local_id(0); // thread id
    const int NT = item.get_local_range(0);

    const int total_chunks = NSZ / CHUNKSIZE_NSZ;
    // number of blocks (work groups) needed to process a row
    const int nBlks = (NSZ + (NT * CHUNKSIZE_NSZ) - 1) / (NT * CHUNKSIZE_NSZ);
    // NT=4 threads in each block except  in the last block
    const int lbt = total_chunks - (nBlks - 1) * NT;

    // check if it is last block to process a row
    // last block may have less than NT threads
    int lbid = bid % nBlks;
    int ltid = NT;
    if (lbid == nBlks - 1)
        ltid = lbt;

    int i = bid / nBlks; // row id
    int lb_offset = lbid * CHUNKSIZE_NSZ * NT + j * CHUNKSIZE_NSZ;

    if (j < ltid)
    {
        simd<int, 2> lrowptr2;

        lrowptr2.copy_from(a_row_offsets + i);
        int nnzr = lrowptr2[1] - lrowptr2[0];
        simd<float, CHUNKSIZE_NSZ> e_row = 0.0;
        simd<float, CHUNKSIZE_NSZ> d_row;
#pragma unroll
        for (int l = 0; l < nnzr; l++)
        {
            int idx = lrowptr2[0] + l;
            int colid = a_column_indices[idx] * NSZ;
            float av = a_values[idx];
            // load d_row of size  CHUNKSIZE_NSZ
            d_row.copy_from(d + colid + lb_offset);
            e_row = e_row + av * d_row;
        }
        int eoffset = i * NSZ + lb_offset;
        e_row.copy_to(e + eoffset);
    }
}

#endif