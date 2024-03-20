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

#ifndef NT
#define NT 4
#endif

#define RP 32

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

// ESIMD Code
// template <int MSZ, int KSZ, int NSZ>
// void SpmmEsimd(float const *const __restrict__ a_values,
//                int const *const __restrict__ a_row_offsets, int const *const __restrict__ a_column_indices,
//                float const *const __restrict__ b, float *const __restrict__ c, sycl::nd_item<1> item)

// restriction: NSZ needs to be a multiple of CHUNKSIZE_NSZ
template <int CHUNKSIZE_NSZ>
void SpmmEsimd(float const *const __restrict__ a_values,
               int const *const __restrict__ a_row_offsets, int const *const __restrict__ a_column_indices,
               float const *const __restrict__ b, float *const __restrict__ c, sycl::nd_item<1> item,
               const int MSZ, const int KSZ, const int NSZ)
{
    const int idxij = item.get_global_linear_id();

    if (idxij >= MSZ) // too many items, return
        return;

    const simd<int, 2> lrowptr2 = lsc_block_load<int, 2>(a_row_offsets + idxij);
    const int nnzr = lrowptr2[1] - lrowptr2[0]; // if this is 0
    const int nchunks = (nnzr + RP - 1) / RP;   // then this is 0
    if (nchunks == 0)
        return;

    // last chunk size
    const int lcsz = std::max<int>(0, nnzr - (nchunks - 1) * RP); // if nnzr == 0 and nchunks == 0, this might be completely wrong

    simd<int, RP> col_ida;
    simd<float, RP> a_row;
    simd<float, CHUNKSIZE_NSZ> c_row;
    simd<float, CHUNKSIZE_NSZ> b_row;

    for (int nsz_iter = 0; nsz_iter < NSZ; nsz_iter += CHUNKSIZE_NSZ) // Note that we assume these CHUNKSIZE_NSZ divides NSZ
    {
        int idxb = lrowptr2[0];
        lsc_prefetch<int, RP, DSZ, L1_C, L3_C>(&a_column_indices[idxb]);
        lsc_prefetch<float, RP, DSZ, L1_C, L3_C>(&a_values[idxb]);
        c_row = 0.0f;

        for (int l = 0; l < nchunks - 1; l++)
        {
            col_ida = lsc_block_load<int, RP, DSZ, L1_C, L3_C>(a_column_indices + idxb);
            a_row = lsc_block_load<float, RP, DSZ, L1_C, L3_C>(a_values + idxb);
            idxb += RP; // prepare next iteration
            lsc_prefetch<int, RP, DSZ, L1_C, L3_C>(&a_column_indices[idxb]);
            lsc_prefetch<float, RP, DSZ, L1_C, L3_C>(&a_values[idxb]);
            col_ida *= NSZ;      // times NSZ because we load rows
            col_ida += nsz_iter; // shift in the rows to the right by nsz_iter
#pragma unroll
            for (int j0 = 0; j0 < RP; j0++)
            {
                // load b_row of size  NSZ identified by colid
                const int colid = col_ida[j0];
                b_row = my_lsc_block_load<float, CHUNKSIZE_NSZ>(b + colid);
                const float av = a_row[j0];
                c_row += av * b_row;
            }
        }

        // load  RP column indices and non-zero values in  vector registers
        if (lcsz != 0)
        {
            simd<unsigned, RP> sa(0, 1);
            simd_mask<RP> mask = sa < lcsz;

            simd<uint32_t, RP> offset(0, sizeof(int));
            col_ida = 0;
            col_ida = lsc_gather(a_column_indices + idxb, offset, mask);
            a_row = 0.0f;
            a_row = lsc_gather(a_values + idxb, offset, mask);
            col_ida *= NSZ;
            col_ida += nsz_iter;

#pragma unroll
            for (int j0 = 0; j0 < lcsz; j0++)
            {
                const int colid = col_ida[j0];
                // load b_row of size  NSZ identified by colid
                b_row = my_lsc_block_load<float, CHUNKSIZE_NSZ>(b + colid);
                const float av = a_row[j0];
                c_row += av * b_row;
            }
        }

        // store in c
        const size_t coffset = static_cast<size_t>(idxij) * NSZ + nsz_iter;
        my_lsc_block_store(c + coffset, c_row);
    }
}

#endif