#ifndef SDDMMESIMD_HPP
#define SDDMMESIMD_HPP

#include "my_loads.h"
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

#ifndef MSZ
#define MSZ (1*1024)
#endif
#ifndef KSZ
#define KSZ (1*1024)
#endif
#ifndef NSZ
#define NSZ 32
#endif
#ifndef SP
#define SP 0.1
#endif

//#define VLC 4
#define VL MIN(NSZ,256)
#define A_CHUNK MIN(1024,NSZ)
#define NCHUNKS_A_IN_K NSZ/A_CHUNK
#define NCHUNKS_VL_IN_K (NSZ + VL - 1)/VL
#define NCHUNKS_VL_IN_A A_CHUNK/VL


constexpr int32_t my_ceil(float num)
{
    return (static_cast<float>(static_cast<int32_t>(num)) == num)
        ? static_cast<int32_t>(num)
        : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

constexpr int ComputeNT()
{
    constexpr int n_hardware_threads_1t_pvc = 4096;
    double max_occupancy = 0;
    int NT = 1;
    //we find the best NT value in [1,2,4,8,16] to maximize occupancy. 
    //If multiple NT values achieve same occupancy, we take the smallest.
    for (int tmp_nt = 1; tmp_nt <= 16; tmp_nt *=2) 
    {
        double tmp_occupancy = static_cast<double>(MSZ*tmp_nt)/n_hardware_threads_1t_pvc;
        tmp_occupancy /= my_ceil(tmp_occupancy); //normalize occupancy between 0 and 100%;
        if (tmp_occupancy > max_occupancy + 0.01) //increase in occupancy by at least 1%
        {
            max_occupancy = tmp_occupancy;
            NT = tmp_nt;
        }
    }

    return NT;
}

constexpr int ComputeVLC(int NT)
{
    //SP*K is the expected value of nonzeros per row
    //how many nnzs per esimd work-item and then VLC is chosen as the next power of 2, capped to 64
    //This should maximize the amount of work per work-item and minimze the number of loads 
    //of the matrix A.
    return 16;//MIN(64, static_cast<int>(std::bit_ceil((static_cast<unsigned>(SP*KSZ) + NT -1) / NT))); //bit_ceil gives next power of 2, requires c++20
}

// /// @brief Selects a suitable number of rows per work item/ group to minimize the number
// /// of threads while keeping maximum occpunacy. 
// /// @return If NT > 1, we return 1. Otherwise we return the maximum integer such that number of threads is the smallest multiple of 4096
// constexpr int ComputeRowsPerItem(int NT)
// {
//     constexpr int number_of_hardware_threads_in_pvc = 4096;
//     if (NT > 1 || MSZ <= number_of_hardware_threads_in_pvc)
//         return 1;
//     else if (NT == 1 && MSZ > number_of_hardware_threads_in_pvc)
//         return MAX(1,MIN(16,static_cast<int>(std::bit_floor(static_cast<unsigned>(MSZ/number_of_hardware_threads_in_pvc)))));

//     return 1;
// }

constexpr int NT = ComputeNT();
constexpr int VLC = ComputeVLC(NT);

// ESIMD Code
// General version which uses variable NT and VLC. Works well on 
// all sizes except the largest cases.
void SddmmEsimd(const int * row_offsets,
                const int *column_indices, const float *lhs_matrix,
                const float *rhs_matrix, float *output_values, sycl::nd_item<2> item)
{
    // number of threads needed to storare a row of size NSZ is NSZ/VL
    const int i = item.get_global_id(0);    // work group id
    const int j = item.get_local_id(1); // thread id //in this case global id should be local id

    // load row_v register with the row of  lhs_matrix using vector load
    simd<int, 2> lrowptr2 = lsc_block_load<int,2>(row_offsets + i);

    // number of non-zero elements in a row
    const int nnzr = lrowptr2.select<1,1>(1) - lrowptr2.select<1,1>(0);
    // number of non-zero elements processed by thread j
    const int nnzt = (nnzr + NT - 1) / NT; 
    const int nnzm = MIN(nnzt, nnzr - j * nnzt); // left over part //thread 0 processes the first nnzt, thread 1 the next nnzt etc.
    if (nnzm <=0 ) return; //without this, the code gives wrong results. Check why

    simd<float, VLC> resultA; //Processing the contiguous elements nnzt in chunks of VLC
    // loop over nnzm non-zero elements in chunks of size VL elements except the last chunk
    int chunk_size = VLC;
    const int nchunks = (nnzm + chunk_size - 1) / chunk_size;
    simd<int, VLC> idx_na;
    for (int l = 0; l < nchunks - 1; l++)
    {
        const int idxb = lrowptr2.select<1,1>(0) + j * nnzt + l * VLC;
        idx_na = my_lsc_block_load<int,VLC>(column_indices + idxb);
        resultA = 0.0f;
        
        for (int aiter = 0; aiter < NCHUNKS_A_IN_K; aiter++) {
            simd<float, A_CHUNK> reg_left;
            for (int iterk = 0; iterk < NCHUNKS_VL_IN_A; iterk++) {
                reg_left.select<VL,1>(iterk*VL) = my_lsc_block_load<float, VL>(lhs_matrix + i * NSZ + aiter*A_CHUNK + iterk*VL); //can have 2k entries in registers. After that we would spill.
            }

            #pragma unroll
            for (int l0=0; l0 < VLC; l0++) //ATTENTION: TODO: Unrolling this may not be possible if chunk_size == 1. In case of VL = 32, we may want to unroll by 8 even.
            {
                #pragma unroll
                for (int iterk = 0; iterk < NCHUNKS_VL_IN_A; iterk++) {
                    simd<float, VL> reg_right = my_lsc_block_load<float, VL>(rhs_matrix + idx_na.select<1,1>(l0) * NSZ + aiter*A_CHUNK + iterk*VL);
                    // aggregate the results of the vector operations
                    resultA.select<1,1>(l0) += reduce<float, float, VL>(reg_left.select<VL,1>(iterk*VL) * reg_right, std::plus<>());
                }
            }
        }
        lsc_block_store<float, VLC>(output_values + idxb, resultA);
    }

    // last chunk
    chunk_size = nnzm - (nchunks - 1) * VLC;
    const int idxb = lrowptr2.select<1,1>(0) + j * nnzt + (nchunks - 1) * VLC;
    idx_na = my_lsc_block_load<int,VLC>(column_indices + idxb);
    
    resultA = 0.0f;
    for (int aiter = 0; aiter < NCHUNKS_A_IN_K; aiter++) 
    {
        simd<float, A_CHUNK> reg_left;
        for (int iterk = 0; iterk < NCHUNKS_VL_IN_A; iterk++) {
            reg_left.select<VL,1>(iterk*VL) = my_lsc_block_load<float, VL>(lhs_matrix + i * NSZ + aiter*A_CHUNK + iterk*VL); //can have 2k entries in registers. After that we would spill.
        }
        
        for (int l0 = 0; l0 < chunk_size; l0++)
        {
            for (int iterk = 0; iterk < NCHUNKS_VL_IN_A; iterk++) {
                simd<float, VL> reg_right = my_lsc_block_load<float, VL>(rhs_matrix + idx_na.select<1,1>(l0) * NSZ + aiter*A_CHUNK + iterk*VL);
                resultA.select<1, 1>(l0) += reduce<float, float, VL>(reg_left.select<VL,1>(iterk*VL) * reg_right, std::plus<>());
            }
        }
    }
    my_lsc_scatter<float,VLC>(output_values + idxb, simd<uint32_t, VLC>(0, sizeof(float)), resultA, chunk_size); //my scatter has a masking built in. The block loads does not
}

#endif
