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

namespace smallK
{

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
        for (int tmp_nt = 1; tmp_nt <= 8; tmp_nt *= 2)
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
            }
        }

        return NT;
    }

    typedef uint32_t Toffset;

    template <
        typename T, int N,
        sycl::ext::intel::experimental::esimd::lsc_data_size DS =
            sycl::ext::intel::experimental::esimd::lsc_data_size::default_size,
        sycl::ext::intel::experimental::esimd::cache_hint L1H =
            sycl::ext::intel::experimental::esimd::cache_hint::none,
        sycl::ext::intel::experimental::esimd::cache_hint L3H =
            sycl::ext::intel::experimental::esimd::cache_hint::none,
        typename Toffset>
    __ESIMD_NS::simd<T, N>
    my_lsc_atomic_add(T *p, __ESIMD_NS::simd<Toffset, N> offsets,
                      __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd_mask<N> pred)
    {

        constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
            __ESIMD_NS::native::lsc::atomic_op::fadd;

        constexpr uint16_t _AddressScale = 1;
        constexpr int _ImmOffset = 0;
        constexpr sycl::ext::intel::experimental::esimd::lsc_data_size _DS =
            sycl::ext::intel::experimental::esimd::detail::expand_data_size(
                sycl::ext::intel::experimental::esimd::detail::finalize_data_size<
                    T, DS>());
        constexpr sycl::ext::intel::experimental::esimd::detail::lsc_vector_size _VS =
            sycl::ext::intel::experimental::esimd::detail::to_lsc_vector_size<1>();
        constexpr sycl::ext::intel::experimental::esimd::detail::lsc_data_order
            _Transposed = sycl::ext::intel::experimental::esimd::detail::
                lsc_data_order::nontranspose;
        using _MsgT =
            typename sycl::ext::intel::experimental::esimd::detail::lsc_expand_type<
                T>::type;
        __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
        addrs += convert<uintptr_t>(offsets);
        __ESIMD_NS::simd<_MsgT, N> Tmp =
            __esimd_lsc_xatomic_stateless_1<_MsgT, _Op, L1H, L3H, _AddressScale,
                                            _ImmOffset, _DS, _VS, _Transposed, N>(
                pred.data(), addrs.data(),
                src0.template bit_cast_view<_MsgT>().data());
        return sycl::ext::intel::experimental::esimd::detail::lsc_format_ret<T>(Tmp);
    }

// ESIMD Code
// Note that NSZ is the inner dimension in this case and the MSZx KSZ are the
// results dimensions
#define MAXNNZ 32

    template <int NT, int NSZC>
    inline void SddmmEsimd(int const *const __restrict__ row_offsets,
                           int const *const __restrict__ column_indices,
                           float const *const __restrict__ lhs_matrix,
                           float const *const __restrict__ rhs_matrix,
                           float *const __restrict__ output_values,
                           const sycl::nd_item<1> &item, const int NSZ)
    {

        const int bid = item.get_group(0);  // work group id
        const int j = item.get_local_id(0); // thread id
        const int nBlks = (NSZ + (NT * NSZC) - 1) / (NT * NSZC);
        const int total_chunks = (NSZ + NSZC - 1) / NSZC;
        const int lbt = total_chunks - (nBlks - 1) * NT;
        const int nszcl = NSZ % NSZC;

        // check if it is last block to process a row
        // last block may have less than NT threads
        const int lbid = bid % nBlks;
        const int ltid = lbid == nBlks - 1 ? lbt : NT;

        const int i = bid / nBlks; // row id

        if (j >= ltid)
            return;

        const simd<int, 2> lrowptr2 = lsc_block_load<int, 2>(row_offsets + i);
        const int nnzr = lrowptr2[1] - lrowptr2[0];
        if (nnzr == 0)
            return;

        // assumption nnzr is less than 8
        simd<float, MAXNNZ> res = 0.0f;
        const int left_offset = i * NSZ + lbid * NT * NSZC + j * NSZC;
        const int rboffset = lbid * NT * NSZC + j * NSZC;

        // to handle last thread in last block
        // check first if NNZ is not a multiple of NNZC
        // maskc is set to 1 if it is a multiple of NSZC else
        // maskc is set based on the left over part

        const simd<float, NSZC> l_row = my_lsc_block_load<float, NSZC>(lhs_matrix + left_offset);
        simd_mask<NSZC> maskc = (nszcl > 0) && (lbid == nBlks - 1) && (j == ltid - 1) ? simd<unsigned, NSZC>(0, 1) < nszcl : 1;

        const simd<int, MAXNNZ> aoffset(0, sizeof(float));
        //
        const int nLoop = nnzr / MAXNNZ; /* round down! */
        const int nLeft = nnzr % MAXNNZ;
        int out_offset = lrowptr2[0];
        for (int ii = 0; ii < nLoop; ii++)
        {
            const simd<int, MAXNNZ> col_array = lsc_block_load<int, MAXNNZ>(column_indices + out_offset) * NSZ;
#pragma unroll
            for (int l = 0; l < MAXNNZ; l++)
            {
                const simd<float, NSZC> r_row = my_lsc_block_load<float, NSZC>(rhs_matrix + col_array[l] + rboffset);
                res.select<1, 1>(l) = reduce<float, float, NSZC>(merge(l_row * r_row, simd<float, NSZC>(0.0f), maskc), std::plus<>());
            }

            lsc_atomic_update<atomic_op::fadd, float, MAXNNZ>(output_values + out_offset, aoffset, res, simd_mask<MAXNNZ>(1));
            out_offset += MAXNNZ;
        }

        if (nLeft > 0)
        {
            const simd<int, MAXNNZ> col_array = lsc_block_load<int, MAXNNZ>(column_indices + out_offset) * NSZ;
#pragma unroll 4
            for (int l = 0; l < nLeft; l++)
            {
                const simd<float, NSZC> r_row = my_lsc_block_load<float, NSZC>(rhs_matrix + col_array[l] + rboffset);
                res.select<1, 1>(l) = reduce<float, float, NSZC>(merge(l_row * r_row, simd<float, NSZC>(0.0f), maskc), std::plus<>());
            }
            // my_lsc_atomic_add(output_values + out_offset, aoffset, res, simd<unsigned, MAXNNZ>(0, 1) < nLeft);
            lsc_atomic_update<atomic_op::fadd>(output_values + out_offset, aoffset, res, simd<unsigned, MAXNNZ>(0, 1) < nLeft);
        }
    }

    template <int NT>
    inline sycl::event
    LauncherNSZCHUNKS(int const *const __restrict__ row_offsets_dev,
                      int const *const __restrict__ column_indices_dev,
                      float const *const __restrict__ lhs_matrix_dev,
                      float const *const __restrict__ rhs_matrix_dev,
                      float *const __restrict__ output_dev, const int NSZ,
                      const int nh_threads, const int NSZC, sycl::queue dev_queue)
    {
        // if (NSZ % 1024)
        //     return dev_queue.parallel_for(
        //         sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item)
        //         SYCL_ESIMD_KERNEL { SddmmEsimd<NT, 1024>(row_offsets_dev,
        //         column_indices_dev, lhs_matrix_dev,
        //                                rhs_matrix_dev, output_dev, item, NSZ); });
        // else if (NSZ % 512)
        //     return dev_queue.parallel_for(
        //         sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item)
        //         SYCL_ESIMD_KERNEL { SddmmEsimd<NT, 512>(row_offsets_dev,
        //         column_indices_dev, lhs_matrix_dev,
        //                               rhs_matrix_dev, output_dev, item,NSZ); });
        // else
        // if (NSZ % 256 == 0)
        //     return dev_queue.parallel_for(
        //         sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item)
        //         SYCL_ESIMD_KERNEL { SddmmEsimd<NT, 256>(row_offsets_dev,
        //         column_indices_dev, lhs_matrix_dev,
        //                               rhs_matrix_dev, output_dev, item, NSZ); });
        // else
        if (NSZC == 128)
            return dev_queue.parallel_for(sycl::nd_range<1>(nh_threads, NT),
                                          [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                                          {
                                              SddmmEsimd<NT, 128>(
                                                  row_offsets_dev, column_indices_dev,
                                                  lhs_matrix_dev, rhs_matrix_dev,
                                                  output_dev, item, NSZ);
                                          });
        else if (NSZC == 64)
            return dev_queue.parallel_for(sycl::nd_range<1>(nh_threads, NT),
                                          [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                                          {
                                              SddmmEsimd<NT, 64>(
                                                  row_offsets_dev, column_indices_dev,
                                                  lhs_matrix_dev, rhs_matrix_dev,
                                                  output_dev, item, NSZ);
                                          });
        else if (NSZC == 32)
            return dev_queue.parallel_for(sycl::nd_range<1>(nh_threads, NT),
                                          [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                                          {
                                              SddmmEsimd<NT, 32>(
                                                  row_offsets_dev, column_indices_dev,
                                                  lhs_matrix_dev, rhs_matrix_dev,
                                                  output_dev, item, NSZ);
                                          });
        else if (NSZC == 16)
            return dev_queue.parallel_for(sycl::nd_range<1>(nh_threads, NT),
                                          [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                                          {
                                              SddmmEsimd<NT, 16>(
                                                  row_offsets_dev, column_indices_dev,
                                                  lhs_matrix_dev, rhs_matrix_dev,
                                                  output_dev, item, NSZ);
                                          });
        else
            return dev_queue.parallel_for(sycl::nd_range<1>(nh_threads, NT),
                                          [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                                          {
                                              SddmmEsimd<NT, 8>(
                                                  row_offsets_dev, column_indices_dev,
                                                  lhs_matrix_dev, rhs_matrix_dev,
                                                  output_dev, item, NSZ);
                                          });
    }

    static inline sycl::event Launcher(int const *const __restrict__ row_offsets,
                                       int const *const __restrict__ column_indices,
                                       float const *const __restrict__ lhs_matrix,
                                       float const *const __restrict__ rhs_matrix,
                                       float *const __restrict__ output_values,
                                       const int NSZ, const int NT,
                                       const int nh_threads, const int NSZC,
                                       sycl::queue dev_queue)
    {
        if (NT == 1)
            return LauncherNSZCHUNKS<1>(row_offsets, column_indices, lhs_matrix,
                                        rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                        dev_queue);
        else if (NT == 2)
            return LauncherNSZCHUNKS<2>(row_offsets, column_indices, lhs_matrix,
                                        rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                        dev_queue);
        else if (NT == 3)
            return LauncherNSZCHUNKS<3>(row_offsets, column_indices, lhs_matrix,
                                        rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                        dev_queue);
        else if (NT == 4)
            return LauncherNSZCHUNKS<4>(row_offsets, column_indices, lhs_matrix,
                                        rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                        dev_queue);
        else if (NT == 5)
            return LauncherNSZCHUNKS<5>(row_offsets, column_indices, lhs_matrix,
                                        rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                        dev_queue);
        else if (NT == 6)
            return LauncherNSZCHUNKS<6>(row_offsets, column_indices, lhs_matrix,
                                        rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                        dev_queue);
        else if (NT == 7)
            return LauncherNSZCHUNKS<7>(row_offsets, column_indices, lhs_matrix,
                                        rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                        dev_queue);
        else if (NT == 8)
            return LauncherNSZCHUNKS<8>(row_offsets, column_indices, lhs_matrix,
                                        rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                        dev_queue);
        else if (NT == 9)
            return LauncherNSZCHUNKS<9>(row_offsets, column_indices, lhs_matrix,
                                        rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                        dev_queue);
        else if (NT == 10)
            return LauncherNSZCHUNKS<10>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 11)
            return LauncherNSZCHUNKS<11>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 12)
            return LauncherNSZCHUNKS<12>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 13)
            return LauncherNSZCHUNKS<13>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 14)
            return LauncherNSZCHUNKS<14>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 15)
            return LauncherNSZCHUNKS<15>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 16)
            return LauncherNSZCHUNKS<16>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 17)
            return LauncherNSZCHUNKS<17>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 18)
            return LauncherNSZCHUNKS<18>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 19)
            return LauncherNSZCHUNKS<19>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 20)
            return LauncherNSZCHUNKS<20>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 21)
            return LauncherNSZCHUNKS<21>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 22)
            return LauncherNSZCHUNKS<22>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 23)
            return LauncherNSZCHUNKS<23>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 24)
            return LauncherNSZCHUNKS<24>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 25)
            return LauncherNSZCHUNKS<25>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 26)
            return LauncherNSZCHUNKS<26>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 27)
            return LauncherNSZCHUNKS<27>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 28)
            return LauncherNSZCHUNKS<28>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 29)
            return LauncherNSZCHUNKS<29>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 30)
            return LauncherNSZCHUNKS<30>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 31)
            return LauncherNSZCHUNKS<31>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else if (NT == 32)
            return LauncherNSZCHUNKS<32>(row_offsets, column_indices, lhs_matrix,
                                         rhs_matrix, output_values, NSZ, nh_threads, NSZC,
                                         dev_queue);
        else
            throw std::logic_error("NT should not be larger than 16");
    }

}
