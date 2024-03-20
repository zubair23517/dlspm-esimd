// esimd version of sddmm

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include "../common/my_loads.h"

#ifdef OUTPUT_CSV_FILE
#include <fstream>
#endif

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

#define L1_C cache_hint::cached
#define L3_C cache_hint::cached
#define L1_NC cache_hint::uncached
#define L3_NC cache_hint::uncached
#define DSZ lsc_data_size::default_size

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#ifndef MSZ
#define MSZ (8 * 1024)
#endif
#ifndef KSZ
#define KSZ (2 * 1024)
#endif
#ifndef NSZ
#define NSZ 32
#endif
#ifndef SP
#define SP 0.3
#endif

#define VLC 32

// #define NT 8

// C++ Code,
// Source for sequential:
// https://github.com/google-research/sputnik/blob/master/sputnik/sddmm/sddmm_test.cu.cc#L62

void Sddmm(const int *row_offsets,
           const int *column_indices, const float *lhs_matrix,
           const float *rhs_matrix, float *output_values)
{
    for (int i = 0; i < MSZ; ++i)
    {
        for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j)
        {
            int idx_n = column_indices[j];
            double accumulator = 0.0;
            for (int l = 0; l < NSZ; ++l)
            {
                accumulator += static_cast<double>(lhs_matrix[i * NSZ + l]) *
                               static_cast<double>(rhs_matrix[idx_n * NSZ + l]);
            }
            output_values[j] = static_cast<float>(accumulator);
        }
    }
}

// ESIMD Code
template <int NT>
void SddmmEsimd(const int *row_offsets,
                const int *column_indices, const float *lhs_matrix,
                const float *rhs_matrix, float *output_values, sycl::nd_item<1> item)
{

    // number of threads needed to storare a row of size K is K/VL
    // assume K is VL and NT >= K/VL
    const int i = item.get_group(0);    // work group id
    const int j = item.get_local_id(0); // thread id

    // load row_v register with the row of  lhs_matrix using vector load
    // Assuming VL = K
    simd<float, NSZ> reg_left = my_lsc_block_load<float, NSZ>(lhs_matrix + i * NSZ);

    simd<int, 2> lrowptr2 = lsc_block_load<int, 2>(row_offsets + i);

    // number of non-zero elements in a row
    const int nnzr = lrowptr2.select<1, 1>(1) - lrowptr2.select<1, 1>(0);
    // number of non-zero elements processed by thread j
    const int nnzt = (nnzr + NT - 1) / NT;
    const int nnzm = MIN(nnzt, nnzr - j * nnzt); // left over part

    simd<float, VLC> resultA;
    // loop over nnzm non-zero elements in chunks of size VL elements except the last chunk
    int chunk_size = VLC;
    const int nchunks = (nnzm + chunk_size - 1) / chunk_size;
    // simd<int, VLC + 1> idx_na;
    simd<int, VLC> idx_na;

    int idxb = lrowptr2[0] + j * nnzt;
    lsc_prefetch<int, 32, DSZ, L1_C, L3_C>(&column_indices[idxb]);

    for (int l = 0; l < nchunks - 1; l++)
    {

        idx_na = lsc_block_load<int, VLC>(column_indices + idxb);

#pragma unroll
        for (int l0 = 0; l0 < VLC; l0++)
        {
            // ATTENTION, I assume VL=K
            simd<float, NSZ> reg_right = my_lsc_block_load<float, NSZ>(rhs_matrix + idx_na.select<1, 1>(l0) * NSZ);
            resultA.select<1, 1>(l0) = reduce<float, float, NSZ>(reg_left * reg_right, std::plus<>());
        }
        lsc_block_store<float, VLC>(output_values + idxb, resultA);
        idxb = lrowptr2[0] + j * nnzt + (l + 1) * VLC;
        lsc_prefetch<int, 32, DSZ, L1_C, L3_C>(&column_indices[idxb]);
    }
    // last chunk
    chunk_size = nnzm - (nchunks - 1) * VLC;

    idxb = lrowptr2.select<1, 1>(0) + j * nnzt + (nchunks - 1) * VLC;
    idx_na = lsc_block_load<int, VLC>(column_indices + idxb);
#pragma unroll 4
    for (int l0 = 0; l0 < chunk_size; l0++)
    {
        // index of the non-zero element
        simd<float, NSZ> reg_right = my_lsc_block_load<float, NSZ>(rhs_matrix + idx_na.select<1, 1>(l0) * NSZ /*+ m0 * VL*/);
        resultA.select<1, 1>(l0) = reduce<float, float, NSZ>(reg_left * reg_right, std::plus<>());
    }

    // lsc_scatter(output_values + idxb, offsets, resultA, mask);
    my_lsc_scatter<float, VLC>(output_values + idxb, simd<uint32_t, VLC>(0, sizeof(float)), resultA, chunk_size);
}

int main(int argc, char **argv)
{

    //  auto cacheLineSize = device.get_info<cl::sycl::info::device::global_mem_cache_line_size>();
    sycl::queue dev_queue(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});

    std::cout << "Running on "
              << dev_queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << dev_queue.get_device().get_info<sycl::info::device::global_mem_cache_line_size>() << "\n";

    // create a random sparse matrix with size m*n and initialize it with random values using rand() function
    int mnnz = (1.001 * MSZ * KSZ * SP) + 10;

    int *row_offsets = (int *)malloc((MSZ + 1) * sizeof(int));
    int *column_indices = (int *)malloc(mnnz * sizeof(int));
    // float *values = (float *)malloc(mnnz * sizeof(float));
    // initialize row_offsets and column_indices for a uniform random SP pattern
    row_offsets[0] = 0;
    for (int i = 0; i < MSZ; i++)
    {
        row_offsets[i + 1] = row_offsets[i];
        for (int j = 0; j < KSZ; j++)
        {
            if ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < SP))
            {
                column_indices[row_offsets[i + 1]] = j;
                // values not necessary
                // values[row_offsets[i + 1]] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                row_offsets[i + 1]++;
            }
        }
    }

    int nnz = row_offsets[MSZ];
    printf("nnz = %d mnnz = %d\n", nnz, mnnz);

    // allocate lhs_matrix array of size m*k
    float *lhs_matrix = (float *)malloc(MSZ * NSZ * sizeof(float));
    // create a random lhs_matrix array with size m*k and initialize it with random values using rand() function
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        lhs_matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    // allocate rhs_matrix array of size n*k
    // we are interested in multiplication with rhs_matrix transpose,
    // so we will read columns of rhs_matrix stored as rows (continuous memory access)
    float *rhs_matrix = (float *)malloc(KSZ * NSZ * sizeof(float));
    // create a random rhs_matrix array with size n*k and initialize it with random values using rand() function
    for (int i = 0; i < KSZ * NSZ; i++)
    {
        rhs_matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    // allocate output array of size nnz
    float *output_cpu = (float *)malloc(nnz * sizeof(float));
    float *output_gpu = (float *)malloc(nnz * sizeof(float));
    // float *output_gpu = (float *)malloc(nnz * sizeof(float));
    //  initialize output array with zeros
    for (int i = 0; i < nnz; i++)
    {
        output_cpu[i] = 0.0;
        output_gpu[i] = 0.0;
    }
    // CPU Calls for Validation
    // call sddmm host function
    Sddmm(row_offsets, column_indices, lhs_matrix, rhs_matrix, output_cpu);
    double checksum_cpu = 0.0;
    // print output values and create checksum
    for (int i = 0; i < nnz; i++)
    {
        checksum_cpu += output_cpu[i] * output_cpu[i];
        //  std::cout << output_cpu[i] << " ";
    }

    // allocate device arrays and copy data from host to device
    int *row_offsets_dev = (int *)sycl::malloc_device((MSZ + 1) * sizeof(int), dev_queue);
    // copy data from host to device
    dev_queue.memcpy(row_offsets_dev, row_offsets, (MSZ + 1) * sizeof(int));
    int *column_indices_dev = (int *)sycl::malloc_device(nnz * sizeof(int), dev_queue);
    dev_queue.memcpy(column_indices_dev, column_indices, nnz * sizeof(int));
    float *lhs_matrix_dev = (float *)sycl::malloc_device(MSZ * NSZ * sizeof(float), dev_queue);
    dev_queue.memcpy(lhs_matrix_dev, lhs_matrix, MSZ * NSZ * sizeof(float));
    float *rhs_matrix_dev = (float *)sycl::malloc_device(KSZ * NSZ * sizeof(float), dev_queue);
    dev_queue.memcpy(rhs_matrix_dev, rhs_matrix, KSZ * NSZ * sizeof(float));
    float *output_dev = (float *)sycl::malloc_device(nnz * sizeof(float), dev_queue);

    // printf("copy to device completed\n");

    double tt;
    double exec_time = 12345678999999.0;

    // printf("Start kernel execution\n");

    constexpr int n_hardware_threads_1t_pvc = 4096;
    double max_occupancy = 0;
    int NT = 1;
    // we find the best NT value in [1,2,4,8,16] to maximize occupancy.
    // If multiple NT values achieve same occupancy, we take the smallest.
    for (int tmp_nt = 1; tmp_nt <= 16; tmp_nt *= 2)
    {
        double tmp_occupancy = static_cast<double>(MSZ * tmp_nt) / n_hardware_threads_1t_pvc;
        tmp_occupancy /= std::ceil(tmp_occupancy); // normalize occupancy between 0 and 100%;
        if (tmp_occupancy > max_occupancy + 0.01)  // increase in occupancy by at least 1%
        {
            max_occupancy = tmp_occupancy;
            NT = tmp_nt;
        }
    }
    // NT = 1;
    const int nh_threads = MSZ * NT;
    // std::cout << "#threads = " << nh_threads << ", group_size = " << NT << ", occupancy = " << max_occupancy * 100 << std::endl;

    for (int loop = 0; loop < 20; ++loop)
    {
        dev_queue.memset(output_dev, 0, nnz * sizeof(float));
        event e0;
        if (NT == 1)
            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                {
#pragma forceinline recursive
                    SddmmEsimd<1>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                        rhs_matrix_dev, output_dev, item); });
        else if (NT == 2)
            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                {
#pragma forceinline recursive
                    SddmmEsimd<2>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                        rhs_matrix_dev, output_dev, item); });
        else if (NT == 4)
            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                {
#pragma forceinline recursive
                    SddmmEsimd<4>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                        rhs_matrix_dev, output_dev, item); });
        else if (NT == 8)
            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                {
#pragma forceinline recursive
                    SddmmEsimd<8>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                        rhs_matrix_dev, output_dev, item); });
        else if (NT == 16)
            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                {
#pragma forceinline recursive
                    SddmmEsimd<16>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                        rhs_matrix_dev, output_dev, item); });
        else
            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
                {
#pragma forceinline recursive
                    SddmmEsimd<32>(row_offsets_dev, column_indices_dev, lhs_matrix_dev,
                        rhs_matrix_dev, output_dev, item); });

        dev_queue.wait();
        tt = (e0.template get_profiling_info<info::event_profiling::command_end>() -
              e0.template get_profiling_info<info::event_profiling::command_start>());

        if (tt < exec_time)
            exec_time = tt;
    }
    // print M, K, N values
    printf("MSZ = %d,  NSZ = %d  KSZ = %d  SP = %f  NT = %d\n", MSZ, NSZ, KSZ, SP, NT);

    // exec time in ms
    exec_time = exec_time / 1000000.0;
    printf("Total kernel time was %f ms.\n", exec_time);

    double gflops1 = (static_cast<double>(nnz) * (2.0 * NSZ - 1.0)) / (exec_time * 1.0E+6);

    printf("GFLOPS :  %8.2f\n", gflops1);

    double bytes_accessed = ((static_cast<double>(nh_threads) * NSZ + static_cast<double>(NSZ) * nnz + static_cast<double>(nnz)) * sizeof(float)); // every A row is only accessed once for each thread. Every Col of B is accesses once for each nnz. There are probably also some cache effects. Every value in C is written to once
    double calc_bw;

    calc_bw = (bytes_accessed / (exec_time * 0.001)) * 0.000000001;

#ifdef OUTPUT_CSV_FILE
    std::ofstream csv_file;
    csv_file.open(OUTPUT_CSV_FILE, std::ios::app | std::ios::out);
    csv_file << MSZ << "," << KSZ << "," << NSZ << "," << SP << "," << gflops1 << "," << calc_bw << std::endl;
    csv_file.close();
#endif

    // printf("Application  bandwidth, GB/s:  %6.2f\n", calc_bw);

    // copy data from device to host
    dev_queue.memcpy(output_gpu, output_dev, nnz * sizeof(float)).wait();

    double checksum_gpu = 0.0;
    // print output values
    for (int i = 0; i < nnz; i++)
    {
        checksum_gpu += output_gpu[i] * output_gpu[i];
    }
    // print end of line

    // print checksums
    std::cout.precision(7);
    std::cout << "checksum_cpu = " << checksum_cpu / nnz << std::endl;
    std::cout << "checksum_gpu = " << checksum_gpu / nnz << std::endl;
    //  find the maximum difference between the two arrays
    double max_diff = 0.0;
    int maxi = 0;
    for (int i = 0; i < nnz; i++)
    {
        double diff = fabs(output_cpu[i] - output_gpu[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
            maxi = i;
        }
    }
    // print maxi and the maximum difference
    std::cout << "maxi = " << maxi << " max_diff = " << max_diff << std::endl;
    // print the values of the two arrays at maxi with 12 digits of precision
    std::cout.precision(7);
    std::cout << "output_cpu[" << maxi << "] = " << output_cpu[maxi] << std::endl;
    std::cout << "output_gpu[" << maxi << "] = " << output_gpu[maxi] << std::endl;

    // free host and device memory
    free(row_offsets);
    free(column_indices);
    free(lhs_matrix);
    free(rhs_matrix);
    free(output_cpu);
    free(output_gpu);

    // sycl::free(row_offsets_dev, dev_queue);
    // sycl::free(column_indices_dev, dev_queue);
    // sycl::free(lhs_matrix_dev, dev_queue);
    // sycl::free(rhs_matrix_dev, dev_queue);
    // sycl::free(output_dev, dev_queue);

    return 0;
}
