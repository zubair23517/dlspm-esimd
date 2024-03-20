// esimd version of sddmm
/*

 */

#include <iostream>
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>
#include <random>
#include <bit>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include "sddmm-esimd-v1.hpp"

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



int main(int argc, char **argv)
{

    //  auto cacheLineSize = device.get_info<cl::sycl::info::device::global_mem_cache_line_size>();
    sycl::queue dev_queue(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});

    std::cout << "Running on "
              << dev_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << dev_queue.get_device().get_info<sycl::info::device::global_mem_cache_line_size>() << std::endl;

    // create a random sparse matrix with size m*n and initialize it with random values using rand() function
    const int nnz = MSZ * KSZ * SP;
    std::vector<int> row_offsets(MSZ+1, 0);
    std::vector<int> column_indices(nnz, 0);
    std::vector<int> all_indices(MSZ*KSZ,0);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> my_indices;
    std::sample(all_indices.begin(), all_indices.end(), std::back_inserter(my_indices), nnz, g);
    std::vector<int> rows(my_indices.size());
    std::transform(my_indices.begin(), my_indices.end(), rows.begin(), [](int x) { return x/KSZ;});
    std::transform(my_indices.begin(), my_indices.end(), column_indices.begin(), [](int x) { return x%KSZ;});

    for (int iter =0; iter < nnz; iter++) {
        row_offsets[rows[iter]]++;
    }
    for (int iiter = 0; iiter < MSZ; iiter++) {
        row_offsets[iiter + 1] += row_offsets[iiter];
    }
    
    std::cout << "nnz = " << nnz << std::endl;

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
    Sddmm(row_offsets.data(), column_indices.data(), lhs_matrix, rhs_matrix, output_cpu);
    double checksum_cpu = 0.0;
    // print output values and create checksum
    for (int i = 0; i < nnz; i++)
    {
        checksum_cpu += output_cpu[i] * output_cpu[i];
    }
    

    // allocate device arrays and copy data from host to device
    int *row_offsets_dev = (int *)sycl::malloc_device((MSZ + 1) * sizeof(int), dev_queue);
    // copy data from host to device
    dev_queue.memcpy(row_offsets_dev, row_offsets.data(), (MSZ + 1) * sizeof(int));
    int *column_indices_dev = (int *)sycl::malloc_device(nnz * sizeof(int), dev_queue);
    dev_queue.memcpy(column_indices_dev, column_indices.data(), nnz * sizeof(int));
    float *lhs_matrix_dev = (float *)sycl::malloc_device(MSZ * NSZ * sizeof(float), dev_queue);
    dev_queue.memcpy(lhs_matrix_dev, lhs_matrix, MSZ * NSZ * sizeof(float));
    float *rhs_matrix_dev = (float *)sycl::malloc_device(KSZ * NSZ * sizeof(float), dev_queue);
    dev_queue.memcpy(rhs_matrix_dev, rhs_matrix, KSZ * NSZ * sizeof(float));
    float *output_dev = (float *)sycl::malloc_device(nnz * sizeof(float), dev_queue);

    printf("copy to device completed\n");

    double tt;
    double exec_time = 12345678999999.0;

    printf("Start kernel execution\n");
    

    std::cout << "#threads = " << MSZ * NT << ", group_size = " << NT << std::endl;
    // print M, K, N values
    std::cout.precision(7);
    printf("MSZ = %d, NSZ = %d, KSZ = %d\n", MSZ, NSZ, KSZ);
    printf("SP = %f Sparsity fraction of zeros: 1-SP = %f\n", SP, 1.0-SP);
    printf("VL = %d, VLC = %d, NCHUNKS_VL_IN_K = %d, A_CHUNK = %d, NCHUNKS_A_IN_K = %d, NCHUNKS_VL_IN_A = %d, ROWS_PER_ITEM = %d\n", 
        VL, VLC, NCHUNKS_VL_IN_K, A_CHUNK, NCHUNKS_A_IN_K, NCHUNKS_VL_IN_A, 1);
    std::cout << std::endl;


    for (int loop = 0; loop < 20; ++loop)
    {
        dev_queue.memset(output_dev, 0, nnz * sizeof(float));
        event e0;
        e0 = dev_queue.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(MSZ, NT), sycl::range<2>(64/NT, NT)), [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL
        { 
            #pragma forceinline recursive
            SddmmEsimd(row_offsets_dev, column_indices_dev, lhs_matrix_dev, 
                rhs_matrix_dev, output_dev, item); 
        });
        

        dev_queue.wait();
        tt = (e0.template get_profiling_info<info::event_profiling::command_end>() -
              e0.template get_profiling_info<info::event_profiling::command_start>());

        if (tt < exec_time)
            exec_time = tt;
    }
   
   
    // exec time in ms
    exec_time = exec_time / 1000000.0;
    printf("Total kernel time was %f ms.\n", exec_time);
    double opcount = ((double) MSZ * NSZ * KSZ * SP) + ((double) MSZ * KSZ * (NSZ - 1) * SP);
    double gflops = (opcount) / (exec_time * 1.0E+6);
    double gflops1 = (static_cast<double>(nnz) * (2.0 * NSZ - 1.0)) / (exec_time * 1.0E+6);
    printf("GFLOPS:  %8.2f\n", gflops);
    printf("GFLOPS (more accurate based on the actual nnz):  %8.2f\n", gflops1);

    double bytes_accessed = ((static_cast<double>(MSZ * NT) * NSZ) + static_cast<double>(NSZ)*nnz + static_cast<double>(nnz)) * sizeof(float); //every A row is only accessed once for each thread. Every Col of B is accesses once for each nnz. There are probably also some cache effects. Every value in C is written to once
    double calc_bw;
 
    calc_bw = (bytes_accessed / (exec_time * 0.001)) * 0.000000001;

    printf("Application  bandwidth, GB/s:  %6.2f\n", calc_bw);

    #ifdef OUTPUT_CSV_FILE
    std::ofstream csv_file;
    csv_file.open(OUTPUT_CSV_FILE, std::ios::app | std::ios::out);
    csv_file << MSZ << "," << KSZ << "," << NSZ << "," << SP << "," << gflops1 << "," << calc_bw << std::endl;
    csv_file.close();
    #endif

    // copy data from device to host
    dev_queue.memcpy(output_gpu, output_dev, nnz * sizeof(float)).wait();

    double checksum_gpu = 0.0;
    // print output values
    for (int i = 0; i < nnz; i++)
    {
        checksum_gpu += output_gpu[i] * output_gpu[i];
        // print output_cpu, and output_gpu values in 7 digits of precision
        // printf("output_cpu[%d] = %f output_gpu[%d] = %f\n", i, output_cpu[i], i, output_gpu[i]);
    }
    // print end of line

    // print checksums
    
    std::cout << "checksum_cpu = " << checksum_cpu / nnz << std::endl;
    std::cout << "checksum_gpu = " << checksum_gpu / nnz << std::endl;
    //  find the maximum difference between the two arrays
    double max_diff = 0.0;
    double max_rel_diff = 0.0;
    int maxi = 0;
    int maxi_rel = 0;
    for (int i = 0; i < nnz; i++)
    {
        double diff = fabs(output_cpu[i] - output_gpu[i]);
        double rel_diff = diff == 0 ? 0.0 : diff/std::max(std::fabs(output_cpu[i]), output_gpu[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
            maxi = i;
        }
        if (rel_diff > max_rel_diff)
        {
            max_rel_diff = rel_diff;
            maxi_rel = i;
        }
    }
    // print maxi and the maximum difference
    std::cout << "maxi = " << maxi << " max_diff = " << max_diff << std::endl;
    std::cout << "maxi_rel = " << maxi_rel << " max_rel_diff = " << max_rel_diff << std::endl;
    // print the values of the two arrays at maxi with 12 digits of precision
    std::cout.precision(7);
    std::cout << "output_cpu[" << maxi << "] = " << output_cpu[maxi] << std::endl;
    std::cout << "output_gpu[" << maxi << "] = " << output_gpu[maxi] << std::endl;

    // free host and device memory
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
