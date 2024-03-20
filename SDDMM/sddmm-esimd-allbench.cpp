// esimd version of sddmm

#include <iostream>
#include <sycl/sycl.hpp>

#include "sddmm-esimd-fromfile.hpp"

#ifdef OUTPUT_CSV_FILE
#include <fstream>
#endif

using namespace sycl;

// C++ Code,
// Source for sequential:
// https://github.com/google-research/sputnik/blob/master/sputnik/sddmm/sddmm_test.cu.cc#L62

void Sddmm(const int *row_offsets,
           const int *column_indices, const float *lhs_matrix,
           const float *rhs_matrix, float *output_values, const int MSZ, const int NSZ)
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

static inline void RunBenchmarkCase(const int MSZ, const int KSZ, const int NSZ, const double SP, sycl::queue &dev_queue)
{
    std::cout << "NEW RUN" << std::endl;
    const int mnnz = (1.01 * MSZ * KSZ * SP) + 100;

    int *row_offsets = (int *)malloc((MSZ + 1) * sizeof(int));
    int *column_indices = (int *)malloc(mnnz * sizeof(int));
    std::cout << "sparsity arrays allocated" << std::endl;
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
    // create a random lhs_matrix array with size m*n and initialize it with random values using rand() function
    std::cout << "lhs_matrix allocated" << std::endl;
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        lhs_matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float *rhs_matrix = (float *)malloc(KSZ * NSZ * sizeof(float));
    // create a random rhs_matrix array with size n*k and initialize it with random values using rand() function
    for (int i = 0; i < KSZ * NSZ; i++)
    {
        rhs_matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    std::cout << "rhs_matrix allocated" << std::endl;

    // allocate output array of size nnz
    float *output_cpu = (float *)malloc(nnz * sizeof(float));
    float *output_gpu = (float *)malloc(nnz * sizeof(float));
    memset(output_cpu, 0.0f, nnz * sizeof(float));
    memset(output_gpu, 0.0f, nnz * sizeof(float));

    std::cout << "done generating data" << std::endl;

    // CPU Calls for Validation
    // call sddmm host function
    Sddmm(row_offsets, column_indices, lhs_matrix, rhs_matrix, output_cpu, MSZ, NSZ);
    double checksum_cpu = 0.0;
    // print output values and create checksum
    for (int i = 0; i < nnz; i++)
    {
        checksum_cpu += output_cpu[i] * output_cpu[i];
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
    dev_queue.wait();

    // printf("copy to device completed\n");

    double tt;
    double exec_time = 12345678999999.0;

    // printf("Start kernel execution\n");
    const int NT = ComputeNT(MSZ);

    const int nh_threads = MSZ * NT;

    for (int loop = 0; loop < 20; ++loop)
    {
        dev_queue.memset(output_dev, 0, nnz * sizeof(float));
#pragma forceinline recursive
        event e0 = Launcher(row_offsets_dev,
                            column_indices_dev, lhs_matrix_dev,
                            rhs_matrix_dev, output_dev,
                            NSZ, NT, nh_threads, dev_queue);

        tt = (e0.template get_profiling_info<info::event_profiling::command_end>() -
              e0.template get_profiling_info<info::event_profiling::command_start>());

        if (tt < exec_time)
            exec_time = tt;
    }
    // print M, K, N values
    printf("MSZ = %d,  NSZ = %d  KSZ = %d  SP = %f  NT = %d\n", MSZ, NSZ, KSZ, SP, NT);
#ifdef NSZ_kernel
    std::cout << "NSZ_kernel = " << NSZ_kernel << std::endl;
#endif

    // exec time in ms
    exec_time = exec_time / 1000000.0;
    printf("Total kernel time was %f ms.\n", exec_time);

    double gflops1 = (static_cast<double>(nnz) * (2.0 * NSZ - 1.0)) / (exec_time * 1.0E+6);

    printf("GFLOPS :  %8.2f\n", gflops1);

    double bytes_accessed = ((static_cast<double>(nh_threads) * NSZ + static_cast<double>(NSZ) * nnz + static_cast<double>(nnz)) * sizeof(float)); // every A row is only accessed once for each thread. Every Col of B is accesses once for each nnz. There are probably also some cache effects. Every value in C is written to once
    double calc_bw;

    calc_bw = (bytes_accessed / (exec_time * 0.001)) * 0.000000001;

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

#ifdef OUTPUT_CSV_FILE
    std::ofstream csv_file;
    csv_file.open(OUTPUT_CSV_FILE, std::ios::app | std::ios::out);
    csv_file << MSZ << "," << KSZ << "," << NSZ << "," << SP << "," << gflops1 << "," << calc_bw << ", " << max_diff << std::endl;
    csv_file.close();
#endif

    // free host and device memory
    free(row_offsets);
    free(column_indices);
    free(lhs_matrix);
    free(rhs_matrix);
    free(output_cpu);
    free(output_gpu);

    sycl::free(row_offsets_dev, dev_queue);
    sycl::free(column_indices_dev, dev_queue);
    sycl::free(lhs_matrix_dev, dev_queue);
    sycl::free(rhs_matrix_dev, dev_queue);
    sycl::free(output_dev, dev_queue);

    dev_queue.wait();

    std::cout << "DONE WITH RUN" << std::endl
              << std::endl;
}

int main(int argc, char **argv)
{

    //  auto cacheLineSize = device.get_info<cl::sycl::info::device::global_mem_cache_line_size>();
    sycl::queue dev_queue(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});

    std::cout << "Running on "
              << dev_queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << dev_queue.get_device().get_info<sycl::info::device::global_mem_cache_line_size>() << "\n";

    std::vector<int> MSZ_arr = {1024, 3072, 4096, 2048, 6144, 8192, 4096, 12288, 16384, 8192, 24576, 32768};
    std::vector<int> KSZ_arr = {1024, 1024, 1024, 2048, 2048, 2048, 4096, 4096, 4096, 8192, 8192, 8192};
    // std::vector<int> NSZ_arr = {32, 128};
    std::vector<double> SP_arr = {0.3, 0.2, 0.1};

    // for (auto NSZ : NSZ_arr)
    {
        for (int case_iter = 0; case_iter < MSZ_arr.size(); case_iter++)
        {
            for (auto SP : SP_arr)
            {
                RunBenchmarkCase(MSZ_arr[case_iter], KSZ_arr[case_iter], NSZ_kernel, SP, dev_queue);
            }
        }
    }

    return 0;
}
