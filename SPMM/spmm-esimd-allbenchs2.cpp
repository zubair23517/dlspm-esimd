// esimd version of spmm
/*
 */

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "spmmesimd2.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

#define L1_C cache_hint::cached
#define L3_C cache_hint::cached
#define L1_NC cache_hint::uncached
#define L3_NC cache_hint::uncached
#define DSZ lsc_data_size::default_size

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// multiply sparse matrix of size MSZ X KSZ with a dense matrix of size KSZ X NSZ
// to produce a dense matrix of size MSZ X NSZ

#ifdef OUTPUT_CSV_FILE
#include <fstream>
#endif

// C++ Code,
// Source for sequential:
// https://github.com/google-research/sputnik/blob/master/sputnik/sddmm/sddmm_test.cu.cc#L62

void Spmm(const float *a_values,
          const int *a_row_offsets, const int *a_column_indices,
          const float *b, float *c, const int MSZ, const int NSZ)
{
    for (int i = 0; i < MSZ; ++i)
    {
        for (int j = 0; j < NSZ; ++j)
        {
            float accum = 0.0;
            for (int l = a_row_offsets[i]; l < a_row_offsets[i + 1]; ++l)
            {
                float a_val = a_values[l];
                // print a_val
                // std::cout << "a_val = " << a_val << std::endl;
                int a_col = a_column_indices[l];
                accum += a_val * b[a_col * NSZ + j];
            }
            c[i * NSZ + j] = accum;
        }
    }
}

void RunBenchmarkCase(const int MSZ, const int KSZ, const int NSZ, const double SP, sycl::queue &dev_queue)
{
    // create a random sparse matrix with size MSZ x KSZ and initialize it with random values using rand() function
    const int mnnz = (1.01 * MSZ * KSZ * SP) + 100;
    int *row_offsets = (int *)malloc((MSZ + 1) * sizeof(int));
    int *column_indices = (int *)malloc(mnnz * sizeof(int));
    float *values = (float *)malloc(mnnz * sizeof(float));
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
                values[row_offsets[i + 1]] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 0.01;
                row_offsets[i + 1]++;
            }
        }
    }

    const int nnz = row_offsets[MSZ];
    printf("nnz = %d mnnz = %d\n", nnz, mnnz);

    // allocate rhs_matrix array of size KSZ X NSZ

    float *rhs_matrix = (float *)malloc(KSZ * NSZ * sizeof(float));
    printf("rhs_matrix allocated\n");

    for (int i = 0; i < NSZ * KSZ; i++)
    {
        rhs_matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // allocate output array of size MSZ X NSZ
    float *output_cpu = (float *)malloc(MSZ * NSZ * sizeof(float));
    float *output_gpu = (float *)malloc(MSZ * NSZ * sizeof(float));
    // initialize output array with zeros using memset function
    memset(output_cpu, 0, MSZ * NSZ * sizeof(float));
    memset(output_gpu, 0, MSZ * NSZ * sizeof(float));

    // call Spmm function
    Spmm(values, row_offsets, column_indices, rhs_matrix, output_cpu, MSZ, NSZ);

    double checksum_cpu = 0.0;
    // print output values
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        checksum_cpu += output_cpu[i] * output_cpu[i];
    }

    std::cout.precision(7);
    std::cout << "checksum_cpu = " << checksum_cpu / ((float)MSZ * NSZ) << std::endl;

    int *row_offsets_dev = (int *)sycl::malloc_device((MSZ + 1) * sizeof(int), dev_queue);
    dev_queue.memcpy(row_offsets_dev, row_offsets, (MSZ + 1) * sizeof(int));
    int *column_indices_dev = (int *)sycl::malloc_device(nnz * sizeof(int), dev_queue);
    dev_queue.memcpy(column_indices_dev, column_indices, nnz * sizeof(int));
    float *values_dev = (float *)sycl::malloc_device(nnz * sizeof(float), dev_queue);
    dev_queue.memcpy(values_dev, values, nnz * sizeof(float));
    float *rhs_matrix_dev = (float *)sycl::malloc_device(KSZ * NSZ * sizeof(float), dev_queue);
    dev_queue.memcpy(rhs_matrix_dev, rhs_matrix, KSZ * NSZ * sizeof(float));
    float *output_dev = (float *)sycl::malloc_device(MSZ * NSZ * sizeof(float), dev_queue);

    printf("copy to device completed\n");

    double tt;
    double exec_time = 12345678999.0;

    printf("Start kernel execution\n");
    int chunksize;
    int NT;
    ComputeNT(MSZ, NSZ, chunksize, NT);

    const int nBlks = (NSZ + (NT * chunksize) - 1) / (NT * chunksize);
    const int nh_threads = MSZ * nBlks * NT;

    for (int loop = 0; loop < 20; ++loop)
    {
        dev_queue.memset(output_dev, 0, MSZ * NSZ * sizeof(float));
        sycl::event e0;
        if (chunksize == 128)
        {
            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item, sycl::kernel_handler kh) SYCL_ESIMD_KERNEL
                { SpmmEsimd<128>(
                      values_dev, row_offsets_dev, column_indices_dev,
                      rhs_matrix_dev, output_dev, item, MSZ, KSZ, NSZ); });
        }
        else if (chunksize == 64)
        {
            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item, sycl::kernel_handler kh) SYCL_ESIMD_KERNEL
                { SpmmEsimd<64>(
                      values_dev, row_offsets_dev, column_indices_dev,
                      rhs_matrix_dev, output_dev, item, MSZ, KSZ, NSZ); });
        }
        else if (chunksize == 32)
        {
            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item, sycl::kernel_handler kh) SYCL_ESIMD_KERNEL
                { SpmmEsimd<32>(
                      values_dev, row_offsets_dev, column_indices_dev,
                      rhs_matrix_dev, output_dev, item, MSZ, KSZ, NSZ); });
        }
        else if (chunksize == 16)
        {

            e0 = dev_queue.parallel_for(
                sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item, sycl::kernel_handler kh) SYCL_ESIMD_KERNEL
                { SpmmEsimd<16>(
                      values_dev, row_offsets_dev, column_indices_dev,
                      rhs_matrix_dev, output_dev, item, MSZ, KSZ, NSZ); });
        }
        else
            throw std::invalid_argument("NSZ has to be a multiple of 16");

        dev_queue.wait();
        tt = (e0.template get_profiling_info<info::event_profiling::command_end>() -
              e0.template get_profiling_info<info::event_profiling::command_start>());

        if (tt < exec_time)
            exec_time = tt;
    }

    printf("MSZ = %d KSZ = %d NSZ = %d SP = %f NT = %d chunksize = %d \n", MSZ, KSZ, NSZ, SP, NT, chunksize);

    // exec time in ms
    exec_time = exec_time / 1000000.0;
    printf("Total kernel time was %f ms.\n", exec_time);
    // flop count: multiply a row i of sparse matrix  with a column of dense matrix KSZ x NSZ
    // (2 *nnz_i - 1)  * NSZ flops
    // total flops = (2 *nnz - MSZ) * NSZ
    float opcount = (2 * (float)nnz - MSZ) * NSZ;
    float gflops = (opcount) / (exec_time * 1.0E+6);
    printf("GFLOPS:  %8.2f\n", gflops);

    // copy data from device to host
    dev_queue.memcpy(output_gpu, output_dev, MSZ * NSZ * sizeof(float)).wait();

    double checksum_gpu = 0.0;
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        checksum_gpu += output_gpu[i] * output_gpu[i];
    }

    std::cout << "checksum_gpu = " << checksum_gpu / ((float)MSZ * NSZ) << std::endl;
    //  find the maximum difference between the two arrays
    double max_diff = 0.0;
    double max_rel_diff = 0.0;
    int maxi = 0;
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        const double diff = fabs(output_cpu[i] - output_gpu[i]);
        max_rel_diff = std::max<double>(max_rel_diff, diff / std::max<double>(std::abs<double>(output_cpu[i]), std::abs<double>(output_gpu[i])));
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
    std::cout << "*********************************" << std::endl
              << std::endl;

#ifdef OUTPUT_CSV_FILE
    std::ofstream csv_file;
    csv_file.open(OUTPUT_CSV_FILE, std::ios::app | std::ios::out);
    csv_file << MSZ << "," << NSZ << "," << KSZ << "," << SP << "," << gflops << "," << max_rel_diff << std::endl;
    csv_file.close();
#endif

    // free host and device memory
    free(row_offsets);
    free(column_indices);
    free(values);
    free(rhs_matrix);
    free(output_cpu);
    free(output_gpu);
    sycl::free(row_offsets_dev, dev_queue);
    sycl::free(column_indices_dev, dev_queue);
    sycl::free(values_dev, dev_queue);
    sycl::free(rhs_matrix_dev, dev_queue);
    sycl::free(output_dev, dev_queue);
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
    std::vector<int> NSZ_arr = {32, 128};
    std::vector<double> SP_arr = {0.3, 0.2, 0.1};

    for (auto NSZ : NSZ_arr)
    {
        for (int case_iter = 0; case_iter < MSZ_arr.size(); case_iter++)
        {
            for (auto SP : SP_arr)
            {
                RunBenchmarkCase(MSZ_arr[case_iter], KSZ_arr[case_iter], NSZ, SP, dev_queue);
            }
        }
    }

    return 0;
}
