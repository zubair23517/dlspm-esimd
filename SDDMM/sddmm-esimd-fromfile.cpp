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

int main(int argc, char **argv)
{
    if (argc != 3)
        throw std::invalid_argument("Usage: ./app <filename>.smtx <batchsize>");

    std::string filename(argv[1]);
    const int NSZ = std::stoi(argv[2]);
    std::cout << "Filename = " << filename << ", batchsize (=NSZ) = " << NSZ << std::endl;

    // initialize sycl queue
    sycl::queue dev_queue(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});

    // check that we are running on the correct device.
    std::cout << "Running on "
              << dev_queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << dev_queue.get_device().get_info<sycl::info::device::global_mem_cache_line_size>() << "\n";

    // Load lhs matrix from file
    FILE *fp;
    fp = fopen(filename.c_str(), "r");
    if (fp == NULL)
    {
        throw std::logic_error("Error opening file!");
    }

    int MSZ, KSZ, nnz;
    if (fscanf(fp, "%d,%d,%d", &MSZ, &KSZ, &nnz) != 3)
        throw std::logic_error("Something went wrong readin M, K, nnz");

    printf("MSZ: %d, KSZ: %d, NSZ: %d, nnz: %d\n", MSZ, KSZ, NSZ, nnz);

    int *row_offsets = (int *)malloc((MSZ + 1) * sizeof(int));
    int *column_indices = (int *)malloc(nnz * sizeof(int));

    for (int i = 0; i < MSZ + 1; i++)
    {
        if (fscanf(fp, "%d", &row_offsets[i]) != 1)
            throw std::logic_error("Something went wrong reading row_offsets");
    }

    for (int i = 0; i < nnz; i++)
    {
        if (fscanf(fp, "%d", &column_indices[i]) != 1)
            throw std::logic_error("Something went wrong reading column_indices");
    }

    fclose(fp);

    float *lhs_matrix = (float *)malloc(MSZ * NSZ * sizeof(float));
    // create a random lhs_matrix array with size m*k and initialize it with random values using rand() function
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
    // allocate output array of size nnz
    float *output_cpu = (float *)malloc(nnz * sizeof(float));
    float *output_gpu = (float *)malloc(nnz * sizeof(float));
    memset(output_cpu, 0, nnz * sizeof(float));
    memset(output_gpu, 0, nnz * sizeof(float));

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
    int *column_indices_dev = (int *)sycl::malloc_device((nnz + 2 * VLC) * sizeof(int), dev_queue);
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

        dev_queue.wait();
        tt = (e0.template get_profiling_info<info::event_profiling::command_end>() -
              e0.template get_profiling_info<info::event_profiling::command_start>());

        if (tt < exec_time)
            exec_time = tt;
    }
    // print M, K, N values
    printf("MSZ = %d,  NSZ = %d  KSZ = %d  NT = %d\n", MSZ, NSZ, KSZ, NT);

    // exec time in ms
    exec_time = exec_time / 1000000.0;
    printf("Total kernel time was %f ms.\n", exec_time);

    const double flops = static_cast<double>(nnz) * (2.0 * NSZ - 1.0);
    const double gflops1 = flops / (exec_time * 1.0E+6);

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
    csv_file << MSZ << "," << KSZ << "," << NSZ << "," << (double)nnz / (MSZ * KSZ)
             << "," << gflops1 << "," << flops << "," << max_diff << std::endl;
    csv_file.close();
#endif

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
