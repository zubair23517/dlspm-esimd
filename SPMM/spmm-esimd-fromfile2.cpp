// esimd version of spmm
/*
 */

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <stdio.h>

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

// sanity check the input data
bool SanityCheck(const int MSZ, const int KSZ, const int NSZ, const int nnz,
                 const int *const a_row_offsets, const int *const a_column_indices)
{
    if (nnz > MSZ * KSZ)
        throw std::invalid_argument("nnz is larger than matrix.");
    if (a_row_offsets[MSZ] != nnz)
        throw std::invalid_argument("a_row_offsets[MSZ] != nnz.");
    for (int i = 0; i < MSZ; i++)
    {
        if (a_row_offsets[i + 1] < a_row_offsets[i])
            throw std::invalid_argument("a_row_offsets[i+1] < a_row_offsets[i]");

        if (a_row_offsets[i + 1] - a_row_offsets[i] > KSZ)
            throw std::invalid_argument("a_row_offsets[i + 1] - a_row_offsets[i] > KSZ");

        // std::cout << "Elems in Row = " << a_row_offsets[i + 1] - a_row_offsets[i] << std::endl;
    }

    for (int i = 0; i < nnz; i++)
    {
        if (a_column_indices[i] >= KSZ)
            throw std::invalid_argument("a_column_indices[i] >= KSZ");
    }

    return true;
}

void Spmm(const float *a_values,
          const int *a_row_offsets, const int *a_column_indices,
          const float *b, float *c, const int MSZ, const int NSZ)
{
    for (int i = 0; i < MSZ; ++i)
    {
        for (int j = 0; j < NSZ; ++j)
        {
            float accum = 0.0f;
            for (int l = a_row_offsets[i]; l < a_row_offsets[i + 1]; l++)
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

int main(int argc, char **argv)
{
    // check if code was launched correctly.
    try
    {
        if (argc != 3)
            throw std::invalid_argument("Usage: ./app <filename>.smtx <batchsize>");

        std::string filename(argv[1]);
        const int NSZ_rt = std::stoi(argv[2]);
        std::cout << "Filename = " << filename << ", batchsize (=NSZ) = " << NSZ_rt << std::endl;

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

        int MSZ_rt, KSZ_rt, nnz_rt;
        if (fscanf(fp, "%d,%d,%d", &MSZ_rt, &KSZ_rt, &nnz_rt) != 3)
            throw std::logic_error("Something went wrong readin M, K, nnz");

        printf("MSZ: %d, KSZ: %d, NSZ: %d, nnz: %d\n", MSZ_rt, KSZ_rt, NSZ_rt, nnz_rt);

        int *row_offsets = (int *)malloc((MSZ_rt + 1) * sizeof(int));
        int *column_indices = (int *)malloc(nnz_rt * sizeof(int));
        float *lhs_values = (float *)malloc(nnz_rt * sizeof(float));

        for (int i = 0; i < MSZ_rt + 1; i++)
        {
            if (fscanf(fp, "%d", &row_offsets[i]) != 1)
                throw std::logic_error("Something went wrong reading row_offsets");
        }

        for (int i = 0; i < nnz_rt; i++)
        {
            if (fscanf(fp, "%d", &column_indices[i]) != 1)
                throw std::logic_error("Something went wrong reading column_indices");

            lhs_values[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // fill lhs with random values
        }

        fclose(fp);

        SanityCheck(MSZ_rt, KSZ_rt, NSZ_rt, nnz_rt, row_offsets, column_indices);

        // allocate random dense rhs_matrix array of size KSZ X NSZ
        float *rhs_matrix = (float *)malloc(KSZ_rt * NSZ_rt * sizeof(float));
        std::cout << "rhs_matrix allocated" << std::endl;

        for (int i = 0; i < NSZ_rt * KSZ_rt; i++)
        {
            rhs_matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }

        // allocate output array of size MSZ X NSZ
        float *output_cpu = (float *)malloc(MSZ_rt * NSZ_rt * sizeof(float));
        float *output_gpu = (float *)malloc(MSZ_rt * NSZ_rt * sizeof(float));
        // initialize output array with zeros using memset function
        memset(output_cpu, 0, MSZ_rt * NSZ_rt * sizeof(float));
        memset(output_gpu, 0, MSZ_rt * NSZ_rt * sizeof(float));

        // call Spmm function
        Spmm(lhs_values, row_offsets, column_indices, rhs_matrix, output_cpu, MSZ_rt, NSZ_rt);

        double checksum_cpu = 0.0;
        // print output values
        for (int i = 0; i < MSZ_rt * NSZ_rt; i++)
        {
            checksum_cpu += output_cpu[i] * output_cpu[i];
        }

        std::cout.precision(7);
        std::cout << "checksum_cpu = " << checksum_cpu / ((double)MSZ_rt * NSZ_rt) << std::endl;

        // allocate device arrays and copy data from host to device

        int *row_offsets_dev = (int *)sycl::malloc_device((MSZ_rt + 1) * sizeof(int), dev_queue);
        dev_queue.memcpy(row_offsets_dev, row_offsets, (MSZ_rt + 1) * sizeof(int)).wait();
        int *column_indices_dev = (int *)sycl::malloc_device((nnz_rt + 2 * VLC) * sizeof(int), dev_queue); // extend this array to ensure that the prefetching is not doing an out of bounds access
        dev_queue.memcpy(column_indices_dev, column_indices, nnz_rt * sizeof(int)).wait();
        float *values_dev = (float *)sycl::malloc_device((nnz_rt + 2 * VLC) * sizeof(float), dev_queue);
        dev_queue.memcpy(values_dev, lhs_values, nnz_rt * sizeof(float)).wait();
        float *rhs_matrix_dev = (float *)sycl::malloc_device(KSZ_rt * NSZ_rt * sizeof(float), dev_queue);
        dev_queue.memcpy(rhs_matrix_dev, rhs_matrix, KSZ_rt * NSZ_rt * sizeof(float)).wait();
        float *output_dev = (float *)sycl::malloc_device(MSZ_rt * NSZ_rt * sizeof(float), dev_queue);

        std::cout << "copy to device completed" << std::endl;

        double tt;
        double exec_time = 12345678999.0;

        std::cout << "Start kernel execution" << std::endl;
        int chunksize;
        int NT;
        ComputeNT(MSZ_rt, NSZ_rt, chunksize, NT);
        const int nBlks = (NSZ_rt + (NT * chunksize) - 1) / (NT * chunksize);
        const int nh_threads = MSZ_rt * nBlks * NT;

        for (int loop = 0; loop < 20; loop++)
        {
            dev_queue.memset(output_dev, 0, MSZ_rt * NSZ_rt * sizeof(float)).wait();
            sycl::event e0;
            if (chunksize == 128)
            {
                e0 = dev_queue.parallel_for(
                    sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item, sycl::kernel_handler kh) SYCL_ESIMD_KERNEL
                    { SpmmEsimd<128>(
                          values_dev, row_offsets_dev, column_indices_dev,
                          rhs_matrix_dev, output_dev, item, MSZ_rt, KSZ_rt, NSZ_rt); });
            }
            else if (chunksize == 64)
            {
                e0 = dev_queue.parallel_for(
                    sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item, sycl::kernel_handler kh) SYCL_ESIMD_KERNEL
                    { SpmmEsimd<64>(
                          values_dev, row_offsets_dev, column_indices_dev,
                          rhs_matrix_dev, output_dev, item, MSZ_rt, KSZ_rt, NSZ_rt); });
            }
            else if (chunksize == 32)
            {
                e0 = dev_queue.parallel_for(
                    sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item, sycl::kernel_handler kh) SYCL_ESIMD_KERNEL
                    { SpmmEsimd<32>(
                          values_dev, row_offsets_dev, column_indices_dev,
                          rhs_matrix_dev, output_dev, item, MSZ_rt, KSZ_rt, NSZ_rt); });
            }
            else if (chunksize == 16)
            {
                e0 = dev_queue.parallel_for(
                    sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item, sycl::kernel_handler kh) SYCL_ESIMD_KERNEL
                    { SpmmEsimd<16>(
                          values_dev, row_offsets_dev, column_indices_dev,
                          rhs_matrix_dev, output_dev, item, MSZ_rt, KSZ_rt, NSZ_rt); });
            }
            else
                throw std::invalid_argument("NSZ is not a multiple of at least 16");

            // dev_queue.wait();
            tt = (e0.template get_profiling_info<info::event_profiling::command_end>() -
                  e0.template get_profiling_info<info::event_profiling::command_start>());

            dev_queue.wait();
            e0.wait();

            if (tt < exec_time)
                exec_time = tt;
        }

        const double SP_rt = (double)nnz_rt / (MSZ_rt * KSZ_rt);
        printf("MSZ = %d KSZ = %d NSZ = %d SP = %f NT = %d chunksize = %d \n", MSZ_rt, KSZ_rt, NSZ_rt, SP_rt, NT, chunksize);

        // exec time in ms
        exec_time = exec_time / 1000000.0;
        printf("Total kernel time was %f ms.\n", exec_time);
        // flop count: multiply a row i of sparse matrix  with a column of dense matrix KSZ x NSZ
        // (2 *nnz_i - 1)  * NSZ flops
        // total flops = (2 *nnz - MSZ) * NSZ
        float opcount = (2 * (float)nnz_rt - MSZ_rt) * NSZ_rt; // this is in flops
        float gflops = (opcount) / (exec_time * 1.0E+6);
        printf("GFLOPS:  %8.2f\n", gflops);

        // copy data from device to host
        dev_queue.memcpy(output_gpu, output_dev, MSZ_rt * NSZ_rt * sizeof(float)).wait();

        double checksum_gpu = 0.0;
        for (int i = 0; i < MSZ_rt * NSZ_rt; i++)
        {
            checksum_gpu += output_gpu[i] * output_gpu[i];
        }

        std::cout << "checksum_gpu = " << checksum_gpu / ((double)MSZ_rt * NSZ_rt) << std::endl;
        //  find the maximum difference between the two arrays
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        int maxi = 0;
        for (int i = 0; i < MSZ_rt * NSZ_rt; i++)
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
        std::cout << "output_gpu[" << maxi << "] = " << output_gpu[maxi] << std::endl
                  << std::endl;

#ifdef OUTPUT_CSV_FILE
        std::ofstream csv_file;
        csv_file.open(OUTPUT_CSV_FILE, std::ios::app | std::ios::out);
        csv_file << MSZ_rt << "," << NSZ_rt << "," << KSZ_rt << "," << SP_rt << "," << gflops << "," << max_diff << "," << opcount << std::endl;
        csv_file.close();
#endif

        // free host and device memory
        free(row_offsets);
        free(column_indices);
        free(lhs_values);
        free(rhs_matrix);
        free(output_cpu);
        free(output_gpu);
        sycl::free(row_offsets_dev, dev_queue);
        sycl::free(column_indices_dev, dev_queue);
        sycl::free(values_dev, dev_queue);
        sycl::free(rhs_matrix_dev, dev_queue);
        sycl::free(output_dev, dev_queue);
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Caught some exception" << std::endl;
    }

    return 0;
}
