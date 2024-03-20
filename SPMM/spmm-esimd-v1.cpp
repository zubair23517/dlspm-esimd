// esimd version of spmm
/*
 */

#include <iostream>
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

#ifndef NT
#define NT 4
#endif

// multiply sparse matrix of size MSZ X KSZ with a dense matrix of size KSZ X NSZ
// to produce a dense matrix of size MSZ X NSZ

#ifdef OUTPUT_CSV_FILE
#include <fstream>
#endif

#ifndef MSZ
#define MSZ (1 * 1024)
#endif

#define NBLKS (MSZ / NT)

#ifndef NSZ
#define NSZ (32)
#endif

#ifndef KSZ
#define KSZ (1 * 1024)
#endif

#ifndef SP
#define SP 0.3
#endif

#define RP 32
// #define RP2 16 // RP/2

// C++ Code,
// Source for sequential:
// https://github.com/google-research/sputnik/blob/master/sputnik/sddmm/sddmm_test.cu.cc#L62

void Spmm(const float *a_values,
          const int *a_row_offsets, const int *a_column_indices,
          const float *b, float *c)
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

/*

*/

void SpmmEsimdCPU(const float *a_values,
                  const int *a_row_offsets, const int *a_column_indices,
                  const float *b, float *c)
{

    for (int i = 0; i < NBLKS; i++)
    {
        for (int j = 0; j < NT; j++)
        {
            // number of non zeroes in the ith row of lhs matrix
            int start = a_row_offsets[i * NT + j];
            int end = a_row_offsets[i * NT + j + 1];
            int nnzr = end - start;
            int nchunks = (nnzr + RP - 1) / RP;
            int col_ida[RP];
            float a_row[RP];
            float c_row[NSZ] = {0.0};
            for (int l = 0; l < nchunks - 1; l++)
            {
                // load  RP column indices and non-zero values in  temporary arrays
                int idxb = start + l * RP;
                for (int l0 = 0; l0 < RP; l0++)
                {
                    int idx = idxb + l0;
                    col_ida[l0] = a_column_indices[idx];
                    a_row[l0] = a_values[idx];
                }
                float b_row[NSZ];
                // loop over RP
                for (int j0 = 0; j0 < RP; j0++)
                {
                    float av = a_row[j0];
                    int colid = col_ida[j0];

                    // load b_row of size  NSZ identified by colid
                    for (int l0 = 0; l0 < NSZ; l0++)
                    {
                        b_row[l0] = b[colid * NSZ + l0];
                    }
                    // perform vector operations to compute c_row
                    for (int l0 = 0; l0 < NSZ; l0++)
                    {
                        c_row[l0] = c_row[l0] + av * b_row[l0];
                    }
                }
            }

            // last chunk size
            int lcsz = nnzr - (nchunks - 1) * RP;
            // load  RP column indices and non-zero values in  vector registers
            int idxb = start + (nchunks - 1) * RP;
            for (int l0 = 0; l0 < lcsz; l0++)
            {
                int idx = idxb + l0;
                col_ida[l0] = a_column_indices[idx];
                a_row[l0] = a_values[idx];
            }
            float b_row[NSZ];
            // loop over RP
            for (int j0 = 0; j0 < lcsz; j0++)
            {
                float av = a_row[j0];
                int colid = col_ida[j0];

                // load b_row of size  NSZ identified by colid
                for (int l0 = 0; l0 < NSZ; l0++)
                {
                    b_row[l0] = b[colid * NSZ + l0];
                }
                // perform vector operations to compute c_row
                for (int l0 = 0; l0 < NSZ; l0++)
                {
                    c_row[l0] = c_row[l0] + av * b_row[l0];
                }
            }

            // store in c
            for (int j0 = 0; j0 < NSZ; j0++)
            {

                c[(i * NT + j) * NSZ + j0] = c_row[j0];
                // if (i==0) printf("c at %d : %f\n", j0, c_row[j0]);
            }

        } // end of thread loop over j
    }     // end of msz loop over i
}

template <int N>
void printfv(simd<float, N> v)
{
    // create a loop over the simd object and print each element
    for (int i = 0; i < N; i++)
    {
        float vs = v[i];
        sycl::ext::oneapi::experimental::printf("%f   ", vs);
    }
    sycl::ext::oneapi::experimental::printf("\n");
}

template <int N>
void printiv(simd<int, N> v)
{
    // create a loop over the simd object and print each element
    for (int i = 0; i < N; i++)
    {
        int vs = v[i];
        sycl::ext::oneapi::experimental::printf("%d   ", vs);
    }
    sycl::ext::oneapi::experimental::printf("\n");
}

// ESIMD Code
void SpmmEsimd(const float *a_values,
               const int *a_row_offsets, const int *a_column_indices,
               const float *b, float *c, sycl::nd_item<1> item)
{

    const int i = item.get_group(0);    // work group id
    const int j = item.get_local_id(0); // thread id
    // int doff = i * NSZ;
    // lsc_prefetch<float, NSZ/2, DSZ, L1_NC, L3_C>(&b[doff]);
    // lsc_prefetch<float, NSZ/2, DSZ, L1_NC, L3_C>(&b[doff+(NSZ/2)]);
    // lsc_prefetch<float, NSZ/2, DSZ, L1_NC, L3_C>(&b[doff]);
    // lsc_prefetch<float, NSZ/2, DSZ, L1_NC, L3_C>(&b[doff+(NSZ/2)]);

    simd<int, 2> lrowptr2;
    int idxij = i * NT + j;
    int coffset = idxij * NSZ;
    lrowptr2.copy_from(a_row_offsets + idxij);
    int nnzr = lrowptr2[1] - lrowptr2[0];
    int nchunks = (nnzr + RP - 1) / RP;

    // sycl::ext::oneapi::experimental::printf("nchunks %d i = %d j = %d\n", nchunks, i, j);

    simd<int, RP> col_ida;
    simd<float, RP> a_row;
    simd<float, NSZ> c_row;

    simd<float, NSZ> b_row;
    c_row = 0.0;

    simd<uint32_t, RP> offsetb0;
    simd<uint32_t, RP> offsetb1;

    simd_mask<RP> maskp = 1;
    int idxb = lrowptr2[0] ;
    lsc_prefetch<int, 32, DSZ, L1_C, L3_C>(&a_column_indices[idxb]);
    lsc_prefetch<float, 32, DSZ, L1_C, L3_C>(&a_values[idxb]);
    
    for (int l = 0; l < nchunks - 1; l++)
    {
        col_ida = lsc_block_load<int,RP>(a_column_indices + idxb);
        col_ida = col_ida * NSZ;
        a_row = lsc_block_load<float,RP>(a_values + idxb);
        idxb = lrowptr2[0] + (l+1) * RP;
        lsc_prefetch<int, 32, DSZ, L1_C, L3_C>(&a_column_indices[idxb]);
        lsc_prefetch<float, 32, DSZ, L1_C, L3_C>(&a_values[idxb]);        
        #pragma unroll
        for (int j0 = 0; j0 < RP; j0++)
        {
            float av = a_row.select<1, 1>(j0);
            int colid = col_ida.select<1, 1>(j0);
            // load b_row of size  NSZ identified by colid
            b_row.copy_from(b + colid);
            c_row = c_row + av * b_row;
        }
    }
    #if 1
    // last chunk size
    int lcsz = nnzr - (nchunks - 1) * RP;
    // load  RP column indices and non-zero values in  vector registers
    idxb = lrowptr2[0] + (nchunks - 1) * RP;

    simd<unsigned, RP> sa(0, 1);
    simd_mask<RP> mask = sa < lcsz;
 
    simd<uint32_t, RP> offset(0, sizeof(int));
    col_ida = 0;
    col_ida   = lsc_gather(a_column_indices + idxb, offset, mask);
    col_ida = col_ida * NSZ;
    a_row = 0.0;
    a_row   = lsc_gather(a_values + idxb, offset, mask);
    #pragma unroll
    for (int j0 = 0; j0 < lcsz; j0++)
    {
        float av = a_row[j0];
        
        int colid = col_ida[j0];
        // load b_row of size  NSZ identified by colid
        b_row.copy_from(b + colid);
        c_row = c_row + av * b_row;
    }
    #endif
    // store in c
    c_row.copy_to(c + coffset);
}

int main(int argc, char **argv)
{

    //  auto cacheLineSize = device.get_info<cl::sycl::info::device::global_mem_cache_line_size>();
    sycl::queue dev_queue(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});

    std::cout << "Running on "
              << dev_queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << dev_queue.get_device().get_info<sycl::info::device::global_mem_cache_line_size>() << "\n";

    // create a random sparse matrix with size MSZ x KSZ and initialize it with random values using rand() function
    int mnnz = (1.01 * MSZ * KSZ * SP) + 10;
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

    int nnz = row_offsets[MSZ];
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
    Spmm(values, row_offsets, column_indices, rhs_matrix, output_cpu);

    double checksum_cpu = 0.0;
    // print output values
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        checksum_cpu += output_cpu[i] * output_cpu[i];
    }

    std::cout.precision(7);
    std::cout << "checksum_cpu = " << checksum_cpu / ((float)MSZ * NSZ) << std::endl;

    memset(output_cpu, 0, MSZ * NSZ * sizeof(float));
    SpmmEsimdCPU(values, row_offsets, column_indices, rhs_matrix, output_cpu);
    // end of line

    checksum_cpu = 0.0;
    // print output values
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        checksum_cpu += output_cpu[i] * output_cpu[i];
    }

    std::cout << "checksum_cpu = " << checksum_cpu / ((float)MSZ * NSZ) << std::endl;

    // allocate device arrays and copy data from host to device

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

    int nh_threads = NBLKS * NT;

    for (int loop = 0; loop < 20; ++loop)
    {
        dev_queue.memset(output_dev, 0, MSZ * NSZ * sizeof(float));
        event e0;
        e0 = dev_queue.parallel_for(
            sycl::nd_range<1>(nh_threads, NT), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
            { SpmmEsimd(values_dev, row_offsets_dev, column_indices_dev,
                        rhs_matrix_dev, output_dev, item); });

        dev_queue.wait();
        tt = (e0.template get_profiling_info<info::event_profiling::command_end>() -
              e0.template get_profiling_info<info::event_profiling::command_start>());

        if (tt < exec_time)
            exec_time = tt;
    }

    printf("MSZ = %d KSZ = %d NSZ = %d SP = %f\n", MSZ, KSZ, NSZ, SP);
    
    // exec time in ms
    exec_time = exec_time / 1000000.0;
    printf("Total kernel time was %f ms.\n", exec_time);
    // flop count: multiply a row i of sparse matrix  with a column of dense matrix KSZ x NSZ
    // (2 *nnz_i - 1)  * NSZ flops
    // total flops = (2 *nnz - MSZ) * NSZ
    float opcount = (2 * (float)nnz - MSZ) * NSZ;
    float gflops = (opcount) / (exec_time * 1.0E+6);
    printf("GFLOPS:  %8.2f\n", gflops);

    #ifdef OUTPUT_CSV_FILE
    std::ofstream csv_file;
    csv_file.open(OUTPUT_CSV_FILE, std::ios::app | std::ios::out);
    csv_file << MSZ << "," << NSZ << "," << KSZ << "," << SP << "," << gflops <<  std::endl;
    csv_file.close();
    #endif



    // copy data from device to host
    dev_queue.memcpy(output_gpu, output_dev, MSZ * NSZ * sizeof(float)).wait();

    double checksum_gpu = 0.0;
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        // check for nan
        // if (isnan(output_gpu[i]) && i < 32)
        // {
        //     // printf i, output_gpu[i], ooutput_cpu[i]
        //     std::cout << "i = " << i << " output_gpu[i] = " << output_gpu[i] << " output_cpu[i] = " << output_cpu[i] << std::endl;
        // }
        checksum_gpu += output_gpu[i] * output_gpu[i];
    }

    std::cout << "checksum_gpu = " << checksum_gpu / ((float)MSZ * NSZ) << std::endl;
    //  find the maximum difference between the two arrays
    double max_diff = 0.0;
    int maxi = 0;
    for (int i = 0; i < MSZ * NSZ; i++)
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
    free(values);
    free(rhs_matrix);
    free(output_cpu);
    free(output_gpu);
    sycl::free(row_offsets_dev, dev_queue);
    sycl::free(column_indices_dev, dev_queue);
    sycl::free(values_dev, dev_queue);
    sycl::free(rhs_matrix_dev, dev_queue);
    sycl::free(output_dev, dev_queue);

    return 0;
}
