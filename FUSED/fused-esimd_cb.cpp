// esimd version of sddmm

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include "../common/my_loads.h"

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

/*
A = C B^T
C is a dense matrix of size MSZ X NSZ
B is a dense matrix of size KSZ X NSZ
D is a dense matrix of size KSZ X NSZ

A is a sparse matrix of size MSZ X KSZ
E is a dense matrix of size MSZ X NSZ

E = A D

E = FUSEDMM(Sparsity(A), B, C, D)


*/

// C++ Code,
// Source for sequential:
// https://github.com/google-research/sputnik/blob/master/sputnik/sddmm/sddmm_test.cu.cc#L62

void Sddmm(const int *a_row_offsets,
           const int *a_column_indices, const float *c,
           const float *b, float *a_values)
{
    for (int i = 0; i < MSZ; ++i)
    {
        for (int j = a_row_offsets[i]; j < a_row_offsets[i + 1]; ++j)
        {
            int idx_n = a_column_indices[j];
            double accumulator = 0.0;
            for (int l = 0; l < NSZ; ++l)
            {
                accumulator += static_cast<double>(c[i * NSZ + l]) *
                               static_cast<double>(b[idx_n * NSZ + l]);
            }
            a_values[j] = static_cast<float>(accumulator);
        }
    }
}
// E = A D
void Spmm(const float *a_values,
          const int *a_row_offsets, const int *a_column_indices,
          const float *d, float *e)
{
    for (int i = 0; i < MSZ; ++i)
    {
        for (int j = 0; j < NSZ; ++j)
        {
            double accum = 0.0;
            for (int l = a_row_offsets[i]; l < a_row_offsets[i + 1]; ++l)
            {
                float a_val = a_values[l];
                // print a_val
                // std::cout << "a_val = " << a_val << std::endl;
                int a_col = a_column_indices[l];
                accum += static_cast<double>(a_val) *
                         static_cast<double>(d[a_col * NSZ + j]);
            }
            e[i * NSZ + j] = static_cast<float>(accum);
            // if (i==0) printf("accum at %d : %f\n", i, accum);
        }
    }
}

// ESIMD Code
void FusedEsimd(int const *const __restrict__ a_row_offsets,
                int const *const __restrict__ a_column_indices,
                float const *const __restrict__ c,
                float const *const __restrict__ b,
                float const *const __restrict__ d,
                float *const __restrict__ e, sycl::nd_item<1> item)
{
    const int i = item.get_global_linear_id(); // work group id
    if (i >= MSZ)
        return;

    //  load d into L2/L3 cache
    const int doff = i * NSZ;
    lsc_prefetch<float, NSZ / 2, DSZ, L1_NC, L3_C>(d + doff);
    lsc_prefetch<float, NSZ / 2, DSZ, L1_NC, L3_C>(d + doff + (NSZ / 2));
    lsc_prefetch<float, NSZ / 2, DSZ, L1_NC, L3_C>(b + doff);
    lsc_prefetch<float, NSZ / 2, DSZ, L1_NC, L3_C>(b + doff + (NSZ / 2));

    const simd<float, NSZ> reg_left = my_lsc_block_load<float, NSZ>(c + i * NSZ);
    const simd<int, 2> lrowptr2 = lsc_block_load<int, 2>(a_row_offsets + i);
    const int nnzr = lrowptr2[1] - lrowptr2[0];
    if (nnzr == 0)
        return;
    simd<float, VLC> a_row;
    simd<float, NSZ> e_row = 0.0f;

    const int nchunks = nnzr % VLC == 0 ? nnzr / VLC + 1 : (nnzr + VLC - 1) / VLC;
    int idxb = lrowptr2[0];
    lsc_prefetch<int, VLC, DSZ, L1_C, L3_C>(a_column_indices + idxb);

    for (int l = 0; l < nchunks - 1; l++)
    {
        const simd<int, VLC> col_ida = lsc_block_load<int, VLC>(a_column_indices + idxb) * NSZ;

#pragma unroll
        for (int l0 = 0; l0 < VLC; l0++)
        {
            const simd<float, NSZ> reg_right = my_lsc_block_load<float, NSZ>(b + col_ida[l0]);
            a_row.select<1, 1>(l0) = reduce<float, float, NSZ>(reg_left * reg_right, std::plus<>());
        }

#pragma unroll
        for (int j0 = 0; j0 < VLC; j0++)
        {
            // load b_row of size  NSZ identified by colid
            // const simd<float, NSZ> d_row = my_lsc_block_load<float, NSZ>(d + col_ida[j0]);
            simd<float, NSZ> d_row;
            d_row.copy_from(d + col_ida[j0]);
            const float av = a_row[j0];
            e_row += av * d_row;
        }

        idxb += VLC;
        lsc_prefetch<int, VLC, DSZ, L1_C, L3_C>(a_column_indices + idxb);
    }

    // last chunk
    const int lcsz = nnzr - (nchunks - 1) * VLC;

    if (lcsz > 0)
    {

        a_row = 0.0f;
        const simd<unsigned, VLC> sa(0, 1);
        const simd_mask<VLC> mask = sa < lcsz;
        const simd<int, VLC> col_ida = lsc_gather(a_column_indices + idxb, simd<uint32_t, VLC>(0, sizeof(int)), mask) * NSZ;

#pragma unroll 4
        for (int l0 = 0; l0 < lcsz; l0++)
        {
            // index of the non-zero element
            const simd<float, NSZ> reg_right = my_lsc_block_load<float, NSZ>(b + col_ida[l0]);
            a_row.select<1, 1>(l0) = reduce<float, float, NSZ>(reg_left * reg_right, std::plus<>());
        }
        // spmm code
#pragma unroll
        for (int j0 = 0; j0 < lcsz; j0++)
        {
            // load b_row of size  NSZ identified by colid
            // const simd<float, NSZ> d_row = my_lsc_block_load<float, NSZ>(d + col_ida[j0]);
            simd<float, NSZ> d_row;
            d_row.copy_from(d + col_ida[j0]);
            const float av = a_row[j0];
            e_row += av * d_row;
        }
    }

    // store in c
    my_lsc_block_store(e + i * NSZ, e_row);
}

double checksum(float *x, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += x[i] * x[i];
    }
    sum = sqrt(sum / n);
    return sum;
}

void validate(float *x_gold, float *x, int n)
{
    double max_diff = 0.0;
    int maxi = 0;
    for (int i = 0; i < n; i++)
    {
        double diff = fabs(x_gold[i] - x[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
            maxi = i;
        }
    }
    // print maxi and the maximum difference
    std::cout << "maxi = " << maxi << " max_diff = " << max_diff << std::endl;
    // print the values of the two arrays at maxi with 7 digits of precision
    std::cout.precision(7);
    std::cout << "gold output[" << maxi << "] = " << x_gold[maxi] << std::endl;
    std::cout << "computed[" << maxi << "] = " << x[maxi] << std::endl;
}

int main(int argc, char **argv)
{

    //  auto cacheLineSize = device.get_info<cl::sycl::info::device::global_mem_cache_line_size>();
    sycl::queue dev_queue(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});

    std::cout << "Running on "
              << dev_queue.get_device().get_info<sycl::info::device::name>() << "\n";

    // create a random sparse matrix with size m*n and initialize it with random values using rand() function
    int mnnz = (1.01 * MSZ * KSZ * SP) + 100;

    int *a_row_offsets = (int *)malloc((MSZ + 1) * sizeof(int));
    int *a_column_indices = (int *)malloc(mnnz * sizeof(int));
    // float *values = (float *)malloc(mnnz * sizeof(float));
    // initialize a_row_offsets and a_column_indices for a uniform random SP pattern
    a_row_offsets[0] = 0;
    for (int i = 0; i < MSZ; i++)
    {
        a_row_offsets[i + 1] = a_row_offsets[i];
        for (int j = 0; j < KSZ; j++)
        {
            if ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < SP))
            {
                if (a_row_offsets[i + 1] >= mnnz)
                {
                    printf("nnz exceeding mnnz %d mnnz = %d\n", a_row_offsets[i + 1], mnnz);
                    exit(0);
                }

                a_column_indices[a_row_offsets[i + 1]] = j;
                // values not necessary
                // a_values[a_row_offsets[i + 1]] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                a_row_offsets[i + 1]++;
            }
        }
    }

    int nnz = a_row_offsets[MSZ];
    printf("MSZ = %d KSZ = %d NSZ = %d SP = %f\n", MSZ, KSZ, NSZ, SP);
    printf("nnz = %d mnnz = %d\n", nnz, mnnz);

    float *c = (float *)malloc(MSZ * NSZ * sizeof(float));
    float *e_gold = (float *)malloc(MSZ * NSZ * sizeof(float));
    float *e = (float *)malloc(MSZ * NSZ * sizeof(float));

    // create a random c array with size m*k and initialize it with random values using rand() function
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        c[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 0.01;
    }
    // allocate rhs_matrix array of size n*k
    // we are interested in multiplication with rhs_matrix transpose,
    // so we will read columns of rhs_matrix stored as rows (continuous memory access)
    float *b = (float *)malloc(KSZ * NSZ * sizeof(float));
    float *d = (float *)malloc(KSZ * NSZ * sizeof(float));
    // create a random b array with size n*k and initialize it with random values using rand() function
    for (int i = 0; i < KSZ * NSZ; i++)
    {
        b[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 0.01;
        d[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 0.01;
    }
    //
    float *a_values = (float *)malloc(nnz * sizeof(float));
    //  initialize a_values with zeros using memset
    memset(a_values, 0, nnz * sizeof(float));
    Sddmm(a_row_offsets, a_column_indices, c, b, a_values);
    //  print checksum for a_values
    printf("checksum_a_values = %f\n", checksum(a_values, nnz));
    memset(e_gold, 0, MSZ * NSZ * sizeof(float));
    Spmm(a_values, a_row_offsets, a_column_indices, d, e_gold);
    printf("checksum_e = %f\n", checksum(e_gold, MSZ * NSZ));

    // allocate device arrays and copy data from host to device
    int *a_row_offsets_dev = (int *)sycl::malloc_device((MSZ + 1) * sizeof(int), dev_queue);
    // copy data from host to device
    dev_queue.memcpy(a_row_offsets_dev, a_row_offsets, (MSZ + 1) * sizeof(int));
    int *a_column_indices_dev = (int *)sycl::malloc_device((nnz + 4 * VLC) * sizeof(int), dev_queue);
    dev_queue.memcpy(a_column_indices_dev, a_column_indices, nnz * sizeof(int));
    float *c_dev = (float *)sycl::malloc_device(MSZ * NSZ * sizeof(float) + 1000, dev_queue);
    dev_queue.memcpy(c_dev, c, MSZ * NSZ * sizeof(float));
    float *b_dev = (float *)sycl::malloc_device(KSZ * NSZ * sizeof(float) + 1000, dev_queue);
    dev_queue.memcpy(b_dev, b, KSZ * NSZ * sizeof(float));

    float *d_dev = (float *)sycl::malloc_device(KSZ * NSZ * sizeof(float) + 1000, dev_queue);
    dev_queue.memcpy(d_dev, d, KSZ * NSZ * sizeof(float));

    float *e_dev = (float *)sycl::malloc_device(MSZ * NSZ * sizeof(float) + 1000, dev_queue);
    printf("copy to device completed\n");

    double tt;
    double exec_time = 12345678999999.0;

    printf("Start kernel execution\n");

    const int nh_threads = MSZ;
    // std::cout << "#threads = " << nh_threads << ", group_size = " << NT << ", occupancy = " << max_occupancy * 100 << std::endl;
    for (int loop = 0; loop < 10; ++loop)
    {

        dev_queue.memset(e_dev, 0, MSZ * NSZ * sizeof(float));
        event e0;

        e0 = dev_queue.parallel_for(
            sycl::nd_range<1>(nh_threads, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL
            {
#pragma forceinline recursive
            FusedEsimd(a_row_offsets_dev, a_column_indices_dev, c_dev,
                b_dev, d_dev, e_dev, item); });

        tt = (e0.template get_profiling_info<info::event_profiling::command_end>() -
              e0.template get_profiling_info<info::event_profiling::command_start>());

        if (tt < exec_time)
            exec_time = tt;
    }
    // print M, K, N values
    printf("MSZ = %d,  NSZ = %d  KSZ = %d  SP = %f\n", MSZ, NSZ, KSZ, SP);

    // exec time in ms
    exec_time = exec_time / 1000000.0;
    printf("Total kernel time was %f ms.\n", exec_time);

    float opcount1 = (2 * (float)nnz - MSZ) * NSZ;
    float opcount2 = (static_cast<double>(nnz) * (2.0 * NSZ - 1.0));
    float opcount = opcount1 + opcount2;

    double gflops1 = opcount / (exec_time * 1.0E+6);

    printf("GFLOPS/s :  %8.2f\n", gflops1);

    // double bytes_accessed = ((static_cast<double>(nh_threads) * NSZ + static_cast<double>(NSZ) * nnz + static_cast<double>(nnz)) * sizeof(float)); // every A row is only accessed once for each thread. Every Col of B is accesses once for each nnz. There are probably also some cache effects. Every value in C is written to once
    // double calc_bw;

    // calc_bw = (bytes_accessed / (exec_time * 0.001)) * 0.000000001;

    // printf("Application  bandwidth, GB/s:  %6.2f\n", calc_bw);

    // copy data from device to host
    dev_queue.memcpy(e, e_dev, MSZ * NSZ * sizeof(float)).wait();

    printf("checksum_e using esimd = %f\n", checksum(e, MSZ * NSZ));
    //
    validate(e_gold, e, MSZ * NSZ);

    // free host and device memory
    free(a_row_offsets);
    free(a_column_indices);
    free(a_values);
    free(c);
    free(b);
    free(d);
    free(e_gold);
    free(e);

    sycl::free(a_row_offsets_dev, dev_queue);
    sycl::free(a_column_indices_dev, dev_queue);
    sycl::free(c_dev, dev_queue);
    sycl::free(b_dev, dev_queue);
    sycl::free(d_dev, dev_queue);
    sycl::free(e_dev, dev_queue);

    return 0;
}
