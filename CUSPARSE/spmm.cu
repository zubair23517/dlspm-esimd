#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#ifdef OUTPUT_CSV_FILE
#include <fstream>
#endif

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
C =  A B
A is a sparse matrix of size MSZ X KSZ
B is a dense matrix of size  KSZ x NSZ
C is a dense matrix of size  MSZ x NSZ

*/

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

    int nDevices;
    float peak_bw;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);

        peak_bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        printf("  Peak Memory Bandwidth (GB/s): %6.2f\n\n", peak_bw);
    }

    // create a random sparse matrix with size m*n and initialize it with random values using rand() function
    int mnnz = (1.01 * MSZ * KSZ * SP) + 1000;
    float *a_values = (float *)malloc(mnnz * sizeof(float));
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
                a_values[a_row_offsets[i + 1]] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 0.01;
                a_row_offsets[i + 1]++;
            }
        }
    }

    int nnz = a_row_offsets[MSZ];
    printf("MSZ = %d KSZ = %d NSZ = %d SP = %f\n", MSZ, KSZ, NSZ, SP);
    printf("nnz = %d mnnz = %d\n", nnz, mnnz);

    float *b = (float *)malloc(KSZ * NSZ * sizeof(float));

    // create a random c array with size m*k and initialize it with random values using rand() function
    for (int i = 0; i < KSZ * NSZ; i++)
    {
        b[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 0.01;
    }
    float *c = (float *)malloc(MSZ * NSZ * sizeof(float));
    memset(c, 0, MSZ * NSZ * sizeof(float));

    Spmm(a_values, a_row_offsets, a_column_indices, b, c);
    // print checksum for a_values
    // printf("checksum_a_values = %f\n", checksum(c, MSZ * NSZ));

#if 1
    // device arrays and copy data from host to device
    int *a_row_offsets_dev;
    cudaMalloc((void **)&a_row_offsets_dev, (MSZ + 1) * sizeof(int));
    // copy data from host to device
    cudaMemcpy(a_row_offsets_dev, a_row_offsets, (MSZ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    int *a_column_indices_dev;
    cudaMalloc((void **)&a_column_indices_dev, nnz * sizeof(int));
    cudaMemcpy(a_column_indices_dev, a_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    float *a_values_dev;
    cudaMalloc((void **)&a_values_dev, nnz * sizeof(float));
    cudaMemcpy(a_values_dev, a_values, nnz * sizeof(float), cudaMemcpyHostToDevice);

    float *b_dev;
    cudaMalloc((void **)&b_dev, KSZ * NSZ * sizeof(float));
    cudaMemcpy(b_dev, b, KSZ * NSZ * sizeof(float), cudaMemcpyHostToDevice);

    float *c_dev;
    cudaMalloc((void **)&c_dev, MSZ * NSZ * sizeof(float));
    cudaMemset(c_dev, 0, MSZ * NSZ * sizeof(float));

    cusparseHandle_t handle = NULL;
    cusparseDnMatDescr_t matC, matB;
    cusparseSpMatDescr_t matA;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Create dense matrix C
    cusparseCreateDnMat(&matC, MSZ, NSZ, NSZ, c_dev,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Create dense matrix B
    cusparseCreateDnMat(&matB, KSZ, NSZ, NSZ, b_dev,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Create sparse matrix C in CSR format
    cusparseCreateCsr(&matA, MSZ, KSZ, nnz,
                      a_row_offsets_dev, a_column_indices_dev, a_values_dev,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

    // print buffer size
   // printf("sddmm_csr_example buffer size: %lld bytes\n", (long long)bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    float timeKernel;
    float exectime = 123456789.0;

    for (int loop = 0; loop < 10; ++loop)
    {
        cudaEventRecord(tstart);

        cusparseSpMM(handle,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                     CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

        cudaEventRecord(tstop);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&timeKernel, tstart, tstop);

        if (exectime > timeKernel)
            exectime = timeKernel;
    }

    printf("Kernel time in ms:  %.12f\n", exectime);
    float opcount = (2 * (float)nnz - MSZ) * NSZ;
    float gflops1 = (opcount) / (exectime * 1.0E+6);
    printf("GFLOPS:  %8.2f\n", gflops1);

    float *c_host;
    c_host = (float *)malloc(MSZ * NSZ * sizeof(float));
    cudaMemcpy(c_host, c_dev, MSZ * NSZ * sizeof(float), cudaMemcpyDeviceToHost);
    printf("checksum c values from CPU = %f\n", checksum(c, MSZ * NSZ));
    printf("checksum C values from GPU = %f\n", checksum(c_host, MSZ * NSZ));
    validate(c, c_host, MSZ * NSZ);

#ifdef OUTPUT_CSV_FILE
    std::ofstream csv_file;
    csv_file.open(OUTPUT_CSV_FILE, std::ios::app | std::ios::out);
    csv_file << MSZ << "," << KSZ << "," << NSZ << "," << SP << "," << gflops1 << std::endl;
    csv_file.close();
#endif

    cusparseDestroyDnMat(matC);
    cusparseDestroyDnMat(matB);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);

#endif
    // free cuda memory
    cudaFree(a_row_offsets_dev);
    cudaFree(a_column_indices_dev);
    cudaFree(a_values_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    cudaFree(dBuffer);

    // free host and device memory
    free(a_row_offsets);
    free(a_column_indices);
    free(a_values);
    free(c);
    free(b);

    return 0;
}
